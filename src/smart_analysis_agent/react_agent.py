"""
Smart Analysis Agent using LangGraph's prebuilt ReAct agent.
A domain-agnostic intelligent analysis system that can be adapted to any field
requiring data analysis, reasoning, and insights.
"""

import asyncio
import time
import re
import csv
import io
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import structlog
import uuid
import asyncio
import sys

# Import from data directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data import (
    QueryRequest, 
    QueryResponse, 
    AgentConfig, 
    OutputMode,
    LLMProvider,
    DEFAULT_MODELS,
    PostgreSQLConnectionConfig
)

# Import from tools directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.postgres_mcp_server import POSTGRES_TOOLS, PostgreSQLMCPClient, create_postgres_tools_with_config
from .prompt_manager import get_prompt_manager
from .conversation_manager import ConversationManager

logger = structlog.get_logger(__name__)


class LLMProviderBase(ABC):
    """Base class for LLM provider implementations."""
    
    @abstractmethod
    def create_llm(self, config: AgentConfig, model: str):
        """Create and return the LLM instance for this provider."""
        pass
    
    @abstractmethod
    def get_api_key(self, config: AgentConfig) -> Optional[str]:
        """Get the API key for this provider from config."""
        pass
    
    @abstractmethod
    def validate_api_key(self, config: AgentConfig) -> None:
        """Validate that the API key is available."""
        pass


class GoogleProvider(LLMProviderBase):
    """Google LLM provider implementation."""
    
    def get_api_key(self, config: AgentConfig) -> Optional[str]:
        return config.google_api_key
    
    def validate_api_key(self, config: AgentConfig) -> None:
        if not self.get_api_key(config):
            raise ValueError("Google API key not provided in configuration.")
    
    def create_llm(self, config: AgentConfig, model: str):
        self.validate_api_key(config)
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=self.get_api_key(config)
        )


class AnthropicProvider(LLMProviderBase):
    """Anthropic LLM provider implementation."""
    
    def get_api_key(self, config: AgentConfig) -> Optional[str]:
        return config.anthropic_api_key
    
    def validate_api_key(self, config: AgentConfig) -> None:
        if not self.get_api_key(config):
            raise ValueError("Anthropic API key not provided in configuration.")
    
    def create_llm(self, config: AgentConfig, model: str):
        self.validate_api_key(config)
        return ChatAnthropic(
            model=model,
            api_key=self.get_api_key(config),
            timeout=30.0,
            stop=None
        )


class OpenAIProvider(LLMProviderBase):
    """OpenAI LLM provider implementation."""
    
    def get_api_key(self, config: AgentConfig) -> Optional[str]:
        return config.openai_api_key
    
    def validate_api_key(self, config: AgentConfig) -> None:
        if not self.get_api_key(config):
            raise ValueError("OpenAI API key not provided in configuration.")
    
    def create_llm(self, config: AgentConfig, model: str):
        self.validate_api_key(config)
        return ChatOpenAI(
            model=model,
            api_key=self.get_api_key(config)
        )


class LLMFactory:
    """Factory class for creating LLM instances."""
    
    _providers = {
        "google": GoogleProvider(),
        "anthropic": AnthropicProvider(),
        "openai": OpenAIProvider(),
    }
    
    @classmethod
    def create_llm(cls, config: AgentConfig, provider: str, model: str):
        """Create an LLM instance for the given provider and model."""
        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        provider_instance = cls._providers[provider]
        return provider_instance.create_llm(config, model)


class SmartAnalysisAgent:
    """Smart Analysis Agent using LangGraph's prebuilt ReAct agent."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = self._create_llm()  # Default LLM
        self.postgres_config = config.postgres_config
        self.mcp_client = None  # Will be initialized when needed
        self.checkpointer = None  # Will be initialized async
        self.agent = None  # Will be initialized async
        self.current_thread_id: Optional[str] = None  # Track current conversation
        self.database_url = os.getenv("DATABASE_URL")
        self._checkpointer_context = None  # For proper cleanup
        
    async def _initialize_checkpointer(self):
        """Initialize the PostgreSQL checkpointer with proper error handling."""
        if not self.database_url:
            logger.warning("No DATABASE_URL provided, falling back to InMemorySaver")
            return InMemorySaver()
        
        try:
            # Fix Windows asyncio event loop compatibility
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            # Create the async context manager
            self._checkpointer_context = AsyncPostgresSaver.from_conn_string(self.database_url)
            
            # Enter the context to get the actual saver
            checkpointer = await self._checkpointer_context.__aenter__()
            
            # Set up the schema - LangGraph handles existing tables gracefully
            await checkpointer.setup()
            
            logger.info("PostgreSQL checkpointer schema setup completed")
            logger.info("PostgreSQL checkpointer initialized successfully")
            return checkpointer
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}")
            logger.info("Falling back to InMemorySaver")
            return InMemorySaver()

    async def _ensure_agent_initialized(self):
        """Ensure the agent is initialized with checkpointer."""
        if self.agent is None:
            self.checkpointer = await self._initialize_checkpointer()
            
            # Initialize conversation manager if using PostgreSQL
            if isinstance(self.checkpointer, AsyncPostgresSaver) and self.database_url:
                self.conversation_manager = ConversationManager(self.database_url)
            else:
                self.conversation_manager = None
                
            self.agent = self._create_agent()
        
    def _create_llm(self, provider: Optional[str] = None, model: Optional[str] = None):
        """Create the LLM based on configuration or request parameters."""
        # Use provided parameters or fall back to config defaults
        provider = provider or self.config.llm_provider
        model = model or self.config.llm_model
        
        # Use the factory to create the LLM
        return LLMFactory.create_llm(self.config, provider, model)
    
    def _create_agent(self, llm=None, tools: Optional[List] = None):
        """Create the ReAct agent with PostgreSQL and analysis tools."""
        logger.info("🔧 Creating ReAct agent with PostgreSQL and analysis tools")
        
        # Use provided LLM or default
        current_llm = llm or self.llm
        
        if tools is None:
            # Create pre-configured tools that have the PostgreSQL config baked in
            from tools.postgres_mcp_server import create_postgres_tools_with_config
            from tools.analysis_mcp_server import create_analysis_tools_with_config
            
            # Create PostgreSQL tools with embedded configuration
            postgres_tools = create_postgres_tools_with_config(self.postgres_config)
            
            # Create analysis tools with embedded configuration
            analysis_tools = create_analysis_tools_with_config(self.postgres_config)
            
            agent_tools = postgres_tools + analysis_tools
        else:
            agent_tools = tools
        
        # Import ToolNode for custom configuration handling
        from langgraph.prebuilt import ToolNode
        
        # Create a custom ToolNode that properly handles configuration
        # This ensures that config is passed to tools with config parameters
        tool_node = ToolNode(
            agent_tools,
            handle_tool_errors=True  # Enable error handling
        )
        
        # Create ReAct agent with custom tool node instead of raw tools
        agent = create_react_agent(
            current_llm, 
            tools=tool_node,  # Pass ToolNode instead of raw tools list
            checkpointer=self.checkpointer
        )
        
        logger.info("✅ ReAct agent created successfully", 
                   tool_count=len(agent_tools))
        return agent

    async def query(self, request: QueryRequest, thread_id: Optional[str] = None, progress_callback: Optional[callable] = None) -> QueryResponse:
        """Process an analysis query and return results."""
        start_time = time.time()
        
        # Set current thread ID for conversation tracking
        self.current_thread_id = thread_id or str(uuid.uuid4())
        
        logger.info("🚀 Starting analysis query", 
                   question=request.question[:100],
                   output_mode=request.output_mode.value,
                   thread_id=self.current_thread_id)
        
        # Emit initial progress event
        if progress_callback:
            await progress_callback({
                "event_type": "query_start",
                "timestamp": time.time(),
                "message": f"🚀 Starting analysis: {request.question[:100]}...",
                "thread_id": self.current_thread_id,
                "step": "initialization"
            })
        
        try:
            # Emit progress: Agent initialization
            if progress_callback:
                await progress_callback({
                    "event_type": "agent_initialization",
                    "timestamp": time.time(),
                    "message": "🔧 Initializing agent and database connections",
                    "thread_id": self.current_thread_id,
                    "step": "initialization"
                })
            
            # Ensure agent is initialized
            await self._ensure_agent_initialized()
            
            # Emit progress: Conversation setup
            if progress_callback:
                await progress_callback({
                    "event_type": "conversation_setup",
                    "timestamp": time.time(),
                    "message": "💾 Setting up conversation thread",
                    "thread_id": self.current_thread_id,
                    "step": "conversation_setup"
                })
            
            # Save conversation start if conversation manager is available
            if self.conversation_manager:
                await self.conversation_manager.create_or_update_conversation(
                    thread_id=self.current_thread_id,
                    title=request.question[:50] + "..." if len(request.question) > 50 else request.question,
                    increment_message_count=True
                )
                logger.info("💾 Conversation saved", thread_id=self.current_thread_id)
            
            # Emit progress: Model selection
            if progress_callback:
                provider = request.llm_provider.value if request.llm_provider else self.config.llm_provider
                model = request.llm_model or self.config.llm_model
                await progress_callback({
                    "event_type": "model_selection",
                    "timestamp": time.time(),
                    "message": f"🤖 Using {provider} model: {model}",
                    "thread_id": self.current_thread_id,
                    "step": "model_selection",
                    "provider": provider,
                    "model": model
                })
            
            # Override LLM if specified in request
            if request.llm_provider or request.llm_model:
                provider = request.llm_provider.value if request.llm_provider else self.config.llm_provider
                model = request.llm_model or self.config.llm_model
                custom_llm = self._create_llm(provider, model)
                agent = self._create_agent(llm=custom_llm)
            else:
                agent = self.agent
                
            # Emit progress: Tool configuration
            if progress_callback:
                await progress_callback({
                    "event_type": "tool_configuration",
                    "timestamp": time.time(),
                    "message": "🔧 Configuring tools (Database Query and Analysis)",
                    "thread_id": self.current_thread_id,
                    "step": "tool_configuration"
                })
            
            # Create configuration dict for the agent with PostgreSQL config
            config = {
                "configurable": {
                    "thread_id": self.current_thread_id,
                    "postgres_config": self.postgres_config,  # Pass PostgreSQL config to tools
                    "progress_callback": progress_callback,
                    "recursion_limit": request.graph_recursion_limit or self.config.graph_recursion_limit
                },
                # Also pass postgres_config at the top level for tool access
                "postgres_config": self.postgres_config,
                "progress_callback": progress_callback
            }
            
            # Emit progress: Prompt generation
            if progress_callback:
                await progress_callback({
                    "event_type": "prompt_generation",
                    "timestamp": time.time(),
                    "message": "📝 Generating system prompt and analysis strategy",
                    "thread_id": self.current_thread_id,
                    "step": "prompt_generation"
                })
            
            # Get system prompt for analysis
            prompt_manager = get_prompt_manager()
            request_context = {
                "domain_focus": request.domain_focus,
                "timeframe": request.timeframe,
                "include_analysis": request.include_analysis,
            }
            system_prompt = await prompt_manager.get_analysis_prompt(request, request_context)
            
            # Create the full prompt with user question
            full_prompt = f"{system_prompt}\n\nUser Question: {request.question}"
            
            # Emit progress: Agent execution start
            if progress_callback:
                await progress_callback({
                    "event_type": "agent_execution_start",
                    "timestamp": time.time(),
                    "message": "🚀 Starting agent execution with ReAct reasoning",
                    "thread_id": self.current_thread_id,
                    "step": "agent_execution"
                })
            
            # Execute the agent
            logger.info("🤖 Executing smart analysis agent")
            message = HumanMessage(content=full_prompt)
            
            if agent is None:
                raise ValueError("Agent not initialized")
            
            # Create a wrapper to emit progress during agent execution
            async def progress_wrapper():
                if progress_callback:
                    await progress_callback({
                        "event_type": "agent_thinking",
                        "timestamp": time.time(),
                        "message": "🤔 Agent is analyzing your question and planning tool usage",
                        "thread_id": self.current_thread_id,
                        "step": "agent_thinking"
                    })
            
            # Start progress monitoring
            await progress_wrapper()
            
            response = await agent.ainvoke(
                {"messages": [message]}, 
                config=config  # type: ignore
            )
            
            # Emit progress: Agent execution complete
            if progress_callback:
                await progress_callback({
                    "event_type": "agent_execution_complete",
                    "timestamp": time.time(),
                    "message": "✅ Agent execution completed, processing results",
                    "thread_id": self.current_thread_id,
                    "step": "result_processing"
                })
            
            # Extract final message from agent response
            final_message = response["messages"][-1].content if response["messages"] else "No response generated"
            
            # Emit progress: Conversation update
            if progress_callback:
                await progress_callback({
                    "event_type": "conversation_update",
                    "timestamp": time.time(),
                    "message": "💾 Updating conversation history",
                    "thread_id": self.current_thread_id,
                    "step": "conversation_update"
                })
            
            # Update conversation with response if conversation manager is available
            if self.conversation_manager:
                await self.conversation_manager.create_or_update_conversation(
                    thread_id=self.current_thread_id,
                    increment_message_count=True
                )
                logger.info("💾 Conversation updated with response", thread_id=self.current_thread_id)
            
            # Emit progress: Final processing
            if progress_callback:
                await progress_callback({
                    "event_type": "final_processing",
                    "timestamp": time.time(),
                    "message": f"📊 Processing results for {request.output_mode.value} format",
                    "thread_id": self.current_thread_id,
                    "step": "final_processing"
                })
            
            logger.info("✅ Agent execution completed", 
                       message_length=len(final_message),
                       execution_time=time.time() - start_time)
            
            # Emit progress: Query complete
            if progress_callback:
                await progress_callback({
                    "event_type": "query_complete",
                    "timestamp": time.time(),
                    "message": "🎉 Analysis complete! Results ready.",
                    "thread_id": self.current_thread_id,
                    "step": "complete",
                    "execution_time": time.time() - start_time
                })
            
            # Process the result based on output mode
            return await self._process_result(request, final_message, start_time)
            
        except Exception as e:
            logger.error("❌ Query execution failed", error=str(e))
            
            # Emit error progress event
            if progress_callback:
                await progress_callback({
                    "event_type": "query_error",
                    "timestamp": time.time(),
                    "message": f"❌ Analysis failed: {str(e)}",
                    "thread_id": self.current_thread_id,
                    "step": "error",
                    "error": str(e)
                })
            
            return QueryResponse(
                success=False,
                output_mode=request.output_mode,
                question=request.question,
                error_message=str(e),
                total_execution_time_ms=(time.time() - start_time) * 1000,
                thread_id=self.current_thread_id,
                llm_provider=request.llm_provider.value if request.llm_provider else self.config.llm_provider,
                llm_model=request.llm_model or self.config.llm_model,
                graph_recursion_limit=request.graph_recursion_limit or self.config.graph_recursion_limit
            )
    
    async def _process_result(self, request: QueryRequest, final_message: str, start_time: float) -> QueryResponse:
        """Process the agent's response based on the requested output mode."""
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Base response data - simplified to only support answer mode
        response_data = {
            "success": True,
            "output_mode": request.output_mode,
            "question": request.question,
            "total_execution_time_ms": execution_time_ms,
            "thread_id": self.current_thread_id,
            "llm_provider": request.llm_provider.value if request.llm_provider else self.config.llm_provider,
            "llm_model": request.llm_model or self.config.llm_model,
            "graph_recursion_limit": request.graph_recursion_limit or self.config.graph_recursion_limit,
            "queries_executed": [],  # Could be populated from tool calls if needed
            "iterations": 1,
            "answer": final_message  # Always return as answer
        }
        
        return QueryResponse(**response_data)
    
    def _get_mcp_client(self) -> PostgreSQLMCPClient:
        """Get or create PostgreSQL MCP client."""
        if not self.mcp_client:
            self.mcp_client = PostgreSQLMCPClient(self.postgres_config)
            logger.info("Created PostgreSQL MCP client", host=self.postgres_config.host)
        return self.mcp_client
    
    async def load_conversation(self, thread_id: str) -> bool:
        """Load a conversation thread for continued analysis."""
        try:
            await self._ensure_agent_initialized()
            
            if not self.conversation_manager:
                logger.warning("Conversation manager not available (using in-memory storage)")
                return False
            
            # Set current thread for future operations
            self.current_thread_id = thread_id
            
            # Check if conversation exists
            checkpoints = await self.get_conversation_checkpoints(thread_id)
            conversation_exists = len(checkpoints) > 0
            
            if conversation_exists:
                logger.info(f"Loaded conversation thread: {thread_id} with {len(checkpoints)} checkpoints")
            else:
                logger.info(f"Started new conversation thread: {thread_id}")
            
            return conversation_exists
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False
    
    async def get_conversation_checkpoints(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get checkpoints for a conversation thread."""
        try:
            await self._ensure_agent_initialized()
            
            if not isinstance(self.checkpointer, AsyncPostgresSaver):
                return []
            
            target_thread_id = thread_id or self.current_thread_id
            if not target_thread_id:
                return []
            
            # Get checkpoints from LangGraph's PostgreSQL saver
            checkpoints = []
            async for checkpoint in self.checkpointer.alist({"configurable": {"thread_id": target_thread_id}}):
                checkpoints.append({
                    "checkpoint_id": checkpoint.config["configurable"]["checkpoint_id"],
                    "thread_id": target_thread_id,
                    "timestamp": checkpoint.metadata.get("timestamp", time.time())
                })
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to get conversation checkpoints: {e}")
            return []
    
    async def rewind_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rewind conversation to a specific checkpoint."""
        try:
            await self._ensure_agent_initialized()
            
            if not isinstance(self.checkpointer, AsyncPostgresSaver):
                logger.warning("Checkpoint rewinding requires PostgreSQL storage")
                return False
            
            # This would require implementing checkpoint rewinding logic
            # For now, just log the attempt
            logger.info(f"Checkpoint rewind requested: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rewind to checkpoint: {e}")
            return False
    
    def get_current_thread_id(self) -> Optional[str]:
        """Get the current conversation thread ID."""
        return self.current_thread_id
    
    def get_current_task_id(self) -> Optional[str]:
        """Get the current task ID (if any)."""
        return None  # Could be implemented if task tracking is needed
    
    async def close(self):
        """Clean up resources."""
        try:
            if self._checkpointer_context:
                await self._checkpointer_context.__aexit__(None, None, None)
                logger.info("PostgreSQL checkpointer context closed")
                
            if self.mcp_client:
                await self.mcp_client.close()
                logger.info("PostgreSQL MCP client closed")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Alias for backward compatibility
SimpleTrinoAgent = SmartAnalysisAgent
