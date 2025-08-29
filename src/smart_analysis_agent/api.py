"""
FastAPI server for Smart Analysis Agent.
Provides RESTful endpoints for domain-agnostic data analysis and insights.
"""

import asyncio
import os
import time
import json
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import uvicorn
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Import from data directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data import (
    QueryRequest, 
    QueryResponse, 
    AgentConfig, 
    PostgreSQLConnectionConfig,
    LLMProvider,
    DEFAULT_MODELS
)
from .react_agent import SmartAnalysisAgent

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global agent instance
_agent: Optional[SmartAnalysisAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    global _agent
    
    try:
        logger.info("🚀 Starting Smart Analysis Agent API server")
        
        # Create agent configuration from environment
        config = create_agent_config_from_env()
        
        # Initialize the agent
        _agent = SmartAnalysisAgent(config)
        logger.info("✅ Smart Analysis Agent initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error("❌ Failed to initialize Smart Analysis Agent", error=str(e))
        raise
    finally:
        # Cleanup
        if _agent:
            await _agent.close()
            logger.info("🔄 Smart Analysis Agent cleanup completed")


def create_agent_config_from_env() -> AgentConfig:
    """Create agent configuration from environment variables."""
    
    # PostgreSQL configuration
    postgres_config = PostgreSQLConnectionConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DATABASE", "analysis_db"),
        username=os.getenv("POSTGRES_USERNAME", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        schema_name=os.getenv("POSTGRES_SCHEMA", "public"),
        ssl_mode=os.getenv("POSTGRES_SSL_MODE", "prefer")
    )
    
    # Agent configuration
    config = AgentConfig(
        # LLM settings
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("LLM_MODEL", DEFAULT_MODELS[LLMProvider.ANTHROPIC]),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        
        # Database
        postgres_config=postgres_config,
        
        # Agent behavior
        max_query_iterations=int(os.getenv("MAX_QUERY_ITERATIONS", "10")),
        query_timeout_seconds=int(os.getenv("QUERY_TIMEOUT_SECONDS", "30")),
        max_result_rows=int(os.getenv("MAX_RESULT_ROWS", "1000")),
        graph_recursion_limit=int(os.getenv("GRAPH_RECURSION_LIMIT", "75")),
        
        # Logging
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_queries=os.getenv("LOG_QUERIES", "true").lower() == "true"
    )
    
    return config


# Create FastAPI app
app = FastAPI(
    title="Smart Analysis Agent API",
    description="Domain-agnostic intelligent analysis system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamingEvent(BaseModel):
    """Model for streaming events."""
    event_type: str
    timestamp: float
    message: str
    thread_id: Optional[str] = None
    step: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    agent_initialized: bool


class ConversationListResponse(BaseModel):
    """Response model for conversation listing."""
    conversations: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        agent_initialized=_agent is not None
    )


@app.get("/config")
async def get_config():
    """Get agent configuration information."""
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    config = _agent.config
    return {
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "graph_recursion_limit": config.graph_recursion_limit,
        "max_query_iterations": config.max_query_iterations,
        "query_timeout_seconds": config.query_timeout_seconds,
        "max_result_rows": config.max_result_rows,
        "database_host": config.postgres_config.host,
        "database_name": config.postgres_config.database,
        "database_schema": config.postgres_config.schema_name
    }


@app.post("/query", response_model=QueryResponse)
async def query_analysis(request: QueryRequest):
    """Execute an analysis query and return results."""
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info("🔍 Processing analysis query", 
                   question=request.question[:100], 
                   output_mode=request.output_mode.value)
        
        response = await _agent.query(request)
        
        logger.info("✅ Analysis query completed", 
                   success=response.success,
                   execution_time=response.total_execution_time_ms,
                   thread_id=response.thread_id)
        
        return response
        
    except Exception as e:
        logger.error("❌ Analysis query failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@app.post("/query/stream")
async def stream_analysis_query(request: QueryRequest):
    """Execute an analysis query with streaming progress updates."""
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    async def event_generator():
        """Generate server-sent events for query progress."""
        events = []
        
        async def progress_callback(event_data):
            """Callback to collect progress events."""
            event = StreamingEvent(**event_data)
            events.append(event)
            
            # Send event as SSE
            yield f"data: {event.model_dump_json()}\n\n"
        
        try:
            logger.info("🔍 Starting streaming analysis query", 
                       question=request.question[:100])
            
            # Execute query with progress callback
            response = await _agent.query(request, progress_callback=progress_callback)
            
            # Send final result
            final_event = StreamingEvent(
                event_type="final_result",
                timestamp=time.time(),
                message="Analysis complete",
                details=response.model_dump()
            )
            yield f"data: {final_event.model_dump_json()}\n\n"
            
        except Exception as e:
            logger.error("❌ Streaming query failed", error=str(e))
            error_event = StreamingEvent(
                event_type="error",
                timestamp=time.time(),
                message=f"Query failed: {str(e)}"
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/conversation/load/{thread_id}")
async def load_conversation(thread_id: str):
    """Load a conversation thread."""
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        success = await _agent.load_conversation(thread_id)
        return {
            "success": success,
            "thread_id": thread_id,
            "message": "Conversation loaded" if success else "New conversation started"
        }
    except Exception as e:
        logger.error("❌ Failed to load conversation", error=str(e), thread_id=thread_id)
        raise HTTPException(status_code=500, detail=f"Failed to load conversation: {str(e)}")


@app.get("/conversation/list", response_model=ConversationListResponse)
async def list_conversations(page: int = 1, page_size: int = 50):
    """List available conversations."""
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        if not _agent.conversation_manager:
            return ConversationListResponse(
                conversations=[],
                total_count=0,
                page=page,
                page_size=page_size
            )
        
        offset = (page - 1) * page_size
        conversations = await _agent.conversation_manager.list_conversations(
            limit=page_size, 
            offset=offset
        )
        
        # Convert to dict format
        conversation_dicts = [
            {
                "thread_id": conv.thread_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "message_count": conv.message_count,
                "last_message_at": conv.last_message_at.isoformat() if conv.last_message_at else None
            }
            for conv in conversations
        ]
        
        return ConversationListResponse(
            conversations=conversation_dicts,
            total_count=len(conversation_dicts),  # Note: This is approximate
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error("❌ Failed to list conversations", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@app.get("/conversation/{thread_id}/messages")
async def get_conversation_messages(thread_id: str):
    """Get messages for a conversation thread."""
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        if not _agent.conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        messages = await _agent.conversation_manager.get_conversation_messages(thread_id)
        
        return {
            "thread_id": thread_id,
            "messages": messages,
            "message_count": len(messages)
        }
        
    except Exception as e:
        logger.error("❌ Failed to get conversation messages", error=str(e), thread_id=thread_id)
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")


@app.delete("/conversation/{thread_id}")
async def delete_conversation(thread_id: str):
    """Delete a conversation thread."""
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        if not _agent.conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        success = await _agent.conversation_manager.delete_conversation(thread_id)
        
        return {
            "success": success,
            "thread_id": thread_id,
            "message": "Conversation deleted" if success else "Conversation not found"
        }
        
    except Exception as e:
        logger.error("❌ Failed to delete conversation", error=str(e), thread_id=thread_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "src.smart_analysis_agent.api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
