"""
Pydantic models for the Smart Analysis Agent.
"""

import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

from .postgres import QueryResult, SchemaInfo, PostgreSQLConnectionConfig


class OutputMode(str, Enum):
    """Output modes for the agent."""
    ANSWER_QUESTION = "answer_question"


class LLMProvider(str, Enum):
    """Available LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


class QueryRequest(BaseModel):
    """Request model for querying and analyzing data."""
    
    question: str = Field(..., description="Natural language question about the data")
    output_mode: OutputMode = Field(
        default=OutputMode.ANSWER_QUESTION,
        description="How to return the results"
    )
    
    # LLM Selection
    llm_provider: Optional[LLMProvider] = Field(
        default=None,
        description="LLM provider to use (optional, defaults to server config)"
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="LLM model to use (optional, defaults to server config)"
    )
    
    # Graph Configuration
    graph_recursion_limit: Optional[int] = Field(
        default=None,
        ge=5,
        le=200,
        description="LangGraph recursion limit for this query (defaults to server config: 75)"
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of analysis iterations"
    )
    
    # Analysis scope
    domain_focus: Optional[List[str]] = Field(
        default=None,
        description="Specific domain areas to focus on (e.g., ['trends', 'performance'])"
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Time period for analysis (1h, 24h, 7d, 30d, etc.)"
    )
    include_analysis: bool = Field(
        default=True,
        description="Whether to include detailed analytical insights"
    )
    
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation thread ID for memory (optional)"
    )
    query_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this query (optional, auto-generated if not provided)"
    )


class QueryResponse(BaseModel):
    """Response model for data analysis queries."""
    
    # Basic response info
    success: bool = Field(..., description="Whether the query succeeded")
    output_mode: OutputMode = Field(..., description="Output mode used")
    question: str = Field(..., description="Original question")
    
    # Response content
    answer: Optional[str] = Field(default=None, description="Natural language answer")
    analysis_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured analysis data")
    
    # Query execution details
    total_execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    queries_executed: List[str] = Field(default_factory=list, description="Database queries executed")
    iterations: int = Field(default=1, description="Number of agent iterations")
    
    # Error handling
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # LLM information
    llm_provider: Optional[str] = Field(default=None, description="LLM provider used")
    llm_model: Optional[str] = Field(default=None, description="LLM model used")
    
    # Memory and conversation
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")
    query_id: Optional[str] = Field(default=None, description="Unique query identifier")
    
    # Graph execution details
    graph_recursion_limit: Optional[int] = Field(default=None, description="Recursion limit used")
    
    # Analysis metadata
    focus_areas_analyzed: Optional[List[str]] = Field(default=None, description="Domain areas that were analyzed")
    data_sources: Optional[Dict[str, Any]] = Field(default=None, description="Data sources used in analysis")
    
    # Timestamps
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")


# Available models for each provider
AVAILABLE_MODELS = {
    LLMProvider.ANTHROPIC: [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229", 
        "claude-3-opus-20240229",
        "claude-sonnet-4-20250514"
    ],
    LLMProvider.OPENAI: [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo", 
        "gpt-4o"
    ],
    LLMProvider.GOOGLE: [
        "gemini-pro",
        "gemini-2.5-flash",
        "gemini-1.5-pro"
    ]
}

# Default model for each provider
DEFAULT_MODELS = {
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.GOOGLE: "gemini-2.5-flash"
}


class AgentConfig(BaseModel):
    """Configuration for the Smart Analysis Agent."""
    
    # LLM Configuration
    llm_provider: str = Field(default="anthropic", description="LLM provider (openai, anthropic, google)")
    llm_model: str = Field(default="claude-sonnet-4-20250514", description="LLM model name")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key (for Gemini)")
    
    # Database Configuration
    postgres_config: PostgreSQLConnectionConfig = Field(..., description="PostgreSQL database configuration")
    
    # Agent Behavior
    max_query_iterations: int = Field(default=10, ge=1, le=50)
    query_timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_result_rows: int = Field(default=1000, ge=1, le=100000)
    graph_recursion_limit: int = Field(default=75, ge=5, le=200, description="LangGraph recursion limit for complex queries")
    enable_detailed_analysis: bool = Field(default=True)
    enable_pattern_recognition: bool = Field(default=True)
    
    # Analysis Configuration
    default_timeframe: str = Field(default="30d", description="Default analysis timeframe")
    max_concurrent_queries: int = Field(default=5, description="Max concurrent database queries")
    
    # Caching and Performance
    cache_analysis_results_minutes: int = Field(default=5, description="Minutes to cache analysis results")
    max_concurrent_operations: int = Field(default=5, description="Max concurrent operations")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_queries: bool = Field(default=True, description="Whether to log executed queries")
    
    class Config:
        env_prefix = "SMART_ANALYSIS_"
