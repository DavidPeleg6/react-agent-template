"""
Shared data models for the Smart Analysis Agent.
"""

# PostgreSQL-specific models
from .postgres import (
    PostgreSQLConnectionConfig, QueryResult, SchemaInfo, TableInfo
)

# Agent-specific models  
from .analysis_agent import (
    QueryRequest, QueryResponse, AgentConfig, OutputMode, LLMProvider,
    AVAILABLE_MODELS, DEFAULT_MODELS
)

__all__ = [
    # Query models
    'QueryRequest',
    'QueryResponse', 
    'OutputMode',
    
    # Agent configuration
    'AgentConfig',
    'LLMProvider',
    'AVAILABLE_MODELS',
    'DEFAULT_MODELS',
    
    # PostgreSQL configuration  
    'PostgreSQLConnectionConfig',
    'QueryResult',
    'SchemaInfo', 
    'TableInfo'
] 