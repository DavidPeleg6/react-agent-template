"""
MCP (Model Context Protocol) tools integration for PostgreSQL database access.
Generic database querying and schema exploration tools.
"""

import asyncio
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
import structlog
import asyncpg
from datetime import datetime
from decimal import Decimal
import inspect

# Import from data directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import (
    QueryResult, SchemaInfo, TableInfo, PostgreSQLConnectionConfig
)

logger = structlog.get_logger(__name__)


# Progress event class for MCP tools
class ProgressEvent(BaseModel):
    timestamp: float
    event_type: str  # "start", "sql_query", "thinking", "tool_call", "complete", "error"
    message: str
    details: Optional[Dict[str, Any]] = None


async def emit_progress_event(config: Optional[RunnableConfig], event: ProgressEvent):
    """Emit a progress event if callback is available."""
    try:
        if config and isinstance(config, dict):
            configurable = config.get('configurable', {})
            if configurable and isinstance(configurable, dict):
                progress_callback = configurable.get('progress_callback')
                if progress_callback and callable(progress_callback):
                    # Handle both sync and async callbacks
                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback(event)
                    else:
                        progress_callback(event)
    except Exception as e:
        # Don't let progress events break the main flow
        logger.debug(f"Progress event error: {e}")
        pass


class PostgreSQLMCPClient:
    """Client for interacting with PostgreSQL database for generic data analysis."""
    
    def __init__(self, config: PostgreSQLConnectionConfig):
        self.config = config
        self._connection_pool = None
        logger.info("🔧 Initializing PostgreSQLMCPClient", 
                   host=config.host, 
                   port=config.port,
                   database=config.database,
                   schema=config.schema_name)
        
    async def _get_connection_pool(self):
        """Get or create a connection pool."""
        if not self._connection_pool:
            logger.info("🔌 Creating PostgreSQL connection pool")
            try:
                self._connection_pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    ssl=self.config.ssl_mode if self.config.ssl_mode != 'disable' else False,
                    min_size=2,
                    max_size=10
                )
                logger.info("✅ PostgreSQL connection pool created successfully")
            except Exception as e:
                logger.error("❌ Failed to create PostgreSQL connection pool", error=str(e))
                raise
        return self._connection_pool
    
    async def execute_query(self, query: str, params: Optional[List] = None, limit: Optional[int] = None) -> QueryResult:
        """Execute a query and return results."""
        start_time = time.time()
        logger.info("🔍 Executing PostgreSQL query", 
                   query=query[:100] + "..." if len(query) > 100 else query,
                   params=params,
                   limit=limit)
        
        try:
            pool = await self._get_connection_pool()
            
            async with pool.acquire() as connection:
                # Apply limit if specified
                if limit and not any(keyword in query.upper() for keyword in ['LIMIT', 'TOP']):
                    query = f"{query.rstrip(';')} LIMIT {limit}"
                
                # Execute query
                if params:
                    rows = await connection.fetch(query, *params)
                else:
                    rows = await connection.fetch(query)
                
                # Convert to list of dictionaries
                data = [dict(row) for row in rows]
                
                execution_time = (time.time() - start_time) * 1000
                
                logger.info("✅ Query executed successfully", 
                           rows_returned=len(data),
                           execution_time_ms=execution_time)
                
                return QueryResult(
                    success=True,
                    data=data,
                    query=query,
                    execution_time_ms=execution_time,
                    row_count=len(data)
                )
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Query execution failed: {str(e)}"
            
            logger.error("❌ Query execution failed", 
                        error=str(e),
                        query=query[:100],
                        execution_time_ms=execution_time)
            
            return QueryResult(
                success=False,
                error=error_msg,
                query=query,
                execution_time_ms=execution_time,
                row_count=0
            )
    
    async def get_schema_info(self, table_name: Optional[str] = None) -> SchemaInfo:
        """Get schema information for tables."""
        try:
            if table_name:
                # Get detailed info for specific table
                query = """
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length
                    FROM information_schema.columns 
                    WHERE table_schema = $1 AND table_name = $2
                    ORDER BY ordinal_position
                """
                result = await self.execute_query(query, [self.config.schema_name, table_name])
                
                if result.success and result.data:
                    return SchemaInfo(
                        success=True,
                        tables=[TableInfo(
                            table_name=table_name,
                            columns=result.data
                        )]
                    )
                else:
                    return SchemaInfo(
                        success=False,
                        error=f"Table '{table_name}' not found or no columns"
                    )
            else:
                # Get list of all tables
                query = """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = $1 AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """
                result = await self.execute_query(query, [self.config.schema_name])
                
                if result.success:
                    tables = []
                    for row in result.data:
                        tables.append(TableInfo(
                            table_name=row['table_name'],
                            columns=[]
                        ))
                    
                    return SchemaInfo(success=True, tables=tables)
                else:
                    return SchemaInfo(success=False, error=result.error)
                    
        except Exception as e:
            logger.error("❌ Failed to get schema info", error=str(e))
            return SchemaInfo(success=False, error=str(e))
    
    async def close(self):
        """Close the connection pool."""
        if self._connection_pool:
            await self._connection_pool.close()
            logger.info("🔄 PostgreSQL connection pool closed")


# Standalone tools (original interface)
@tool
async def execute_postgres_query(
    query: str,
    limit: Optional[int] = 100,
    config: Optional[RunnableConfig] = None
) -> str:
    """
    Execute a SQL query against the PostgreSQL database.
    Returns query results in a formatted table or error message.
    """
    try:
        await emit_progress_event(config, ProgressEvent(
            timestamp=time.time(),
            event_type="sql_query",
            message=f"🔍 Executing SQL query: {query[:50]}{'...' if len(query) > 50 else ''}"
        ))
        
        # Get PostgreSQL config from context
        postgres_config = config.get("postgres_config") if config else None
        if not postgres_config:
            raise ValueError("PostgreSQL configuration not found in context")
        
        client = PostgreSQLMCPClient(postgres_config)
        result = await client.execute_query(query, limit=limit)
        await client.close()
        
        if not result.success:
            return f"❌ Query failed: {result.error}"
        
        if not result.data:
            return "✅ Query executed successfully but returned no data"
        
        # Format results as a table
        df = pd.DataFrame(result.data)
        
        # Limit displayed columns and rows for readability
        max_cols = 10
        max_rows = min(50, len(df))
        
        if len(df.columns) > max_cols:
            display_df = df.iloc[:max_rows, :max_cols]
            col_msg = f" (showing first {max_cols} of {len(df.columns)} columns)"
        else:
            display_df = df.iloc[:max_rows]
            col_msg = ""
        
        row_msg = f" (showing first {max_rows} of {len(df)} rows)" if len(df) > max_rows else ""
        
        return f"✅ Query Results{col_msg}{row_msg}:\n\n{display_df.to_string(index=False)}\n\n📊 Total rows: {len(df)} | Execution time: {result.execution_time_ms:.1f}ms"
        
    except Exception as e:
        error_msg = f"❌ Failed to execute query: {str(e)}"
        logger.error("Execute query tool failed", error=str(e))
        return error_msg


@tool
async def describe_postgres_table(
    table_name: str,
    config: Optional[RunnableConfig] = None
) -> str:
    """
    Get detailed schema information for a specific PostgreSQL table.
    Returns column names, data types, constraints, and other metadata.
    """
    try:
        await emit_progress_event(config, ProgressEvent(
            timestamp=time.time(),
            event_type="tool_call",
            message=f"📋 Describing table: {table_name}"
        ))
        
        # Get PostgreSQL config from context
        postgres_config = config.get("postgres_config") if config else None
        if not postgres_config:
            raise ValueError("PostgreSQL configuration not found in context")
        
        client = PostgreSQLMCPClient(postgres_config)
        schema_info = await client.get_schema_info(table_name)
        await client.close()
        
        if not schema_info.success:
            return f"❌ Failed to describe table '{table_name}': {schema_info.error}"
        
        if not schema_info.tables or not schema_info.tables[0].columns:
            return f"❌ Table '{table_name}' not found or has no columns"
        
        # Format schema information
        table_info = schema_info.tables[0]
        result = f"📋 Table: {table_name}\n\n"
        result += "Columns:\n"
        
        for col in table_info.columns:
            col_name = col.get('column_name', 'Unknown')
            data_type = col.get('data_type', 'Unknown')
            nullable = "NULL" if col.get('is_nullable') == 'YES' else "NOT NULL"
            default = col.get('column_default')
            max_length = col.get('character_maximum_length')
            
            type_info = data_type
            if max_length:
                type_info += f"({max_length})"
            
            result += f"  • {col_name}: {type_info} {nullable}"
            if default:
                result += f" DEFAULT {default}"
            result += "\n"
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Failed to describe table: {str(e)}"
        logger.error("Describe table tool failed", error=str(e))
        return error_msg


@tool
async def list_postgres_tables(
    config: Optional[RunnableConfig] = None
) -> str:
    """
    List all tables available in the PostgreSQL database.
    Returns table names and basic information.
    """
    try:
        await emit_progress_event(config, ProgressEvent(
            timestamp=time.time(),
            event_type="tool_call",
            message="📑 Listing database tables"
        ))
        
        # Get PostgreSQL config from context
        postgres_config = config.get("postgres_config") if config else None
        if not postgres_config:
            raise ValueError("PostgreSQL configuration not found in context")
        
        client = PostgreSQLMCPClient(postgres_config)
        schema_info = await client.get_schema_info()
        await client.close()
        
        if not schema_info.success:
            return f"❌ Failed to list tables: {schema_info.error}"
        
        if not schema_info.tables:
            return f"📑 No tables found in schema '{postgres_config.schema_name}'"
        
        # Format table list
        result = f"📑 Tables in schema '{postgres_config.schema_name}':\n\n"
        for table in schema_info.tables:
            result += f"  • {table.table_name}\n"
        
        result += f"\n📊 Total tables: {len(schema_info.tables)}"
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Failed to list tables: {str(e)}"
        logger.error("List tables tool failed", error=str(e))
        return error_msg


# Create tools with embedded PostgreSQL configuration
def create_postgres_tools_with_config(postgres_config: PostgreSQLConnectionConfig):
    """Create PostgreSQL tools with embedded configuration."""
    
    @tool
    async def execute_postgres_query_preconfigured(
        query: str,
        limit: Optional[int] = 100
    ) -> str:
        """
        Execute a SQL query against the PostgreSQL database.
        Returns query results in a formatted table or error message.
        """
        try:
            client = PostgreSQLMCPClient(postgres_config)
            result = await client.execute_query(query, limit=limit)
            await client.close()
            
            if not result.success:
                return f"❌ Query failed: {result.error}"
            
            if not result.data:
                return "✅ Query executed successfully but returned no data"
            
            # Format results as a table
            df = pd.DataFrame(result.data)
            
            # Limit displayed columns and rows for readability
            max_cols = 10
            max_rows = min(50, len(df))
            
            if len(df.columns) > max_cols:
                display_df = df.iloc[:max_rows, :max_cols]
                col_msg = f" (showing first {max_cols} of {len(df.columns)} columns)"
            else:
                display_df = df.iloc[:max_rows]
                col_msg = ""
            
            row_msg = f" (showing first {max_rows} of {len(df)} rows)" if len(df) > max_rows else ""
            
            return f"✅ Query Results{col_msg}{row_msg}:\n\n{display_df.to_string(index=False)}\n\n📊 Total rows: {len(df)} | Execution time: {result.execution_time_ms:.1f}ms"
            
        except Exception as e:
            logger.error(f"❌ Execute query tool failed: {e}")
            return f"❌ Failed to execute query: {str(e)}"
    
    @tool
    async def describe_postgres_table_preconfigured(
        table_name: str
    ) -> str:
        """
        Get detailed schema information for a specific PostgreSQL table.
        Returns column names, data types, constraints, and other metadata.
        """
        try:
            client = PostgreSQLMCPClient(postgres_config)
            schema_info = await client.get_schema_info(table_name)
            await client.close()
            
            if not schema_info.success:
                return f"❌ Failed to describe table '{table_name}': {schema_info.error}"
            
            if not schema_info.tables or not schema_info.tables[0].columns:
                return f"❌ Table '{table_name}' not found or has no columns"
            
            # Format schema information
            table_info = schema_info.tables[0]
            result = f"📋 Table: {table_name}\n\n"
            result += "Columns:\n"
            
            for col in table_info.columns:
                col_name = col.get('column_name', 'Unknown')
                data_type = col.get('data_type', 'Unknown')
                nullable = "NULL" if col.get('is_nullable') == 'YES' else "NOT NULL"
                default = col.get('column_default')
                max_length = col.get('character_maximum_length')
                
                type_info = data_type
                if max_length:
                    type_info += f"({max_length})"
                
                result += f"  • {col_name}: {type_info} {nullable}"
                if default:
                    result += f" DEFAULT {default}"
                result += "\n"
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Describe table tool failed: {e}")
            return f"❌ Failed to describe table: {str(e)}"
    
    @tool
    async def list_postgres_tables_preconfigured() -> str:
        """
        List all tables available in the PostgreSQL database.
        Returns table names and basic information.
        """
        try:
            client = PostgreSQLMCPClient(postgres_config)
            schema_info = await client.get_schema_info()
            await client.close()
            
            if not schema_info.success:
                return f"❌ Failed to list tables: {schema_info.error}"
            
            if not schema_info.tables:
                return f"📑 No tables found in schema '{postgres_config.schema_name}'"
            
            # Format table list
            result = f"📑 Tables in schema '{postgres_config.schema_name}':\n\n"
            for table in schema_info.tables:
                result += f"  • {table.table_name}\n"
            
            result += f"\n📊 Total tables: {len(schema_info.tables)}"
            
            return result
            
        except Exception as e:
            logger.error(f"❌ List tables tool failed: {e}")
            return f"❌ Failed to list tables: {str(e)}"
    
    return [
        execute_postgres_query_preconfigured,
        describe_postgres_table_preconfigured,
        list_postgres_tables_preconfigured
    ]


# Export tools for backward compatibility
POSTGRES_TOOLS = [
    execute_postgres_query,
    describe_postgres_table,
    list_postgres_tables
]