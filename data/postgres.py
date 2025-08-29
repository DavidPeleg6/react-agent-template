"""
PostgreSQL-specific data models for database operations.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class PostgreSQLConnectionConfig(BaseModel):
    """Configuration for PostgreSQL database connection."""
    
    host: str = Field(..., description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Username for authentication")
    password: Optional[str] = Field(default=None, description="Password for authentication")
    schema_name: str = Field(default="public", description="Default schema")
    ssl_mode: str = Field(default="prefer", description="SSL mode (disable, allow, prefer, require)")
    
    def get_dsn(self) -> str:
        """Get PostgreSQL DSN connection string."""
        dsn_parts = [
            f"host={self.host}",
            f"port={self.port}",
            f"dbname={self.database}",
            f"user={self.username}",
            f"sslmode={self.ssl_mode}"
        ]
        if self.password:
            dsn_parts.append(f"password={self.password}")
        return " ".join(dsn_parts)


class QueryResult(BaseModel):
    """Result from executing a PostgreSQL query."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    query: Optional[str] = None
    execution_time_ms: Optional[float] = None
    row_count: Optional[int] = None
    error: Optional[str] = None


class TableInfo(BaseModel):
    """Information about a database table."""
    table_name: str = Field(..., description="Table name")
    columns: Optional[List[Dict[str, Any]]] = Field(default=None, description="Column information")


class SchemaInfo(BaseModel):
    """Schema information for database tables."""
    success: bool
    tables: Optional[List[TableInfo]] = None
    error: Optional[str] = None