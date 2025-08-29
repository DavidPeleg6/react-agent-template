"""
Conversation Manager for LangGraph-based Smart Analysis Agent

This module handles conversation persistence using LangGraph's built-in checkpointing
system and a simplified conversations table for metadata.
"""

import asyncio
import asyncpg
from datetime import datetime
from typing import List, Dict, Optional, Any
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

class ConversationMetadata(BaseModel):
    """Metadata for a conversation thread"""
    thread_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    last_message_at: Optional[datetime] = None

class CheckpointInfo(BaseModel):
    """Information about a checkpoint in a conversation"""
    checkpoint_id: str
    step: int
    created_at: datetime
    values: Dict[str, Any]
    next_nodes: List[str]
    metadata: Dict[str, Any]

class ConversationManager:
    """
    Manages conversation metadata and integrates with LangGraph's checkpointing system.
    
    This manager works alongside LangGraph's AsyncPostgresSaver to provide:
    - Conversation metadata tracking
    - Thread listing and management
    - Integration with LangGraph's time travel capabilities
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._pool = None
    
    async def _get_pool(self):
        """Get or create a connection pool"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.database_url)
        return self._pool
    
    async def close(self):
        """Close the connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def create_or_update_conversation(
        self, 
        thread_id: str, 
        title: Optional[str] = None,
        increment_message_count: bool = False
    ) -> ConversationMetadata:
        """
        Create or update conversation metadata.
        
        Args:
            thread_id: The LangGraph thread ID
            title: Optional title for the conversation
            increment_message_count: Whether to increment the message count
        
        Returns:
            ConversationMetadata object
        """
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            # Check if conversation exists
            existing = await conn.fetchrow(
                "SELECT * FROM conversations WHERE thread_id = $1",
                thread_id
            )
            
            if existing:
                # Update existing conversation
                now = datetime.utcnow()
                
                # Build update query dynamically
                update_fields = ["updated_at = $2"]
                values = [thread_id, now]
                param_count = 2
                
                if title is not None:
                    param_count += 1
                    update_fields.append(f"title = ${param_count}")
                    values.append(title)
                
                if increment_message_count:
                    param_count += 1
                    update_fields.append(f"message_count = message_count + 1")
                    update_fields.append(f"last_message_at = ${param_count}")
                    values.append(now)
                
                query = f"""
                    UPDATE conversations 
                    SET {', '.join(update_fields)}
                    WHERE thread_id = $1
                    RETURNING *
                """
                
                result = await conn.fetchrow(query, *values)
            else:
                # Create new conversation
                now = datetime.utcnow()
                result = await conn.fetchrow(
                    """
                    INSERT INTO conversations (thread_id, title, created_at, updated_at, message_count, last_message_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING *
                    """,
                    thread_id, title, now, now, 1 if increment_message_count else 0, now if increment_message_count else None
                )
        
        return ConversationMetadata(**dict(result))
    
    async def get_conversation(self, thread_id: str) -> Optional[ConversationMetadata]:
        """Get conversation metadata by thread ID"""
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                "SELECT * FROM conversations WHERE thread_id = $1",
                thread_id
            )
            
            if result:
                return ConversationMetadata(**dict(result))
            return None
    
    async def list_conversations(
        self, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[ConversationMetadata]:
        """
        List conversations ordered by most recent activity.
        
        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
        
        Returns:
            List of ConversationMetadata objects
        """
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM conversations 
                ORDER BY updated_at DESC 
                LIMIT $1 OFFSET $2
                """,
                limit, offset
            )
            
            return [ConversationMetadata(**dict(row)) for row in results]
    
    async def delete_conversation(self, thread_id: str) -> bool:
        """
        Delete a conversation and all its checkpoints.
        
        Note: This will delete all LangGraph checkpoints for this thread.
        
        Args:
            thread_id: The thread ID to delete
        
        Returns:
            True if conversation was deleted, False if it didn't exist
        """
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Delete from conversations table
                conversation_result = await conn.execute(
                    "DELETE FROM conversations WHERE thread_id = $1",
                    thread_id
                )
                
                # Delete from LangGraph checkpoints
                await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = $1",
                    thread_id
                )
                
                # Delete from checkpoint writes
                await conn.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = $1",
                    thread_id
                )
                
                # Note: checkpoint_blobs are handled by foreign key constraints
                
                # Return True if we deleted a conversation
                return conversation_result == "DELETE 1"
    
    async def get_conversation_checkpoints(
        self, 
        thread_id: str, 
        limit: int = 50
    ) -> List[CheckpointInfo]:
        """
        Get checkpoints for a conversation from LangGraph's checkpoint tables.
        
        Args:
            thread_id: The thread ID
            limit: Maximum number of checkpoints to return
        
        Returns:
            List of CheckpointInfo objects ordered by step (most recent first)
        """
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            # Get checkpoints from LangGraph's table
            results = await conn.fetch(
                """
                SELECT 
                    checkpoint_id,
                    thread_id,
                    checkpoint_ns,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata
                FROM checkpoints 
                WHERE thread_id = $1 
                ORDER BY checkpoint->>'ts' DESC
                LIMIT $2
                """,
                thread_id, limit
            )
            
            checkpoints = []
            for row in results:
                try:
                    checkpoint_data = row['checkpoint']
                    metadata = row['metadata'] or {}
                    
                    # Extract step from metadata or checkpoint
                    step = metadata.get('step', 0)
                    
                    # Extract values from checkpoint
                    values = checkpoint_data.get('channel_values', {})
                    
                    # Extract next nodes 
                    next_nodes = []
                    if 'pending_sends' in checkpoint_data:
                        next_nodes = [send.get('node', '') for send in checkpoint_data.get('pending_sends', [])]
                    
                    checkpoint_info = CheckpointInfo(
                        checkpoint_id=row['checkpoint_id'],
                        step=step,
                        created_at=datetime.fromisoformat(checkpoint_data.get('ts', '').replace('Z', '+00:00')),
                        values=values,
                        next_nodes=next_nodes,
                        metadata=metadata
                    )
                    checkpoints.append(checkpoint_info)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse checkpoint {row['checkpoint_id']}: {e}")
                    continue
            
            return checkpoints
    
    async def get_thread_message_count(self, thread_id: str) -> int:
        """
        Get the actual message count for a thread by examining checkpoints.
        
        This provides more accurate message counting than relying on our metadata table.
        """
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            # Count checkpoints that represent actual conversation steps
            # (excluding the initial empty checkpoint)
            result = await conn.fetchval(
                """
                SELECT COUNT(*) 
                FROM checkpoints 
                WHERE thread_id = $1 
                AND checkpoint->>'channel_values' IS NOT NULL
                AND jsonb_array_length(COALESCE(checkpoint->'channel_values'->'messages', '[]'::jsonb)) > 0
                """,
                thread_id
            )
            
            return result or 0
    
    async def update_conversation_message_count(self, thread_id: str):
        """Update the message count for a conversation based on actual checkpoints"""
        actual_count = await self.get_thread_message_count(thread_id)
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE conversations 
                SET message_count = $2, updated_at = NOW()
                WHERE thread_id = $1
                """,
                thread_id, actual_count
            )
    
    async def cleanup_old_conversations(self, days: int = 30) -> int:
        """
        Clean up conversations older than specified days.
        
        Args:
            days: Number of days after which to clean up conversations
        
        Returns:
            Number of conversations cleaned up
        """
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Get thread IDs to delete
                old_threads = await conn.fetch(
                    """
                    SELECT thread_id FROM conversations 
                    WHERE updated_at < NOW() - INTERVAL '%s days'
                    """,
                    days
                )
                
                if not old_threads:
                    return 0
                
                thread_ids = [row['thread_id'] for row in old_threads]
                
                # Delete conversations and their checkpoints
                deleted_count = 0
                for thread_id in thread_ids:
                    if await self.delete_conversation(thread_id):
                        deleted_count += 1
                
                logger.info(f"Cleaned up {deleted_count} old conversations")
                return deleted_count
    
    async def get_conversation_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Extract message history from LangGraph checkpoints for a conversation.
        Uses the LangGraph checkpointer API to properly deserialize message data.
        
        Args:
            thread_id: The thread ID
        
        Returns:
            List of message dictionaries with role and content
        """
        try:
            # Import here to avoid circular dependencies
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            
            # Create a temporary checkpointer to read the data
            async with AsyncPostgresSaver.from_conn_string(self.database_url) as checkpointer:
                # Get the latest checkpoint for this thread
                config = {"configurable": {"thread_id": thread_id}}
                
                # Get the state from the checkpointer
                state = await checkpointer.aget_tuple(config)
                
                if not state or not state.checkpoint:
                    logger.info(f"No state found for thread {thread_id}")
                    return []
                
                # Extract messages from the checkpoint
                channel_values = state.checkpoint.get('channel_values', {})
                langraph_messages = channel_values.get('messages', [])
                
                if not langraph_messages:
                    logger.info(f"No messages found in state for thread {thread_id}")
                    return []
                
                # Convert LangGraph messages to frontend format
                messages = []
                logger.info(f"Processing {len(langraph_messages)} LangGraph messages for thread {thread_id}")
                
                # First pass: identify which AI messages are intermediate (followed by tool messages)
                intermediate_ai_indices = set()
                for i, msg in enumerate(langraph_messages):
                    if (hasattr(msg, 'type') and msg.type == 'ai' and 
                        i + 1 < len(langraph_messages) and 
                        hasattr(langraph_messages[i + 1], 'type') and 
                        langraph_messages[i + 1].type == 'tool'):
                        intermediate_ai_indices.add(i)
                
                # Second pass: process messages, skipping intermediate AI messages
                for i, msg in enumerate(langraph_messages):
                    try:
                        logger.debug(f"Message {i}: type={type(msg)}, has_type={hasattr(msg, 'type')}, has_content={hasattr(msg, 'content')}")
                        
                        # Handle LangGraph message objects
                        if hasattr(msg, 'type') and hasattr(msg, 'content'):
                            logger.debug(f"Message {i}: msg.type={msg.type}, content_length={len(str(msg.content))}")
                            
                            if msg.type == 'human':
                                messages.append({
                                    'role': 'user',
                                    'content': msg.content,
                                    'timestamp': getattr(msg, 'id', None)
                                })
                                logger.debug(f"Added user message {i}")
                            elif msg.type == 'ai':
                                # Skip intermediate AI messages that are followed by tool messages
                                if i in intermediate_ai_indices:
                                    logger.debug(f"Skipping intermediate AI message {i} (followed by tool message)")
                                    continue
                                
                                # Handle AI message content that might be a list or string
                                content = msg.content
                                if isinstance(content, list):
                                    # Extract text from list of content blocks
                                    text_parts = []
                                    for block in content:
                                        if isinstance(block, dict) and block.get('type') == 'text':
                                            text_parts.append(block.get('text', ''))
                                        elif isinstance(block, str):
                                            text_parts.append(block)
                                    content = ' '.join(text_parts)
                                
                                # Convert content to string if it's not already
                                if not isinstance(content, str):
                                    try:
                                        content = str(content)
                                    except Exception:
                                        content = "[Unable to display content]"
                                
                                # Additional filtering for very short or system-like messages
                                if len(content.strip()) < 10:
                                    logger.debug(f"Skipping short AI message {i} (length: {len(content.strip())})")
                                    continue
                                
                                # Filter out system-like messages
                                content_lower = content.lower().strip()
                                if (content_lower in ['ok', 'done', 'yes', 'no'] or 
                                    content_lower.startswith('{') or 
                                    content_lower.startswith('<') or
                                    content_lower.startswith('[') or
                                    'json' in content_lower or
                                    'xml' in content_lower):
                                    logger.debug(f"Skipping system-like AI message {i}: {content[:50]}...")
                                    continue
                                
                                messages.append({
                                    'role': 'assistant', 
                                    'content': content,
                                    'timestamp': getattr(msg, 'id', None)
                                })
                                logger.debug(f"Added assistant message {i}")
                            # Skip system messages and tool messages for UI display
                        elif isinstance(msg, dict):
                            # Handle dict format (fallback)
                            if msg.get('type') == 'human':
                                messages.append({
                                    'role': 'user',
                                    'content': msg.get('content', ''),
                                    'timestamp': msg.get('id')
                                })
                                logger.debug(f"Added user message {i} (dict format)")
                            elif msg.get('type') == 'ai':
                                messages.append({
                                    'role': 'assistant', 
                                    'content': msg.get('content', ''),
                                    'timestamp': msg.get('id')
                                })
                                logger.debug(f"Added assistant message {i} (dict format)")
                    except Exception as e:
                        logger.warning(f"Failed to parse message {i} in thread {thread_id}: {e}")
                        continue
                
                logger.info(f"Successfully extracted {len(messages)} messages for thread {thread_id}")
                return messages
                
        except Exception as e:
            logger.warning(f"Failed to extract messages from checkpoints for thread {thread_id}: {e}")
            return []

    async def rewind_to_message(self, thread_id: str, message_index: int) -> Dict[str, Any]:
        """
        Rewind a conversation to a specific message index using LangGraph time travel.
        
        Args:
            thread_id: The thread ID
            message_index: The index of the message to rewind to (0-based)
        
        Returns:
            Dictionary with success status and details
        """
        try:
            logger.info(f"Attempting to rewind conversation {thread_id} to message {message_index}")
            
            # Import here to avoid circular dependencies
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            
            # Create a temporary checkpointer to access the data
            async with AsyncPostgresSaver.from_conn_string(self.database_url) as checkpointer:
                # Get the conversation config
                config = {"configurable": {"thread_id": thread_id}}
                
                # Get the current state to verify the conversation exists
                current_state = await checkpointer.aget_tuple(config)
                if not current_state:
                    return {"success": False, "error": "Conversation not found"}
                
                # Get all checkpoints for this conversation
                checkpoints = []
                async for checkpoint in checkpointer.alist(config):
                    checkpoints.append(checkpoint)
                
                if not checkpoints:
                    return {"success": False, "error": "No checkpoints found for this conversation"}
                
                # Sort checkpoints by timestamp (newest first)
                checkpoints.sort(key=lambda x: x.checkpoint.get('ts', ''), reverse=True)
                
                # Extract messages from each checkpoint to find the right one
                target_checkpoint = None
                target_checkpoint_id = None
                
                for checkpoint_tuple in checkpoints:
                    try:
                        channel_values = checkpoint_tuple.checkpoint.get('channel_values', {})
                        messages = channel_values.get('messages', [])
                        
                        # Count user and assistant messages (exclude system/tool messages)
                        user_assistant_messages = []
                        for msg in messages:
                            if hasattr(msg, 'type') and msg.type in ['human', 'ai']:
                                user_assistant_messages.append(msg)
                        
                        # Check if this checkpoint has the right number of messages
                        # We want the checkpoint right after the target message was added
                        if len(user_assistant_messages) > message_index:
                            target_checkpoint = checkpoint_tuple
                            target_checkpoint_id = checkpoint_tuple.checkpoint.get('id')
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse checkpoint in rewind: {e}")
                        continue
                
                if not target_checkpoint:
                    # If we can't find a perfect match, use the earliest checkpoint
                    if checkpoints:
                        target_checkpoint = checkpoints[-1]  # Oldest checkpoint
                        target_checkpoint_id = target_checkpoint.checkpoint.get('id')
                    else:
                        return {"success": False, "error": "Could not find appropriate checkpoint for rewind"}
                
                # Create config with the target checkpoint
                rewind_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": target_checkpoint_id
                    }
                }
                
                # Verify the checkpoint is accessible
                rewind_state = await checkpointer.aget_tuple(rewind_config)
                if not rewind_state:
                    return {"success": False, "error": "Target checkpoint not accessible"}
                
                logger.info(f"Successfully identified checkpoint {target_checkpoint_id} for message {message_index}")
                
                # Update conversation metadata to reflect the rewind state
                await self.create_or_update_conversation(
                    thread_id=thread_id,
                    title=f"[Rewound] Previous conversation",
                    increment_message_count=False
                )
                
                return {
                    "success": True,
                    "checkpoint_id": target_checkpoint_id,
                    "message": f"Conversation rewound to message {message_index + 1}",
                    "note": "Conversation state identified for rewind"
                }
                
        except Exception as e:
            logger.error(f"Failed to rewind conversation {thread_id} to message {message_index}: {e}")
            return {"success": False, "error": f"Time travel failed: {str(e)}"}
