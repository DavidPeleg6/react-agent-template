#!/usr/bin/env python3
"""
Quick one-liner test for Smart Analysis Agent
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data import QueryRequest, AgentConfig, PostgreSQLConnectionConfig, LLMProvider, DEFAULT_MODELS
from src.smart_analysis_agent import SmartAnalysisAgent

async def quick_test():
    """Quick test of the agent"""
    
    # Simple config (you'll need to set your API key)
    config = AgentConfig(
        llm_provider="anthropic",  # or "openai" or "google"
        llm_model=DEFAULT_MODELS[LLMProvider.ANTHROPIC],
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "your-key-here"),
        postgres_config=PostgreSQLConnectionConfig(
            host="localhost", port=5432, database="your_db", 
            username="postgres", password="", schema_name="public"
        )
    )
    
    # Create agent
    agent = SmartAnalysisAgent(config)
    
    # Quick query
    request = QueryRequest(question="List the available database tables and their schemas")
    
    print("🤖 Running LangGraph Agent...")
    response = await agent.query(request)
    
    if response.success:
        print("✅ Success!")
        print(f"Answer: {response.answer}")
    else:
        print(f"❌ Error: {response.error_message}")
    
    await agent.close()

if __name__ == "__main__":
    asyncio.run(quick_test())
