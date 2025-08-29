#!/usr/bin/env python3
"""
Simple terminal interface for Smart Analysis Agent.
Run queries directly from the command line without UI.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data import (
    QueryRequest, AgentConfig, PostgreSQLConnectionConfig, 
    LLMProvider, DEFAULT_MODELS, OutputMode
)
from src.smart_analysis_agent import SmartAnalysisAgent


def create_agent_config():
    """Create agent configuration from environment or defaults."""
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
    )
    
    return config


async def progress_callback(event):
    """Simple progress callback for terminal output."""
    print(f"🔄 {event.get('message', 'Processing...')}")


async def main():
    """Main terminal chat interface."""
    print("🤖 Smart Analysis Agent - Terminal Interface")
    print("=" * 50)
    
    # Check for .env file
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  No .env file found. Creating from template...")
        template_file = project_root / ".env.template"
        if template_file.exists():
            import shutil
            shutil.copy(template_file, env_file)
            print("📝 .env file created. Please edit it with your configuration.")
        else:
            print("❌ No .env.template found. Please create .env manually.")
        return
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment loaded from .env")
    except ImportError:
        print("⚠️  python-dotenv not available, using system environment")
    
    # Check required environment variables
    required_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]
    if not any(os.getenv(key) for key in required_keys):
        print("❌ No LLM API keys found in environment!")
        print("Please set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY")
        return
    
    # Create agent
    try:
        config = create_agent_config()
        agent = SmartAnalysisAgent(config)
        print(f"✅ Agent initialized with {config.llm_provider} ({config.llm_model})")
        print(f"📊 Database: {config.postgres_config.host}:{config.postgres_config.port}/{config.postgres_config.database}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    print("💡 Tips:")
    print("- Ask questions about data in your PostgreSQL database")
    print("- Use natural language queries")
    print("- Type 'quit' or 'exit' to stop")
    print("- Type 'help' for example queries")
    print()
    
    try:
        while True:
            # Get user input
            try:
                question = input("🔍 Your question: ").strip()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print("""
📚 Example queries you can try:

• "What tables are available in the database?"
• "Show me summary statistics for [column_name] in [table_name]"
• "Are there any correlations between [column1] and [column2]?"
• "Detect outliers in [column_name] from [table_name]"
• "What are the trends in [time_column] and [value_column]?"
• "Analyze the data quality in [table_name]"
• "What patterns exist in the data?"

Replace [column_name], [table_name] etc. with your actual database schema.
                """)
                continue
            
            # Create query request
            request = QueryRequest(
                question=question,
                output_mode=OutputMode.ANSWER_QUESTION,
                graph_recursion_limit=config.graph_recursion_limit
            )
            
            print(f"\n🚀 Analyzing: {question}")
            print("-" * 50)
            
            try:
                # Execute query with progress updates
                response = await agent.query(request, progress_callback=progress_callback)
                
                if response.success:
                    print(f"\n✅ Analysis Complete!")
                    print(f"⏱️  Execution time: {response.total_execution_time_ms:.0f}ms")
                    if response.thread_id:
                        print(f"🧵 Thread ID: {response.thread_id}")
                    print(f"\n📊 Result:\n{response.answer}")
                else:
                    print(f"\n❌ Query failed: {response.error_message}")
                
            except Exception as e:
                print(f"\n❌ Error during query: {e}")
            
            print("\n" + "=" * 50)
    
    finally:
        # Cleanup
        try:
            await agent.close()
            print("🔄 Agent resources cleaned up")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    asyncio.run(main())
