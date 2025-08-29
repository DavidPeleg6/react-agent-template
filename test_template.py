#!/usr/bin/env python3
"""
Quick test script to validate the Smart Analysis Agent template.
Tests imports, configuration, and basic functionality without requiring full setup.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing core imports...")
    
    try:
        # Test data models
        from data import (
            QueryRequest, QueryResponse, AgentConfig, OutputMode, 
            LLMProvider, PostgreSQLConnectionConfig
        )
        print("✅ Data models imported successfully")
        
        # Test agent components
        from src.smart_analysis_agent import SmartAnalysisAgent
        from src.smart_analysis_agent.conversation_manager import ConversationManager
        from src.smart_analysis_agent.prompt_manager import get_prompt_manager
        print("✅ Agent components imported successfully")
        
        # Test tools
        from tools.postgres_mcp_server import create_postgres_tools_with_config
        from tools.analysis_mcp_server import create_analysis_tools_with_config
        print("✅ Analysis tools imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration creation."""
    print("\n🧪 Testing configuration...")
    
    try:
        from data import AgentConfig, PostgreSQLConnectionConfig, LLMProvider
        
        # Create test PostgreSQL config
        postgres_config = PostgreSQLConnectionConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            schema_name="public"
        )
        print("✅ PostgreSQL config created")
        
        # Create test agent config
        agent_config = AgentConfig(
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
            anthropic_api_key="test_key",
            postgres_config=postgres_config,
            graph_recursion_limit=50
        )
        print("✅ Agent config created")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_prompt_manager():
    """Test prompt manager functionality."""
    print("\n🧪 Testing prompt manager...")
    
    try:
        from src.smart_analysis_agent.prompt_manager import get_prompt_manager
        from data import QueryRequest, OutputMode
        
        # Get prompt manager
        pm = get_prompt_manager()
        print("✅ Prompt manager created")
        
        # Test prompt generation
        request = QueryRequest(
            question="Test question about data analysis",
            output_mode=OutputMode.ANSWER_QUESTION
        )
        
        context = {
            "domain_focus": ["trends", "statistics"],
            "timeframe": "30d",
            "include_analysis": True
        }
        
        # This should be async but for testing we'll just check it exists
        assert hasattr(pm, 'get_analysis_prompt')
        assert hasattr(pm, 'get_data_exploration_prompt')
        print("✅ Prompt manager methods available")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt manager test failed: {e}")
        return False


def test_tools_creation():
    """Test tool creation without database connection."""
    print("\n🧪 Testing tools creation...")
    
    try:
        from tools.postgres_mcp_server import create_postgres_tools_with_config
        from tools.analysis_mcp_server import create_analysis_tools_with_config
        from data import PostgreSQLConnectionConfig
        
        # Create mock config
        config = PostgreSQLConnectionConfig(
            host="localhost",
            port=5432,
            database="test",
            username="test",
            password="test",
            schema_name="public"
        )
        
        # Create tools (should not fail even without DB connection)
        postgres_tools = create_postgres_tools_with_config(config)
        analysis_tools = create_analysis_tools_with_config(config)
        
        print(f"✅ Created {len(postgres_tools)} PostgreSQL tools")
        print(f"✅ Created {len(analysis_tools)} analysis tools")
        
        return True
        
    except Exception as e:
        print(f"❌ Tools creation test failed: {e}")
        return False


def test_api_imports():
    """Test API server imports."""
    print("\n🧪 Testing API imports...")
    
    try:
        # Test FastAPI app can be imported
        from src.smart_analysis_agent.api import app
        print("✅ FastAPI app imported successfully")
        
        # Test main entry point
        from smart_analysis_main import main
        print("✅ Main entry point imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ API import test failed: {e}")
        return False


def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    env_file = project_root / ".env"
    env_template = project_root / ".env.template"
    
    if not env_file.exists() and env_template.exists():
        print("\n📝 Creating sample .env file...")
        
        try:
            with open(env_template, 'r') as template:
                content = template.read()
            
            with open(env_file, 'w') as env:
                env.write(content)
            
            print("✅ Sample .env file created from template")
            print("🔧 Please edit .env with your actual configuration")
            
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")


def main():
    """Run all tests."""
    print("🚀 Smart Analysis Agent Template Validation\n")
    
    tests = [
        ("Core Imports", test_imports),
        ("Configuration", test_configuration),
        ("Prompt Manager", test_prompt_manager),
        ("Tools Creation", test_tools_creation),
        ("API Imports", test_api_imports),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Template is ready to use.")
        create_sample_env()
        print("\n🚀 Next steps:")
        print("1. Edit .env with your configuration")
        print("2. Set up your PostgreSQL database")
        print("3. Run: python smart_analysis_main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
