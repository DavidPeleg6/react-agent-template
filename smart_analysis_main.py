"""
Main entry point for Smart Analysis Agent.
Provides both web server and command-line interfaces.
"""

import asyncio
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def start_api_server():
    """Start the FastAPI server."""
    try:
        logger.info("🚀 Starting Smart Analysis Agent API server")
        
        # Set Python path
        project_root = Path(__file__).parent
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(project_root)
        
        # Run the API server
        subprocess.run([
            sys.executable, "-m", "src.smart_analysis_agent.api"
        ], env=env, cwd=project_root)
        
    except KeyboardInterrupt:
        logger.info("🔄 Shutting down API server")
    except Exception as e:
        logger.error("❌ Failed to start API server", error=str(e))
        sys.exit(1)


def start_frontend():
    """Start the frontend development server."""
    try:
        logger.info("🌐 Starting frontend server")
        
        frontend_dir = Path(__file__).parent / "frontend"
        if not frontend_dir.exists():
            logger.warning("Frontend directory not found, skipping frontend start")
            return
        
        # Change to frontend directory and start server
        subprocess.run(["npm", "start"], cwd=frontend_dir)
        
    except KeyboardInterrupt:
        logger.info("🔄 Shutting down frontend server")
    except Exception as e:
        logger.error("❌ Failed to start frontend server", error=str(e))


def start_full_application():
    """Start both API and frontend servers."""
    import threading
    
    logger.info("🚀 Starting Smart Analysis Agent - Full Application")
    
    # Start API server in a separate thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Wait a moment for API server to start
    time.sleep(3)
    
    # Open browser to the application
    try:
        webbrowser.open("http://localhost:3000")
    except Exception:
        logger.info("Could not open browser automatically. Please visit http://localhost:3000")
    
    # Start frontend server (this will block)
    start_frontend()


def show_help():
    """Show help information."""
    help_text = """
Smart Analysis Agent - Domain-Agnostic Intelligence System

Usage:
  python smart_analysis_main.py [command]

Commands:
  start     Start both API and frontend servers (default)
  api       Start only the API server (port 8000)
  frontend  Start only the frontend server (port 3000) 
  help      Show this help message

Environment Variables:
  LLM_PROVIDER           LLM provider (anthropic, openai, google)
  LLM_MODEL             Specific model to use
  ANTHROPIC_API_KEY     Anthropic API key
  OPENAI_API_KEY        OpenAI API key  
  GOOGLE_API_KEY        Google API key
  POSTGRES_HOST         PostgreSQL host (default: localhost)
  POSTGRES_PORT         PostgreSQL port (default: 5432)
  POSTGRES_DATABASE     PostgreSQL database name
  POSTGRES_USERNAME     PostgreSQL username
  POSTGRES_PASSWORD     PostgreSQL password
  GRAPH_RECURSION_LIMIT LangGraph recursion limit (default: 75)

Examples:
  python smart_analysis_main.py              # Start full application
  python smart_analysis_main.py api          # Start only API server
  python smart_analysis_main.py frontend     # Start only frontend

For more information, see the documentation.
"""
    print(help_text)


def main():
    """Main entry point."""
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
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Parse command line arguments
    command = "start" if len(sys.argv) < 2 else sys.argv[1].lower()
    
    if command == "help" or command == "--help" or command == "-h":
        show_help()
    elif command == "api":
        start_api_server()
    elif command == "frontend":
        start_frontend()
    elif command == "start":
        start_full_application()
    else:
        logger.error(f"Unknown command: {command}")
        show_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
