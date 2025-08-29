"""
Smart Analysis Agent - A domain-agnostic intelligent analysis system.

This package provides a LangGraph-based ReAct agent that can be adapted 
to any domain requiring data analysis, reasoning, and insights.
"""

from .react_agent import SmartAnalysisAgent
from .conversation_manager import ConversationManager
from .prompt_manager import get_prompt_manager

__version__ = "0.1.0"
__all__ = ["SmartAnalysisAgent", "ConversationManager", "get_prompt_manager"]
