# 🤖 Smart Analysis Agent

A domain-agnostic intelligent analysis system built with LangGraph, designed to provide deep insights from stored data through natural language queries.

## ✨ Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Advanced Reasoning**: ReAct reasoning pattern for systematic analysis and tool usage
- **Multi-LLM Support**: Anthropic Claude, OpenAI, and Google Gemini models
- **Conversation Memory**: Maintains context across queries using thread IDs
- **Statistical Analysis**: Built-in statistical tools for correlation, trend analysis, and outlier detection
- **Streaming Progress**: Real-time progress updates during analysis execution
- **Database Integration**: PostgreSQL-based data storage and querying
- **Configurable Recursion**: Adjust LangGraph recursion limits for complex analyses

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL database with your data
- LLM API keys (Anthropic, OpenAI, or Google)

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd smart-analysis-agent
```

2. **Install Python dependencies:**
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your API credentials and database settings
```

### Running the Application

**🎯 One Command Start (Recommended):**
```bash
python smart_analysis_main.py
```

This automatically starts:
- FastAPI server on `http://localhost:8000`
- Frontend interface on `http://localhost:3000` (if available)
- Opens your browser to the analysis interface

**Alternative: Manual Start:**
```bash
# Terminal 1: Start API server
python smart_analysis_main.py api

# Terminal 2: Start frontend (if available)
python smart_analysis_main.py frontend
```

## 🖥️ Usage

### Web Interface

1. Open `http://localhost:3000` (if frontend is available)
2. Select your preferred LLM provider and model
3. Adjust recursion limit for query complexity (default: 75)
4. Ask questions about your data!

### API Endpoints

- **Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Query**: `POST /query` - Main analysis endpoint
- **Streaming**: `POST /query/stream` - Get results with progress updates
- **Conversations**: `GET /conversation/list` - List conversation history

### Example Queries

```
• "What are the main trends in our sales data over the last quarter?"
• "Show me statistical summary of customer satisfaction scores"
• "Are there any correlations between user engagement and conversion rates?"
• "Detect outliers in our performance metrics"
• "What patterns exist in our time series data?"
```

## 🧠 Analysis Capabilities

The agent provides sophisticated analytical capabilities:

### **Statistical Analysis**
- Descriptive statistics (mean, median, standard deviation, quartiles)
- Correlation analysis (Pearson and Spearman)
- Outlier detection (IQR and Z-score methods)
- Time series trend analysis with significance testing

### **Data Exploration**
- Automatic schema discovery and data profiling
- Data quality assessment and missing value analysis
- Pattern recognition across different data dimensions
- Comparative analysis between different data segments

### **Intelligent Reasoning**
- ReAct pattern for systematic analysis approach
- Tool selection based on query requirements
- Cross-validation of findings across multiple methods
- Context-aware interpretation of results

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=anthropic              # anthropic, openai, or google
LLM_MODEL=claude-sonnet-4-20250514  # Specific model to use
ANTHROPIC_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
GOOGLE_API_KEY=your-api-key

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=your_database
POSTGRES_USERNAME=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_SCHEMA=public

# Agent Configuration
GRAPH_RECURSION_LIMIT=75      # Agent reasoning depth (5-200)
MAX_QUERY_ITERATIONS=10       # Max analysis iterations
QUERY_TIMEOUT_SECONDS=30      # Query timeout
MAX_RESULT_ROWS=1000          # Max rows returned
```

### Recursion Limits

Control how many reasoning steps the agent can take:

- **Low (5-25)**: Simple queries, faster responses
- **Default (75)**: Balanced performance for most use cases  
- **High (100-200)**: Complex analytical workflows, thorough exploration

## 🧪 Testing

```bash
# Run basic functionality tests
python -c "from src.smart_analysis_agent import SmartAnalysisAgent; print('✅ Import successful')"

# Test database connection
python -c "from tools.postgres_mcp_server import PostgreSQLMCPClient; print('✅ Database tools available')"

# Test API server
curl http://localhost:8000/health
```

## 📁 Project Structure

```
smart-analysis-agent/
├── src/smart_analysis_agent/     # Main agent code
│   ├── react_agent.py           # LangGraph agent implementation  
│   ├── api.py                    # FastAPI server
│   ├── conversation_manager.py   # Memory management
│   └── prompt_manager.py         # Analysis prompts
├── tools/                        # Analysis tools
│   ├── postgres_mcp_server.py   # Database querying tools
│   └── analysis_mcp_server.py   # Statistical analysis tools
├── data/                         # Data models and configurations  
│   ├── analysis_agent.py        # Agent models
│   └── postgres.py              # Database models
├── smart_analysis_main.py        # Application entry point
└── README_TEMPLATE.md            # This documentation
```

## 🔧 Customization

### Adding Domain-Specific Prompts

Modify `src/smart_analysis_agent/prompt_manager.py` to include domain-specific analysis templates:

```python
def _get_domain_specific_prompt(self) -> str:
    return """You are an expert in [YOUR DOMAIN] with access to analytical tools.
    
    Focus on [DOMAIN-SPECIFIC INSIGHTS]...
    """
```

### Custom Analysis Tools

Add new analytical capabilities in `tools/analysis_mcp_server.py`:

```python
@tool
async def your_custom_analysis(
    data_query: str = Field(..., description="SQL query for your analysis"),
    config: RunnableConfig = None
) -> str:
    """Your custom analysis description."""
    # Implementation here
```

### Database Schema Adaptation

The agent works with any PostgreSQL schema. Ensure your database contains the data you want to analyze, and the agent will automatically discover available tables and columns.

## 🐛 Troubleshooting

**Server won't start:**
- Check if ports 8000/3000 are available
- Verify environment variables are set
- Ensure virtual environment is activated

**Database connection issues:**
- Verify PostgreSQL credentials in `.env`
- Test connection manually with psql
- Check network connectivity to database host

**LLM API errors:**
- Verify API keys are correct and active
- Check API quotas and rate limits  
- Ensure model names match provider specifications

**Analysis errors:**
- Check data types in your database columns
- Ensure sufficient data volume for statistical analysis
- Verify column names match your queries

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

---

**Ready to analyze your data intelligently?** Get started with `python smart_analysis_main.py` and ask your first question! 🚀
