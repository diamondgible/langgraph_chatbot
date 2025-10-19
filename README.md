# LangGraph Chatbot

A comprehensive chatbot implementation built with LangGraph, featuring multiple backend architectures, LLM provider integrations, and persistent conversation management. This project demonstrates various approaches to building conversational AI systems with different storage backends and language model providers.

## 🏗️ Architecture Overview

The project provides multiple implementation variants to suit different use cases:

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph Chatbot                        │
├─────────────────────────────────────────────────────────────────┤
│  Frontend Layer (Streamlit)                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ Database UI     │ │ Memory UI       │ │ Generic API UI  │    │
│  │ (Persistent)    │ │ (Session-based) │ │ (Custom API)    │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Backend Layer (LangGraph)                                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ Database        │ │ Memory Saver    │ │ Generic LLM     │    │
│  │ Backend         │ │ Backend         │ │ Provider Backend│    │
│  │ (SQLite)        │ │ (In-Memory)     │ │ (SQLite + API)  │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  LLM Provider Layer                                             │
│  ┌─────────────────┐ ┌─────────────────┐                        │
│  │ Ollama          │ │ Generic API     │                        │
│  │ (Local Models)  │ │ (OpenAI Compat) │                        │
│  └─────────────────┘ └─────────────────┘                        │
├─────────────────────────────────────────────────────────────────┤
│  Storage Layer                                                  │
│  ┌─────────────────┐ ┌─────────────────┐                        │
│  │ SQLite Database │ │ In-Memory Store │                        │
│  │ (Persistent)    │ │ (Temporary)     │                        │
│  └─────────────────┘ └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### Backend Implementations
- **Database Backend** (`langgraph_database_backend.py`) - SQLite-based conversation persistence
- **Memory Backend** (`langgraph_memory_saver_backend.py`) - In-memory conversation storage
- **Generic Provider Backend** (`langgraph_database_backend_generic_provider_integrated.py`) - Custom LLM API integration

### Frontend Implementations
- **Database Frontend** (`streamlit_database_frontend.py`) - Full conversation history management
- **Memory Frontend** (`streamlit_memory_saver_frontend.py`) - Session-based conversations
- **Generic Provider Frontend** (`streamlit_database_frontend_generic_llm_integrated.py`) - Custom API integration

### LLM Provider Support
- **Ollama** - Local model hosting (default: Llama 3.1 8B)
- **Generic API** - Custom OpenAI-compatible API endpoints
- **Extensible** - Easy integration with additional providers

## 📁 Project Structure

```
langgraph_chatbot/
├── Backend Implementations
│   ├── langgraph_database_backend.py                    # SQLite + Ollama
│   ├── langgraph_memory_saver_backend.py               # Memory + Ollama  
│   └── langgraph_database_backend_generic_provider_integrated.py  # SQLite + Generic API
├── Frontend Implementations
│   ├── streamlit_database_frontend.py                  # Database UI
│   ├── streamlit_memory_saver_frontend.py             # Memory UI
│   └── streamlit_database_frontend_generic_llm_integrated.py  # Generic API UI
├── LLM Provider Modules
│   ├── langchain_generic/                              # Generic API integration
│   │   ├── __init__.py
│   │   └── chat_generic.py                            # Custom LangChain wrapper
├── Testing & Development
│   ├── test_chat_generic.py                           # Generic API tests
│   └── chatbot_initial_design.ipynb                   # Design experiments
└── Configuration
    ├── requirements.txt                                # Dependencies
    └── .env.sample                                     # Environment template
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama (for local model hosting) or access to a compatible LLM API

### 1. Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file from `.env.sample` and configure your environment:

```bash
# For Ollama (default)
OLLAMA_HOST=http://localhost:11434

# For Generic API
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://your-api-endpoint.com/v1

# For LangSmith Tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=your_project_name_here
```

### 4. Choose Your Implementation

#### Option A: Database Backend with Ollama (Recommended)
```bash
# Start the Streamlit frontend
streamlit run streamlit_database_frontend.py
```

#### Option B: Memory Backend with Ollama
```bash
# Start the Streamlit frontend
streamlit run streamlit_memory_saver_frontend.py
```

#### Option C: Generic API Integration
```bash
# Start the Streamlit frontend
streamlit run streamlit_database_frontend_generic_llm_integrated.py
```

### 5. Access the Application

Open your browser to the URL shown in the terminal (typically `http://localhost:8501`).

## 🔧 Implementation Details

### Backend Implementations

#### Database Backend (`langgraph_database_backend.py`)
- **Storage**: SQLite database for persistent conversation history
- **LLM**: Ollama with Llama 3.1 8B model
- **Features**: 
  - Thread-based conversation management
  - Persistent conversation history
  - Automatic thread retrieval and management
- **Use Case**: Production-ready chatbot with conversation persistence

#### Memory Backend (`langgraph_memory_saver_backend.py`)
- **Storage**: In-memory conversation storage
- **LLM**: Ollama with Llama 3.1 8B model
- **Features**:
  - Session-based conversations
  - Fast startup and response times
  - No persistent storage
- **Use Case**: Development, testing, or temporary conversations

#### Generic Provider Backend (`langgraph_database_backend_generic_provider_integrated.py`)
- **Storage**: SQLite database for persistent conversation history
- **LLM**: Custom OpenAI-compatible API via `ChatGeneric` wrapper
- **Features**:
  - Flexible API integration
  - Custom model configuration
  - Token usage tracking
- **Use Case**: Integration with custom or third-party LLM APIs

### Frontend Implementations

#### Database Frontend (`streamlit_database_frontend.py`)
- **Features**:
  - Conversation thread management
  - Persistent conversation history
  - Thread switching and loading
  - Real-time streaming responses
- **UI Components**:
  - Sidebar with conversation list
  - "New Chat" button for thread creation
  - Chat interface with message history

#### Memory Frontend (`streamlit_memory_saver_frontend.py`)
- **Features**:
  - Session-based conversations
  - Simplified UI without persistent history
  - Real-time streaming responses
- **UI Components**:
  - Basic chat interface
  - "New Chat" button for session reset

#### Generic Provider Frontend (`streamlit_database_frontend_generic_llm_integrated.py`)
- **Features**:
  - Full conversation management with custom API
  - Thread-based persistence
  - Real-time streaming responses
- **UI Components**:
  - Same as Database Frontend
  - Compatible with custom LLM providers

## 🔍 LangSmith Integration

LangSmith provides powerful observability and debugging capabilities, allowing us to trace execution, monitor performance, and debug issues in real-time.

### Setup and Configuration

#### 1. Create LangSmith Account

1. Visit [LangSmith](https://smith.langchain.com/)
2. Sign up for a free account
3. Create a new project for your chatbot

#### 2. Environment Configuration

Add LangSmith configuration to your `.env` file:

```bash
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=your_project_name_here
```

#### 3. Install LangSmith SDK

The LangSmith SDK is included in the project dependencies, but you can also install it separately:

```bash
pip install langsmith
```

### Usage in the Chatbot

LangSmith tracing is automatically enabled when the environment variables are set. The chatbot will automatically:

- **Trace all LLM calls** - See exactly what prompts are sent and responses received
- **Track conversation flow** - Visualize the complete conversation graph
- **Monitor performance** - Track response times and token usage
- **Log errors** - Capture and display any errors that occur

### Viewing Traces

1. **Access LangSmith Dashboard**
   - Go to [smith.langchain.com](https://smith.langchain.com/)
   - Navigate to your project

2. **Explore Traces**
   - View individual conversation traces
   - Analyze execution timing and costs
   - Debug failed requests
   - Compare different model responses

3. **Filter and Search**
   - Filter by date, user, or conversation thread
   - Search for specific patterns or errors
   - Export traces for analysis

### Troubleshooting LangSmith

Enable debug logging for LangSmith:

```bash
# Add to .env file
LANGCHAIN_VERBOSE=true
```

This will provide detailed logging of LangSmith operations.

## 🤖 LLM Provider Configuration

### Ollama (Local Model Hosting)

Ollama provides local model hosting with excellent performance and privacy. The default configuration uses Llama 3.1 8B.

#### Installation & Setup

1. **Install Ollama**
   ```bash
   # Visit https://ollama.ai for platform-specific instructions
   # Windows: Download installer
   # macOS: brew install ollama
   # Linux: curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Start Ollama Service**
   ```bash
   # Start the daemon (usually automatic)
   ollama serve
   ```

3. **Pull the Model**
   ```bash
   # Pull Llama 3.1 8B (default model)
   ollama pull llama3.1:8b
   
   # Alternative models you can use:
   ollama pull llama3.1:70b        # Larger, more capable
   ollama pull mistral:7b          # Alternative model
   ollama pull codellama:7b        # Code-focused model
   ```

4. **Configure Environment**
   ```bash
   # .env file
   OLLAMA_HOST=http://localhost:11434
   ```

#### Testing Ollama Connection

```bash
# Test with curl
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello, how are you?",
  "stream": false
}'
```

```python
# Test with Python
import requests
import os

host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
url = f"{host}/api/chat"
payload = {
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello from test"}]
}
response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

### Generic API Integration

The project includes a flexible `ChatGeneric` wrapper for integrating with any OpenAI-compatible API.

#### Configuration

```bash
# .env file
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://your-api-endpoint.com/v1
```

#### Supported APIs

The `ChatGeneric` class works with any OpenAI-compatible API, including:
- **OpenAI API** - GPT models
- **Anthropic Claude** - Via compatible endpoints
- **Google Gemini** - Via compatible endpoints
- **Local APIs** - Self-hosted models
- **Third-party Services** - Any OpenAI-compatible provider

#### Custom Model Configuration

```python
from langchain_generic import ChatGeneric

# Configure custom model
llm = ChatGeneric(
    model="your-custom-model",
    temperature=0.7,
    max_tokens=512,
    top_p=0.9
)
```

#### Testing Generic API

```bash
# Run the test script
python test_chat_generic.py
```

This will test both direct API calls and the LangChain integration.

### Model Comparison

| Provider | Model | Use Case | Pros | Cons |
|----------|-------|----------|------|------|
| Ollama | Llama 3.1 8B | Local development | Free, private, fast | Requires local resources |
| Ollama | Llama 3.1 70B | High-quality responses | Very capable | High resource requirements |
| Generic API | Various | Production deployment | Flexible, scalable | API costs, internet required |

## 📋 Dependencies

The project uses the following key dependencies:

### Core Dependencies
- **streamlit** (1.50.0) - Web UI framework
- **langchain-core** (0.3.79) - Core LangChain functionality
- **langgraph** (0.6.10) - Graph-based conversation flow
- **langgraph-checkpoint** (2.1.2) - Conversation state management
- **langgraph-checkpoint-sqlite** (2.0.11) - SQLite persistence

### LLM Integration
- **langchain-ollama** (0.3.10) - Ollama integration
- **openai** (2.5.0) - OpenAI API client
- **ollama** (0.6.0) - Ollama Python client

### Observability & Monitoring
- **langsmith** (0.1.0) - LangSmith tracing and monitoring

### Utilities
- **python-dotenv** (1.1.1) - Environment variable management
- **requests** (2.32.5) - HTTP client
- **httpx** (0.28.1) - Async HTTP client
- **pydantic** (2.12.0) - Data validation

## 🧪 Testing

### Test Generic API Integration

```bash
# Test the generic API wrapper
python test_chat_generic.py
```

This test script validates:
- Direct API function calls
- LangChain ChatGeneric class integration
- Message formatting and response handling
- Token usage tracking

### Manual Testing

1. **Start a backend implementation**
2. **Launch the corresponding frontend**
3. **Test conversation flow**:
   - Send messages
   - Create new conversations
   - Switch between conversation threads (database implementations)
   - Verify streaming responses

## 🔧 Advanced Configuration

### Custom Model Parameters

#### Ollama Models
```python
# In backend files, modify the ChatOllama configuration
llm = ChatOllama(
    model="llama3.1:8b",
    base_url=OLLAMA_HOST,
    temperature=0.7,
    top_p=0.9,
    num_ctx=2048
)
```

#### Generic API Models
```python
# In generic provider backend
llm = ChatGeneric(
    model="Meta-Llama-3.1-8B-Instruct",
    temperature=0.7,
    max_tokens=225,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)
```

### Database Configuration

#### SQLite Database
- **File**: `chatbot.db`
- **Tables**: Automatically created by LangGraph
- **Backup**: Copy `chatbot.db` file to backup conversations

#### Custom Database
```python
# Modify connection in backend files
conn = sqlite3.connect(
    database='custom_chatbot.db',
    check_same_thread=False
)
```

## 🚀 Deployment Options

### Local Development
- Use Ollama for local model hosting
- Memory backend for quick testing
- Database backend for persistent development

### Production Deployment

#### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Configure environment variables
4. Deploy with database backend

#### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_database_frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Cloud Platforms
- **Heroku** - Easy deployment with Procfile
- **Railway** - Simple container deployment
- **AWS/GCP/Azure** - Container services

## 🔍 Troubleshooting

### Common Issues

#### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve
```

#### Database Lock Issues
```bash
# Remove lock files if present
rm -f chatbot.db-shm chatbot.db-wal
```

#### API Key Issues
```bash
# Verify environment variables
python -c "import os; print(os.getenv('LLM_API_KEY'))"
```

#### LangSmith Issues
```bash
# Verify LangSmith configuration
python -c "import os; print('LANGCHAIN_TRACING_V2:', os.getenv('LANGCHAIN_TRACING_V2'))"
python -c "import os; print('LANGCHAIN_API_KEY:', os.getenv('LANGCHAIN_API_KEY')[:10] + '...' if os.getenv('LANGCHAIN_API_KEY') else 'Not set')"
```

#### Memory Issues
- Use memory backend for development
- Monitor system resources with Ollama
- Consider smaller models for limited resources

### Performance Optimization

#### For Ollama
- Use GPU acceleration when available
- Adjust model size based on hardware
- Monitor memory usage

#### For Generic APIs
- Implement request caching
- Use connection pooling
- Monitor API rate limits

## 📚 Development Roadmap

### Completed Features
- ✅ Multiple backend implementations
- ✅ LLM provider flexibility
- ✅ Conversation persistence
- ✅ Real-time streaming
- ✅ Thread management
- ✅ Generic API integration
- ✅ LangSmith tracing and monitoring

### Future Enhancements
- 🔄 Tool calling integration
- 🔄 Vector store integration (FAISS, Pinecone)
- 🔄 Advanced conversation analytics
- 🔄 Multi-modal support (images, documents)
- 🔄 Conversation export/import
- 🔄 Model performance evaluation
- 🔄 Automated testing suite

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the implementation details above
