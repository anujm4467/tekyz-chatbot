# Tekyz AI Assistant v2.0

A modern RAG (Retrieval Augmented Generation) powered chatbot built with **Next.js** and **FastAPI** that provides intelligent responses about Tekyz company using their populated vector database.

## 🎯 Overview

The Tekyz AI Assistant has been completely modernized with a beautiful web interface and high-performance backend, combining the power of vector databases with large language models to provide accurate, contextual responses about Tekyz services, capabilities, and solutions.

### ✨ Key Features

- 🤖 **RAG Architecture**: Retrieval Augmented Generation for accurate responses
- 🔍 **Vector Search**: Semantic search through Tekyz knowledge base
- 💫 **Modern UI**: Beautiful Next.js frontend with shadcn/ui components
- ⚡ **Fast API**: High-performance FastAPI backend
- 📊 **Real-time Analytics**: Live query processing and response tracking
- 🎯 **Smart Classification**: Automatic query categorization and routing
- 🛡️ **Query Validation**: Input sanitization and safety checks
- 🌙 **Dark Theme**: Professional dark interface with smooth animations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Next.js Frontend (Port 3000)       │         shadcn/ui          │
│ ├── Modern Chat Interface          │      Components            │
│ ├── Real-time Updates              │      Tailwind CSS          │
│ └── TypeScript                     │      Responsive Design     │
├─────────────────────────────────────┼─────────────────────────────┤
│ FastAPI Backend (Port 8080)        │      Vector Search         │
│ ├── Query Processing               │      Qdrant Database       │
│ ├── RAG Generation                 │      Embedding Engine      │
│ └── Analytics Engine               │      Groq AI Integration   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: One-Command Launch

```bash
cd chatbot-app
chmod +x start_chatbot.sh
./start_chatbot.sh
```

### Option 2: Manual Setup

1. **Backend Setup:**

   ```bash
   cd chatbot-app
   chmod +x start_backend.sh
   ./start_backend.sh
   ```

2. **Frontend Setup (in another terminal):**
   ```bash
   cd chatbot-app
   chmod +x start_frontend.sh
   ./start_frontend.sh
   ```

### 🌐 Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

## 📋 Prerequisites

- **Python 3.8+** with virtual environment support
- **Node.js 18+** and npm
- **Qdrant Database** running on port 6333
- **Groq AI API Key** for response generation

### Environment Setup

1. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Required Environment Variables:**

   ```env
   # Required
   GROQ_API_KEY=your_groq_api_key_here

   # Vector Database
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_COLLECTION=tekyz_knowledge

   # Application Settings
   APP_NAME="Tekyz AI Assistant"
   SIMILARITY_THRESHOLD=0.7
   MAX_SEARCH_RESULTS=5
   ```

## 🛠️ Technology Stack

### Frontend

- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - High-quality component library
- **Lucide React** - Beautiful icons

### Backend

- **FastAPI** - High-performance Python web framework
- **Pydantic** - Data validation and settings management
- **Qdrant Client** - Vector database integration
- **Sentence Transformers** - Embedding generation
- **Groq AI** - LLM for response generation

## 🎨 User Interface

### Chat Interface Features

- **Professional Design**: Dark gradient theme with modern aesthetics
- **Real-time Messaging**: Instant responses with typing indicators
- **Confidence Scoring**: Visual confidence indicators for each response
- **Source Attribution**: Clickable links to original sources
- **Session Management**: Persistent chat sessions with history
- **Responsive Design**: Works perfectly on desktop and mobile

### Sidebar Features

- **System Status**: Real-time backend connectivity status
- **Quick Questions**: Pre-defined queries for easy interaction
- **Session Info**: Current conversation statistics
- **Professional Branding**: Tekyz AI Assistant with "Powered by RAG" badge

## 🔧 API Endpoints

### Core Endpoints

| Endpoint             | Method | Description               |
| -------------------- | ------ | ------------------------- |
| `/`                  | GET    | Health check and API info |
| `/health`            | GET    | System health status      |
| `/api/chat`          | POST   | Main chat interaction     |
| `/api/system/status` | GET    | Detailed system status    |

### Chat API Usage

```javascript
// Frontend API call example
const response = await fetch("/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "What services does Tekyz offer?",
    session_id: "user-session-123",
  }),
});

const data = await response.json();
```

## 📊 System Components

### Backend Components

- **QueryProcessor**: Validates, preprocesses, and classifies user queries
- **VectorSearchEngine**: Performs semantic search through Tekyz knowledge base
- **ResponseGenerator**: Generates contextual responses using RAG with Groq AI
- **AnalyticsManager**: Tracks usage patterns and performance metrics

### Integration Layer

- **FastAPI Routes**: RESTful API endpoints for frontend communication
- **Pydantic Models**: Type-safe request/response validation
- **Error Handling**: Comprehensive error management and logging
- **CORS Support**: Proper cross-origin resource sharing setup

## 🔍 Example Usage

### Supported Query Types

**Services & Solutions:**

- "What web development services does Tekyz provide?"
- "Can you build e-commerce platforms?"
- "Tell me about mobile app development"

**Portfolio & Projects:**

- "Show me examples of Tekyz's work"
- "What industries do you serve?"
- "Can you share case studies?"

**Company Information:**

- "Who are the founders of Tekyz?"
- "What technologies does Tekyz use?"
- "How can I contact Tekyz?"

## 🚦 System Health Monitoring

### Real-time Status Indicators

- **Backend Connectivity**: Green/Red status in sidebar
- **Database Health**: Vector database connection status
- **Response Times**: Live performance monitoring
- **Error Tracking**: Automatic error logging and recovery

### Analytics Dashboard

- Query processing times
- Response confidence scores
- Search result relevance
- User interaction patterns

## 🔄 Integration with Data Ingestion

This chatbot seamlessly integrates with the existing data-ingestion pipeline:

1. **Shared Vector Database**: Uses the same Qdrant `tekyz_knowledge` collection
2. **Compatible Embeddings**: Uses identical `all-MiniLM-L6-v2` model
3. **Metadata Utilization**: Leverages document metadata for enhanced search
4. **Live Data**: Works with freshly ingested content automatically

## 🐛 Troubleshooting

### Common Issues

1. **Backend won't start**: Check if port 8080 is available and virtual environment is activated
2. **Frontend build fails**: Ensure Node.js 18+ is installed and run `npm install`
3. **Database connection fails**: Verify Qdrant is running on port 6333
4. **No responses generated**: Check GROQ_API_KEY in .env file

### Debug Mode

Set `ENVIRONMENT=development` in `.env` for detailed logging and debug information.

## 📝 Legacy Migration

This version replaces the previous Streamlit-based interface:

- ❌ **Removed**: Streamlit dependency and app.py
- ✅ **Added**: Modern Next.js frontend with FastAPI backend
- ✅ **Improved**: Performance, user experience, and maintainability
- ✅ **Enhanced**: Real-time features and professional design

## 🤝 Contributing

The codebase is organized for easy development:

```
chatbot-app/
├── frontend/                # Next.js application
│   ├── src/app/            # App router pages
│   ├── src/components/ui/  # shadcn/ui components
│   └── package.json        # Frontend dependencies
├── backend_api.py          # FastAPI application
├── src/backend/           # Core chatbot logic
├── src/utils/             # Utilities and configuration
└── start_*.sh             # Launch scripts
```

## 📄 License

This project is part of the Tekyz knowledge management system.
