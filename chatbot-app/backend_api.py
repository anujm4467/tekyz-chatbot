"""
Tekyz Chatbot Backend API

FastAPI backend that integrates with the existing chatbot components
and provides a modern API for the Next.js frontend.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing chatbot components
from src.backend.query_processor import QueryProcessor
from src.backend.vector_search import VectorSearchEngine
from src.backend.response_generator import ResponseGenerator
from src.backend.analytics import AnalyticsManager
from src.models.data_models import Message, MessageType, ChatSession
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger()

# Global components
components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup and cleanup on shutdown"""
    try:
        logger.info("Initializing Tekyz Chatbot Backend...")
        
        # Initialize configuration
        config = ConfigManager()
        components['config'] = config
        
        # Initialize logger
        components['logger'] = get_logger()
        
        # Initialize backend components
        logger.info("Initializing query processor...")
        components['query_processor'] = QueryProcessor()
        
        logger.info("Connecting to vector database...")
        components['vector_search'] = VectorSearchEngine(config)
        
        logger.info("Initializing response generator...")
        components['response_generator'] = ResponseGenerator()
        
        logger.info("Initializing analytics...")
        components['analytics'] = AnalyticsManager()
        
        # Check system health
        health_status = components['vector_search'].health_check()
        if health_status.get("ready_for_search", False):
            logger.info("✅ System initialized successfully")
        else:
            logger.warning("⚠️ System initialized with warnings - vector search may be limited")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        # Still yield to allow the app to start, but with limited functionality
        yield
    finally:
        logger.info("Shutting down Tekyz Chatbot Backend...")

# Create FastAPI app
app = FastAPI(
    title="Tekyz Chatbot API",
    description="Backend API for the Tekyz AI Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]
    ready_for_search: bool
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Tekyz Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check component availability
        vector_search = components.get('vector_search')
        health_status = {}
        
        if vector_search:
            health_status = vector_search.health_check()
        
        components_status = {
            "query_processor": 'query_processor' in components,
            "vector_search": 'vector_search' in components and health_status.get('qdrant_connection', False),
            "response_generator": 'response_generator' in components,
            "analytics": 'analytics' in components,
            "embedding_model": health_status.get('embedding_model', False),
            "database": health_status.get('database_accessible', False)
        }
        
        overall_status = "healthy" if all(components_status.values()) else "degraded"
        ready_for_search = health_status.get("ready_for_search", False)
        
        return HealthResponse(
            status=overall_status,
            components=components_status,
            ready_for_search=ready_for_search,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            components={},
            ready_for_search=False,
            timestamp=datetime.now().isoformat()
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Check if components are available
        query_processor = components.get('query_processor')
        vector_search = components.get('vector_search')
        response_generator = components.get('response_generator')
        analytics = components.get('analytics')
        
        if not all([query_processor, vector_search, response_generator]):
            raise HTTPException(
                status_code=503,
                detail="Backend components not fully initialized"
            )
        
        # Process the query through the existing pipeline
        logger.info("Processing query...")
        processing_result = query_processor.process_query(request.query, request.session_id or "unknown")
        
        if not processing_result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=processing_result.get('error', 'Query processing failed')
            )
        
        processed_query = processing_result['processed_query']
        classification = processing_result['classification']
        
        # Search for relevant documents
        logger.info("Searching vector database...")
        search_results = vector_search.search(
            query=processed_query,
            limit=5,
            score_threshold=0.6
        )
        
        # Generate response
        logger.info("Generating response...")
        response_data = response_generator.generate_response(
            query=request.query,
            context_results=search_results,
            classification=classification,
            session_id=request.session_id or "unknown"
        )
        
        # Log analytics
        if analytics:
            try:
                analytics.log_query_interaction(
                    session_id=request.session_id or "unknown",
                    user_query=request.query,
                    chat_response=response_data,
                    classification_result=classification
                )
            except Exception as e:
                logger.warning(f"Analytics logging failed: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract sources from search results
        sources = []
        if hasattr(search_results, 'results'):
            sources = [result.source_url for result in search_results.results if hasattr(result, 'source_url')]
        elif isinstance(search_results, list):
            sources = [result.source_url for result in search_results if hasattr(result, 'source_url') and result.source_url]
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
        return ChatResponse(
            response=response_data.response_text,
            confidence=response_data.confidence,
            sources=sources,
            metadata={
                "query_type": classification.category.value if classification and hasattr(classification, 'category') else "general",
                "search_results_count": len(search_results) if isinstance(search_results, list) else 0,
                "session_id": request.session_id,
                "tekyz_related": classification.is_tekyz_related if classification else True
            },
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Return a fallback response
        return ChatResponse(
            response=f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again or contact support if the issue persists.",
            confidence=0.0,
            sources=[],
            metadata={
                "error": True,
                "error_type": type(e).__name__,
                "session_id": request.session_id
            },
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

@app.get("/api/system/status")
async def system_status():
    """Get detailed system status"""
    try:
        vector_search = components.get('vector_search')
        
        if not vector_search:
            return {"error": "Vector search component not available"}
        
        health_status = vector_search.health_check()
        
        return {
            "system": {
                "status": "operational" if health_status.get("ready_for_search") else "degraded",
                "uptime": "N/A",  # You can implement uptime tracking
                "version": "1.0.0"
            },
            "components": health_status,
            "database": {
                "connection": health_status.get("qdrant_connection", False),
                "collections": health_status.get("collections", []),
                "vector_count": health_status.get("vector_count", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return {
            "error": "Failed to get system status",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    
    logger.info(f"Starting Tekyz Chatbot Backend on port {port}")
    
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    ) 