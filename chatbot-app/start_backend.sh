#!/bin/bash

echo "🚀 Starting Tekyz Chatbot Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Add FastAPI and uvicorn if not in requirements
pip install fastapi uvicorn

# Check if Qdrant is running
echo "🔍 Checking Qdrant connection..."
if ! curl -s http://localhost:6333/collections > /dev/null; then
    echo "⚠️  Qdrant is not running. Please start Qdrant first:"
    echo "   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest"
    echo ""
    echo "🔄 Attempting to start Qdrant..."
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
    sleep 5
fi

# Start the backend
echo "🔥 Starting FastAPI backend on port 8080..."
echo "📊 Backend will be available at: http://localhost:8080"
echo "📚 API docs will be available at: http://localhost:8080/docs"
echo ""

python backend_api.py 