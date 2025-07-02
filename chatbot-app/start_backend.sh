#!/bin/bash

echo "🚀 Starting Tekyz Chatbot Backend..."

# Activate existing conda environment
echo "🔄 Activating conda environment 'tekyz-chatbot'..."
eval "$(conda shell.bash hook)"
conda activate tekyz-chatbot

# Install/upgrade packages from requirements.txt
echo "📦 Installing/upgrading packages from requirements.txt..."
pip install -r requirements.txt --upgrade

# Check if Qdrant is accessible via data-ingestion backend
echo "🔍 Checking Qdrant connection..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  Data-ingestion backend (with Qdrant) is not running."
    echo "   The chatbot will start but vector search may not work."
    echo "   Make sure to start the data-ingestion backend first."
fi

# Start FastAPI backend
echo "🔥 Starting FastAPI backend on port 8080..."
echo "📊 Backend will be available at: http://localhost:8080"
echo "📚 API docs will be available at: http://localhost:8080/docs"

python backend_api.py 