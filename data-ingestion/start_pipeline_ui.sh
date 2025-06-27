#!/bin/bash

# Tekyz Data Pipeline - Full Stack Startup Script
# This script starts both the frontend and backend in parallel

echo "🚀 Starting Tekyz Data Pipeline Full Stack Application..."

# Function to handle cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    jobs -p | xargs -r kill
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Activate conda environment
echo "📦 Activating conda environment: tekyz-data-ingestion"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tekyz-data-ingestion

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/raw data/processed data/embeddings logs frontend/node_modules

# Start backend in background
echo "🔥 Starting FastAPI backend server..."
python backend_api.py &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 3

# Start frontend in background
echo "🌐 Starting Next.js frontend..."
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

# Wait for both processes
echo "✅ Services started successfully!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for any process to exit
wait $BACKEND_PID $FRONTEND_PID 