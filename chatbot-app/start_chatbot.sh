#!/bin/bash

echo "🤖 Starting Tekyz Chatbot Application..."
echo "============================================"

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Tekyz Chatbot..."
    jobs -p | xargs -r kill
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Activate existing conda environment
echo "🔄 Activating conda environment 'tekyz-chatbot'..."
eval "$(conda shell.bash hook)"
conda activate tekyz-chatbot

# Install/upgrade packages from requirements.txt
echo "📦 Installing/upgrading chatbot packages from requirements.txt..."
pip install -r requirements.txt --upgrade

# Check if data-ingestion backend is running (needed for Qdrant)
echo "🔍 Checking if data-ingestion backend is running..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "🚀 Starting data-ingestion backend..."
    cd ../data-ingestion
    
    # Install/upgrade data-ingestion packages if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing/upgrading data-ingestion packages..."
        pip install -r requirements.txt --upgrade
    fi
    
    python backend_api.py &
    DATA_BACKEND_PID=$!
    cd "$SCRIPT_DIR"
    
    # Wait for data-ingestion backend to start
    echo "⏳ Waiting for data-ingestion backend to initialize..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Data-ingestion backend is running"
            break
        fi
        sleep 1
    done
else
    echo "✅ Data-ingestion backend is already running"
fi

# Make scripts executable
chmod +x start_backend.sh
chmod +x start_frontend.sh

# Start chatbot backend in background
echo ""
echo "1️⃣  Starting Chatbot Backend..."
./start_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for chatbot backend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Chatbot backend is running"
        break
    fi
    sleep 1
done

# Check if backend is running
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "❌ Chatbot backend failed to start. Check the logs above."
    kill $BACKEND_PID 2>/dev/null
    [ ! -z "$DATA_BACKEND_PID" ] && kill $DATA_BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend in background
echo ""
echo "2️⃣  Starting Frontend..."
./start_frontend.sh &
FRONTEND_PID=$!

# Wait for frontend to start
echo "⏳ Waiting for frontend to initialize..."
for i in {1..45}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Frontend is running"
        break
    fi
    sleep 1
done

# Check if frontend is running
if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "❌ Frontend failed to start. Check the logs above."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    [ ! -z "$DATA_BACKEND_PID" ] && kill $DATA_BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 Tekyz Chatbot is now running!"
echo "============================================"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Chatbot Backend: http://localhost:8080"
echo "🔧 Data Backend: http://localhost:8000"
echo "📚 API Docs: http://localhost:8080/docs"
echo ""
echo "💡 Try asking questions like:"
echo "   • What services does Tekyz offer?"
echo "   • Tell me about your web development capabilities"
echo "   • Can you build mobile apps?"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for processes to complete
wait 