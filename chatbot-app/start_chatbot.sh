#!/bin/bash

echo "🤖 Starting Tekyz Chatbot Application..."
echo "============================================"

# Make scripts executable
chmod +x start_backend.sh
chmod +x start_frontend.sh

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Tekyz Chatbot..."
    jobs -p | xargs -r kill
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start backend in background
echo "1️⃣  Starting Backend..."
./start_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 10

# Check if backend is running
if ! curl -s http://localhost:8080/health > /dev/null; then
    echo "❌ Backend failed to start. Check the logs above."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Backend is running"

# Start frontend in background
echo ""
echo "2️⃣  Starting Frontend..."
./start_frontend.sh &
FRONTEND_PID=$!

# Wait for frontend to start
echo "⏳ Waiting for frontend to initialize..."
sleep 15

# Check if frontend is running
if ! curl -s http://localhost:3000 > /dev/null; then
    echo "❌ Frontend failed to start. Check the logs above."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 Tekyz Chatbot is now running!"
echo "============================================"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:8080"
echo "📚 API Docs: http://localhost:8080/docs"
echo ""
echo "Press Ctrl+C to stop both services..."

# Wait for processes to complete
wait 