#!/bin/bash

echo "🎨 Starting Tekyz Chatbot Frontend..."

# Navigate to frontend directory
cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Check if backend is running
echo "🔍 Checking backend connection..."
if ! curl -s http://localhost:8080/health > /dev/null; then
    echo "⚠️  Backend is not running on port 8080."
    echo "   Please start the backend first using: ./start_backend.sh"
    echo ""
    echo "🔄 You can still start the frontend, but it won't be able to process chat requests."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the frontend
echo "🚀 Starting Next.js frontend on port 3000..."
echo "🌐 Frontend will be available at: http://localhost:3000"
echo ""

npm run dev 