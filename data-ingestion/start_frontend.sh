#!/bin/bash

# Tekyz Data Pipeline - Frontend Startup Script
# This script starts the Next.js frontend application

echo "🚀 Starting Tekyz Data Pipeline Frontend..."

# Activate conda environment
echo "📦 Activating conda environment: tekyz-data-ingestion"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tekyz-data-ingestion

# Navigate to frontend directory
cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Start the development server
echo "🌐 Starting Next.js development server on http://localhost:3000"
npm run dev 