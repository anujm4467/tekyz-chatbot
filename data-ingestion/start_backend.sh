#!/bin/bash

# Tekyz Data Pipeline - Backend Startup Script
# This script starts the FastAPI backend server

echo "🚀 Starting Tekyz Data Pipeline Backend..."

# Activate conda environment
echo "📦 Activating conda environment: tekyz-data-ingestion"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tekyz-data-ingestion

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/raw data/processed data/embeddings logs

# Start the FastAPI server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📚 API Documentation available at http://localhost:8000/docs"
python backend_api.py 