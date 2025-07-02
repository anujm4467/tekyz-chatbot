"""
Configuration Management for Tekyz Chatbot

This module handles all configuration loading and management for the chatbot application.
It supports environment variables, configuration files, and default values.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ConfigManager:
    """
    Centralized configuration management for the Tekyz chatbot.
    
    Handles:
    - Environment variable loading
    - Default configuration values
    - Groq AI configuration
    - Vector database configuration
    - Application settings
    """
    
    def __init__(self):
        # Load environment variables
        self._load_environment()
        
        # Application settings
        self.app_title = os.getenv("APP_TITLE", "Tekyz AI Assistant")
        self.app_description = os.getenv("APP_DESCRIPTION", "Your intelligent assistant for Tekyz services and solutions")
        self.max_query_length = int(os.getenv("MAX_QUERY_LENGTH", "500"))
        self.max_response_length = int(os.getenv("MAX_RESPONSE_LENGTH", "1000"))
        
        # Groq AI configuration
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.groq_classifier_model = os.getenv("GROQ_CLASSIFIER_MODEL", "llama3-8b-8192")
        self.groq_generator_model = os.getenv("GROQ_GENERATOR_MODEL", "llama3-70b-8192")
        self.temperature = float(os.getenv("TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "512"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        
        # Vector database configuration
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "tekyz_knowledge")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        
        # Search configuration
        self.search_limit = int(os.getenv("SEARCH_LIMIT", "5"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
        
        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "logs/chatbot.log")
    
    def _load_environment(self):
        """Load environment variables with validation."""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
    
    def get_groq_config(self) -> Dict[str, Any]:
        """
        Get Groq AI configuration.
        
        Returns:
            Dictionary containing Groq configuration
        """
        return {
            "api_key": self.groq_api_key,
            "base_url": self.groq_base_url,
            "models": {
                "classifier": self.groq_classifier_model,
                "generator": self.groq_generator_model
            },
            "default_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p
            }
        }
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """
        Get vector database configuration.
        
        Returns:
            Dictionary containing vector DB configuration
        """
        return {
            "host": self.qdrant_host,
            "port": self.qdrant_port,
            "collection": self.qdrant_collection,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "search_limit": self.search_limit,
            "similarity_threshold": self.similarity_threshold
        }
    
    def get_app_config(self) -> Dict[str, Any]:
        """
        Get application configuration.
        
        Returns:
            Dictionary containing app configuration
        """
        return {
            "title": self.app_title,
            "description": self.app_description,
            "max_query_length": self.max_query_length,
            "max_response_length": self.max_response_length,
            "log_level": self.log_level,
            "log_file": self.log_file
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration.
        
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check Groq API key
        if not self.groq_api_key:
            warnings.append("Groq API key not configured - using fallback classification")
        elif not self.groq_api_key.startswith("gsk_"):
            issues.append("Invalid Groq API key format")
        
        # Check vector database connectivity (this would need to be tested externally)
        if not self.qdrant_host:
            issues.append("Qdrant host not configured")
        
        # Check file permissions for logging
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            test_file = f"{self.log_file}.test"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            issues.append(f"Cannot write to log file: {e}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        } 