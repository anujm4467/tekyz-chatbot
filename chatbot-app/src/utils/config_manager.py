"""
Configuration Management System

This module provides centralized configuration management for the Tekyz chatbot application.
It handles environment variables, default values, and configuration validation.
"""

import os
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager(BaseSettings):
    """
    Central configuration manager for the Tekyz chatbot application.
    
    Handles all application settings including:
    - Groq AI configuration
    - Vector database settings
    - Application behavior settings
    - UI and display preferences
    """
    
    # Groq AI Configuration
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", env="GROQ_BASE_URL")
    groq_classifier_model: str = Field(default="llama3-8b-8192", env="GROQ_CLASSIFIER_MODEL")
    groq_generator_model: str = Field(default="llama3-70b-8192", env="GROQ_GENERATOR_MODEL")
    groq_temperature: float = Field(default=0.3, env="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=500, env="GROQ_MAX_TOKENS")
    groq_top_p: float = Field(default=0.9, env="GROQ_TOP_P")
    
    # Vector Database Configuration
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection: str = Field(default="tekyz_knowledge", env="QDRANT_COLLECTION")
    vector_db_provider: str = Field(default="qdrant", env="VECTOR_DB_PROVIDER")
    
    # Embedding Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # Search Configuration
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS")
    search_limit: int = Field(default=5, env="SEARCH_LIMIT")
    
    # Application Configuration
    app_name: str = Field(default="Tekyz AI Assistant", env="APP_NAME")
    app_title: str = Field(default="Tekyz AI Assistant", env="APP_TITLE") 
    app_description: str = Field(default="Your intelligent assistant for Tekyz services and solutions", env="APP_DESCRIPTION")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    max_query_length: int = Field(default=500, env="MAX_QUERY_LENGTH")
    max_response_length: int = Field(default=1000, env="MAX_RESPONSE_LENGTH")
    session_timeout: int = Field(default=3600, env="SESSION_TIMEOUT")
    max_tokens: int = Field(default=512, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Rate Limiting
    queries_per_minute: int = Field(default=30, env="QUERIES_PER_MINUTE")
    queries_per_hour: int = Field(default=200, env="QUERIES_PER_HOUR")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/chatbot.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_config()
        logger.info(f"Configuration loaded successfully for {self.app_name} v{self.app_version}")
    
    def _validate_config(self):
        """Validate critical configuration values."""
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not provided. AI features will be limited.")
        
        if self.qdrant_port <= 0 or self.qdrant_port > 65535:
            raise ValueError(f"Invalid Qdrant port: {self.qdrant_port}")
        
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError(f"Similarity threshold must be between 0 and 1: {self.similarity_threshold}")
    
    def get_groq_config(self) -> Dict[str, Any]:
        """Get Groq AI configuration."""
        return {
            "api_key": self.groq_api_key,
            "base_url": self.groq_base_url,
            "models": {
                "classifier": self.groq_classifier_model,
                "generator": self.groq_generator_model
            },
            "default_params": {
                "temperature": self.groq_temperature,
                "max_tokens": self.groq_max_tokens,
                "top_p": self.groq_top_p
            }
        }
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        return {
            "provider": self.vector_db_provider,
            "host": self.qdrant_host,
            "port": self.qdrant_port,
            "collection_name": self.qdrant_collection,
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_search_results
        }
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return {
            "app_name": self.app_name,
            "version": self.app_version,
            "max_query_length": self.max_query_length,
            "max_response_length": self.max_response_length,
            "session_timeout": self.session_timeout,
            "rate_limit": {
                "queries_per_minute": self.queries_per_minute,
                "queries_per_hour": self.queries_per_hour
            }
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return {
            "model_name": self.embedding_model,
            "dimension": self.embedding_dimension
        }
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_search_results,
            "boost_factors": {
                "homepage": 1.2,
                "services": 1.1,
                "portfolio": 1.0,
                "about": 0.9
            }
        }
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": self.log_level,
            "file": self.log_file,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }


# Global configuration instance
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


def reload_config() -> ConfigManager:
    """Reload the configuration."""
    global _config_instance
    _config_instance = ConfigManager()
    return _config_instance 