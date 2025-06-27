"""
Configuration settings for the data ingestion pipeline
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main configuration class for the data ingestion pipeline"""
    
    # Tekyz Website Configuration
    tekyz_base_url: str = Field(default="https://tekyz.com", env="TEKYZ_BASE_URL")
    scraping_delay: float = Field(default=1.0, env="SCRAPING_DELAY")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    # Vector Database Configuration
    vector_db_provider: str = Field(default="qdrant", env="VECTOR_DB_PROVIDER")
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection: str = Field(default="tekyz_knowledge", env="QDRANT_COLLECTION")
    
    # Embedding Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    
    # Processing Configuration
    chunk_size: int = Field(default=800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    min_chunk_length: int = Field(default=100, env="MIN_CHUNK_LENGTH")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/data_ingestion.log", env="LOG_FILE")
    
    # Docker Configuration
    qdrant_docker_image: str = Field(default="qdrant/qdrant:latest", env="QDRANT_DOCKER_IMAGE")
    qdrant_docker_port: int = Field(default=6333, env="QDRANT_DOCKER_PORT")
    qdrant_docker_volume: str = Field(default="qdrant_storage", env="QDRANT_DOCKER_VOLUME")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the current settings instance"""
    return settings 