"""
Test Suite for Tekyz Data Ingestion Layer

Comprehensive testing suite including unit tests, integration tests,
and end-to-end pipeline testing.
"""

import pytest
import logging
from pathlib import Path

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

__version__ = "1.0.0" 