# Tekyz Knowledge-Based Chatbot - Data Ingestion Layer

A comprehensive data ingestion pipeline for the Tekyz Knowledge-Based Chatbot, designed to scrape, process, and store web content for intelligent retrieval and response generation.

## 🏗️ Architecture Overview

The data ingestion layer consists of several interconnected components:

1. **Web Scraping Component** - Extracts content from web sources
2. **Content Processing Component** - Cleans, chunks, and analyzes content
3. **Embedding Generation Component** - Creates vector embeddings for semantic search
4. **Vector Database Integration** - Stores and manages embeddings in Qdrant
5. **Integration Pipeline** - Orchestrates the complete workflow
6. **Testing & Monitoring** - Comprehensive quality assurance and performance tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- 8GB+ RAM recommended
- 10GB+ free disk space

### Installation

1. **Clone and Setup**

```bash
git clone <repository-url>
cd tekyz-chatbot/data-ingestion
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Start Vector Database**

```bash
docker-compose up -d
```

3. **Configure Environment**

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run Initial Setup**

```bash
python setup.py
```

### Basic Usage

```python
from src.integration_pipeline import DataIngestionPipeline

# Initialize pipeline
pipeline = DataIngestionPipeline()

# Process URLs
urls = ["https://example.com/docs", "https://example.com/blog"]
results = pipeline.process_urls(urls)

print(f"Processed {results['total_chunks']} chunks")
print(f"Generated {results['total_embeddings']} embeddings")
```

## 📁 Project Structure

```
data-ingestion/
├── src/                          # Source code
│   ├── scraper/                  # Web scraping components
│   │   ├── web_scraper.py       # Main scraping logic
│   │   ├── content_extractor.py # Content extraction
│   │   └── url_manager.py       # URL management
│   ├── processor/               # Content processing
│   │   ├── text_processor.py   # Text cleaning and chunking
│   │   ├── metadata_extractor.py # Metadata extraction
│   │   ├── quality_control.py  # Quality validation
│   │   └── pipeline.py         # Processing orchestration
│   ├── embeddings/             # Embedding generation
│   │   ├── model_manager.py    # Model lifecycle management
│   │   ├── text_preprocessor.py # Text preprocessing
│   │   ├── embedding_generator.py # Embedding creation
│   │   ├── embedding_validator.py # Quality validation
│   │   └── embedding_storage.py # Storage management
│   ├── database/               # Vector database integration
│   │   ├── qdrant_client.py    # Qdrant client
│   │   ├── vector_db_manager.py # Database management
│   │   └── integration_pipeline.py # Complete pipeline
│   └── utils/                  # Utility functions
│       ├── config.py          # Configuration management
│       ├── logging_config.py  # Logging setup
│       └── exceptions.py      # Custom exceptions
├── tests/                      # Test suite
│   ├── test_components.py     # Unit tests
│   ├── test_integration.py    # Integration tests
│   └── test_e2e.py           # End-to-end tests
├── docker-compose.yml         # Docker services
├── requirements.txt          # Python dependencies
├── run_tests.py             # Test runner
├── performance_monitor.py   # Performance monitoring
└── README.md               # This file
```

## 🧪 Testing

The project includes comprehensive testing at multiple levels:

### Test Types

1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Component interaction testing
3. **End-to-End Tests** - Complete pipeline testing
4. **Performance Tests** - Benchmarking and optimization

### Running Tests

**Run All Tests:**

```bash
python run_tests.py
```

**Run Specific Test Suites:**

```bash
# Unit tests only
python run_tests.py --unit

# Integration tests only
python run_tests.py --integration

# End-to-end tests only
python run_tests.py --e2e

# Quick test suite (unit + integration)
python run_tests.py --quick
```

**Run with Coverage:**

```bash
python run_tests.py --coverage
```

**Performance Benchmarks:**

```bash
python run_tests.py --performance
```

### Test Reports

Test results are automatically saved to `test_results/`:

- `test_results.json` - Detailed JSON report
- `test_report.html` - HTML dashboard
- `coverage_html/` - Coverage report
- Individual test suite reports

### Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python run_tests.py --coverage

- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./test_results/coverage.json
```

## 📊 Performance Monitoring

Real-time performance monitoring and alerting system:

### Starting the Monitor

```bash
# Start monitoring
python performance_monitor.py

# Monitor for specific duration
python performance_monitor.py --duration 3600  # 1 hour

# Custom monitoring interval
python performance_monitor.py --interval 0.5  # 500ms
```

### Performance Metrics

**System Metrics:**

- CPU usage percentage
- Memory usage and availability
- Disk usage and free space
- Network I/O statistics
- Process count

**Pipeline Metrics:**

- Processing throughput (items/second)
- Response times
- Error rates
- Memory usage peaks
- Component-specific performance

### Alerting

Automatic alerts for:

- High resource usage (CPU > 80%, Memory > 85%)
- Low throughput (< 1 item/second)
- High error rates (> 5%)
- Slow response times (> 5 seconds)

### Performance Reports

```bash
# Generate performance report
python performance_monitor.py --report --report-hours 24
```

Reports include:

- System performance statistics
- Pipeline throughput analysis
- Error rate trends
- Alert summaries
- Recommendations

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_WORKERS=4

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
BATCH_SIZE=32

# Monitoring Configuration
MONITORING_INTERVAL=1.0
ALERT_THRESHOLDS_CPU=80
ALERT_THRESHOLDS_MEMORY=85
```

### Advanced Configuration

See `src/utils/config.py` for detailed configuration options:

```python
from src.utils.config import Config

config = Config()
config.scraper.max_concurrent_requests = 10
config.processor.chunk_size = 1500
config.embeddings.model_name = "all-mpnet-base-v2"
```

## 🐳 Docker Deployment

### Development Environment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment

```bash
# Production configuration
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale worker=3
```

### Service Health Checks

```bash
# Check Qdrant health
curl http://localhost:6333/health

# Check pipeline status
python -c "from src.database.qdrant_client import QdrantClient; print(QdrantClient().health_check())"
```

## 📈 Performance Optimization

### Recommended Settings

**For High Throughput:**

```python
config = {
    'max_workers': 8,
    'batch_size': 64,
    'chunk_size': 800,
    'concurrent_requests': 20
}
```

**For Low Memory:**

```python
config = {
    'max_workers': 2,
    'batch_size': 16,
    'chunk_size': 500,
    'concurrent_requests': 5
}
```

### Monitoring Performance

Use the performance monitor to identify bottlenecks:

```python
from performance_monitor import PerformanceMonitor, PerformanceContext

monitor = PerformanceMonitor()
monitor.start_monitoring()

# Track operation performance
with PerformanceContext(monitor, 'processor', 'chunk_text') as ctx:
    chunks = processor.chunk_text(content)
    ctx.add_processed_items(len(chunks))
```

## 🔍 Troubleshooting

### Common Issues

**1. Qdrant Connection Failed**

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant
```

**2. Out of Memory Errors**

```python
# Reduce batch size
config.embeddings.batch_size = 16
config.processor.max_workers = 2
```

**3. Slow Processing**

```python
# Increase parallelism
config.processor.max_workers = 8
config.scraper.max_concurrent_requests = 20
```

**4. High Error Rates**

```bash
# Check logs
tail -f logs/data_ingestion.log

# Run diagnostics
python -m src.utils.diagnostics
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
from src.utils.logging_config import setup_logging
setup_logging(level='DEBUG')
```

### Performance Diagnostics

```bash
# Run performance analysis
python performance_monitor.py --report --report-hours 1

# Check system resources
python -c "
from performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor()
monitor.print_current_status()
"
```

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code quality checks
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Testing Guidelines

1. Write tests for all new features
2. Maintain >90% code coverage
3. Include integration tests for new components
4. Add performance tests for critical paths

### Code Quality

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Include error handling and logging

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

1. Check the troubleshooting section above
2. Review the test results and logs
3. Run performance diagnostics
4. Create an issue with detailed information

## 🔄 Version History

- **v1.0.0** - Initial release with complete pipeline
- **v1.1.0** - Added comprehensive testing suite
- **v1.2.0** - Added performance monitoring and alerting
- **v1.3.0** - Enhanced error handling and recovery

---

**Built with ❤️ for the Tekyz Knowledge-Based Chatbot**
