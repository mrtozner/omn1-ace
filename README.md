# Omn1-ACE

An intelligent context management system that uses predictive prefetching and multi-tier caching to optimize context delivery for AI-powered development tools. Built with production-grade vector search, knowledge graphs, and team-based sharing capabilities.

## Background

Omn1-ACE emerged from extensive research into context optimization for AI development tools. The system consolidates proven techniques for predictive prefetching, multi-tier caching, and team-based learning from the [OmniMemory research project](https://github.com/mrtozner/omnimemory). This production-ready implementation focuses on the core features that deliver the most value with the simplest deployment.

## Features

- **Predictive Context Prefetching**: Multi-strategy prediction engine that anticipates context needs based on workflow patterns, code structure analysis, and team behavior
- **Team-Based Caching**: Shared L2 cache layer enables teams to benefit from collective context, reducing redundant processing
- **Multi-Modal Search**: Tri-index architecture combining dense vector search (semantic), sparse search (keyword), and structural search (code AST)
- **Code-Aware Compression**: Specialized compression for code that preserves semantic meaning while achieving high compression ratios
- **Multi-Tier Caching**: Three-layer cache architecture (L1: user, L2: team, L3: archive) for optimal performance and cost efficiency
- **Model-Specific Optimization**: Dynamic context generation tailored to specific LLM architectures (Claude, GPT, Gemini)

## Architecture

Omn1-ACE implements a 4-layer anticipatory system:

1. **Prediction Engine**: Multi-strategy ensemble predictor using knowledge graph traversal, workflow pattern matching, and team co-occurrence analysis
2. **Embedding-First Storage**: Vector-based storage using Qdrant with tri-index search capabilities
3. **Dynamic Context Generation**: Model-aware context summarization that optimizes output for specific LLM targets
4. **Collective Intelligence**: Cross-user learning and team-based pattern aggregation

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (recommended for easiest setup)
- PostgreSQL 15+ (if running locally)
- Redis 7+ (if running locally)
- Qdrant vector database (if running locally)

### Installation

#### Option 1: Docker Compose (Recommended)

This is the fastest way to get started with all services configured correctly.

```bash
# Clone the repository
git clone https://github.com/yourusername/omn1-ace.git
cd omn1-ace

# IMPORTANT: Update passwords in deploy/docker-compose.yml
# Edit the file and change POSTGRES_PASSWORD and update DATABASE_URL accordingly
# You can set these via environment variables:
export POSTGRES_PASSWORD="your_secure_password_here"
export ENVIRONMENT="production"

# Start all services (PostgreSQL, Qdrant, Redis, API)
docker-compose -f deploy/docker-compose.yml up -d

# Verify services are running
curl http://localhost:8000/health

# Check logs
docker-compose -f deploy/docker-compose.yml logs -f api
```

#### Option 2: Local Development

For local development with more control over individual components.

```bash
# Clone the repository
git clone https://github.com/yourusername/omn1-ace.git
cd omn1-ace

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start external services (you need these running separately)
# PostgreSQL on port 5432
# Qdrant on port 6333
# Redis on port 6379

# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost:5432/omn1_ace"
export REDIS_URL="redis://localhost:6379"
export QDRANT_URL="http://localhost:6333"
export ENVIRONMENT="development"

# Run database migrations (if applicable)
# alembic upgrade head

# Start API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Configuration

The application is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://omn1:password@localhost:5432/omn1_ace` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `QDRANT_URL` | Qdrant vector DB URL | `http://localhost:6333` |
| `ENVIRONMENT` | Runtime environment | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_WORKERS` | Number of worker processes | `4` |

For production deployments, create a `.env` file or set these variables in your deployment environment.

## API Endpoints

The API server exposes the following main endpoints:

- `GET /health` - Health check endpoint
- `POST /api/v1/embeddings` - Generate embeddings for text/code
- `POST /api/v1/search` - Search using tri-index (semantic + keyword + structural)
- `POST /api/v1/predict` - Get predicted context based on current workflow
- `GET /api/v1/cache/stats` - Cache performance statistics
- `POST /api/v1/compress` - Compress code while preserving semantics

For complete API documentation, start the server and visit:
- OpenAPI docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Services

Omn1-ACE consists of several core services:

### Embeddings Service
- Generates vector embeddings for code and text
- Uses sentence-transformers with code-optimized models
- Port: 8000 (integrated with main API)

### Compression Service
- Code-aware compression using transformer models
- Achieves high compression ratios while preserving semantic meaning
- Supports multiple programming languages

### Knowledge Graph Service
- Builds and maintains code structure graphs using NetworkX
- Stores relationships in PostgreSQL
- Enables graph-based context prediction

### Tri-Index Search
- Dense vector search via Qdrant
- Sparse keyword search (BM25)
- Structural search using AST patterns

### Cache Service
- 3-tier Redis-based caching (L1: user, L2: team, L3: archive)
- Automatic cache warming based on predictions
- LRU eviction with priority preservation

## Extensions

### LSP Integration (Optional)

The LSP extension provides Language Server Protocol integration for enhanced code intelligence:

- **Location**: `extensions/lsp/`
- **Features**: Symbol extraction, AST analysis, code structure understanding
- **Use Case**: Improves context relevance for code-heavy workflows
- **Documentation**: See [extensions/lsp/README.md](extensions/lsp/README.md)

This extension integrates with your IDE's LSP to extract richer code context automatically.

## Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=api --cov=core --cov=engine tests/
```

### Code Style

The project uses:
- `black` for Python code formatting
- `isort` for import sorting
- `pylint` for linting
- `mypy` for type checking

```bash
# Format code
black .
isort .

# Run linters
pylint api/ core/ engine/
mypy api/ core/ engine/
```

### Project Structure

```
omn1-ace/
├── api/              # FastAPI application and routes
├── core/             # Core business logic and services
├── engine/           # Prediction and processing engines
├── deploy/           # Deployment configurations (Docker, K8s)
├── docs/             # Documentation
├── tests/            # Test suites
└── requirements.txt  # Python dependencies
```

## Monitoring

The system exposes Prometheus metrics at `/metrics`:

- Request latency histograms
- Cache hit/miss ratios
- Prediction accuracy metrics
- Database connection pool stats
- Vector search performance

Integration with Grafana dashboards is recommended for production deployments.

## Performance Considerations

### Recommended Resources

For production deployments:

- **API Server**: 2+ CPU cores, 4GB+ RAM
- **PostgreSQL**: 4GB+ RAM, SSD storage
- **Qdrant**: 8GB+ RAM (depends on corpus size)
- **Redis**: 2GB+ RAM (depends on cache size)

### Scaling

- The API server can be horizontally scaled behind a load balancer
- PostgreSQL can be configured with read replicas for read-heavy workloads
- Qdrant supports clustering for large-scale vector search
- Redis can use clustering for high-availability caching

## Security

For production deployments:

1. **Change all default passwords** in `docker-compose.yml`
2. Use **environment variables** for sensitive configuration
3. Enable **TLS/SSL** for all service connections
4. Configure **authentication** for API endpoints
5. Use **network policies** to restrict service access
6. Regular **security updates** for all dependencies

## Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker logs
docker-compose -f deploy/docker-compose.yml logs

# Ensure ports are not in use
lsof -i :8000,5432,6333,6379
```

**Database connection errors:**
```bash
# Verify PostgreSQL is running
docker-compose -f deploy/docker-compose.yml ps postgres

# Check connection string
echo $DATABASE_URL
```

**Vector search performance issues:**
```bash
# Check Qdrant memory usage
curl http://localhost:6333/metrics

# Consider increasing HNSW parameters for better recall
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass
- Code follows style guidelines (black, isort, pylint)
- New features include tests
- Documentation is updated

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/omn1-ace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/omn1-ace/discussions)
- **Documentation**: [docs/](docs/)

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Qdrant](https://qdrant.tech/) - Vector similarity search engine
- [PostgreSQL](https://www.postgresql.org/) - Relational database
- [Redis](https://redis.io/) - In-memory data store
- [NetworkX](https://networkx.org/) - Graph analysis library
