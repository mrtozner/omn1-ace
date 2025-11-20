<div align="center">

# Omn1-ACE

**Intelligent Context Management for AI Development Tools**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](deploy/docker-compose.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**[Quick Start](QUICK_START.md)** • **[Documentation](docs/)** • **[Architecture](docs/ARCHITECTURE.md)** • **[Report Issue](https://github.com/mrtozner/omn1-ace/issues)**

---

### 🚀 Cut AI API costs by 85%+ with predictive context prefetching

Omn1-ACE uses multi-tier caching, tri-index search, and team learning to deliver only the context you need—saving tokens, time, and money.

</div>

---

## 🚧 Project Status

> **Current Stage**: Prototype / Early Development
>
> - ✅ Architecture designed and documented
> - ✅ Infrastructure setup (Docker, databases)
> - ⚠️ Core API endpoints are placeholders (not yet implemented)
> - ⚠️ Not production-ready
>
> **For production-ready microservices**, see [OmniMemory](https://github.com/mrtozner/omnimemory)

---

## 💡 Why Omn1-ACE?

| Feature | Traditional Approach | Omn1-ACE |
|---------|---------------------|----------|
| **Context Delivery** | Send entire history every query | Send only relevant context (85% reduction) |
| **Token Usage** | 10,000+ tokens per query | ~1,500 tokens per query |
| **Cost** (Claude, 10K queries/month) | ~$450/month | ~$68/month |
| **Team Learning** | Each user rebuilds context | Shared L2 cache learns from team |
| **Search** | Simple keyword matching | Tri-index (semantic + keyword + structural) |
| **Prediction** | Reactive (wait for query) | Proactive (prefetch likely context) |

**Projected Savings**: $382/month per developer at typical usage

---

## ⚡ Key Features

<table>
<tr>
<td width="33%" valign="top">

### 🧠 Predictive Prefetching
Multi-strategy prediction engine that anticipates context needs before you ask
- Workflow pattern matching
- Code structure analysis
- Team behavior learning

</td>
<td width="33%" valign="top">

### 🔍 Tri-Index Search
Three search methods combined for maximum relevance
- **Dense**: Semantic vector search
- **Sparse**: BM25 keyword matching
- **Structural**: AST-based code patterns

</td>
<td width="33%" valign="top">

### 💾 Multi-Tier Caching
Three-layer cache architecture optimized for performance and cost
- **L1**: User cache (personal patterns)
- **L2**: Team cache (shared knowledge)
- **L3**: Archive (long-term storage)

</td>
</tr>
</table>

### Additional Capabilities

- **Code-Aware Compression**: 85-94% token reduction while preserving semantic meaning
- **Model-Specific Optimization**: Context tailored for Claude, GPT, or Gemini
- **Team Intelligence**: Cross-user learning and pattern aggregation
- **LSP Integration**: Enhanced code intelligence via Language Server Protocol

---

## 🎯 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (recommended)
- 4GB+ RAM

### 🐳 Docker Compose (Recommended)

Get started in 5 minutes:

```bash
# Clone the repository
git clone https://github.com/mrtozner/omn1-ace.git
cd omn1-ace

# Copy environment template
cp .env.example .env

# IMPORTANT: Edit .env and change POSTGRES_PASSWORD
nano .env

# Start all services
docker-compose -f deploy/docker-compose.yml up -d

# Verify services
curl http://localhost:8000/health
```

**[📖 Full Setup Guide →](QUICK_START.md)**

---

## 🏗️ Architecture

Omn1-ACE implements a 4-layer anticipatory system:

```
┌─────────────────────────────────────────────────┐
│         AI Development Tools                    │
│  (Claude Code, Cursor, Continue, etc.)          │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │  Prediction Engine     │  ← Multi-strategy prediction
        │  (Prefetch context)    │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │   Tri-Index Search     │  ← Dense + Sparse + Structural
        │  (Find relevant code)  │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │   Multi-Tier Cache     │  ← L1 (user) + L2 (team) + L3
        │  (Smart retrieval)     │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │   Storage Layer        │  ← Qdrant + PostgreSQL + Redis
        │  (Vector DB + Graph)   │
        └────────────────────────┘
```

**[📐 Detailed Architecture →](docs/ARCHITECTURE.md)**

---

## ⚙️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/embeddings` | POST | Generate vector embeddings |
| `/api/v1/search` | POST | Tri-index search (semantic + keyword + structural) |
| `/api/v1/predict` | POST | Get predicted context for current workflow |
| `/api/v1/cache/stats` | GET | Cache performance statistics |
| `/api/v1/compress` | POST | Compress code while preserving semantics |

**Interactive Docs**: `http://localhost:8000/docs` (OpenAPI)

---

## ⚠️ Multi-Tool Context Considerations

### Context Window Limits

Different AI models have different token limits:

| Model | Context Window | Configuration |
|-------|---------------|---------------|
| **Claude 3.5 Sonnet** | 200,000 tokens | `CLAUDE_CONTEXT_WINDOW=200000` |
| **GPT-4 Turbo** | 128,000 tokens | `GPT_CONTEXT_WINDOW=128000` |
| **Gemini 1.5 Pro** | 1,000,000 tokens | `GEMINI_CONTEXT_WINDOW=1000000` |
| **GPT-3.5 Turbo** | 16,000 tokens | `GPT_CONTEXT_WINDOW=16000` |

**Impact**: Context optimized for Gemini may exceed GPT-4's limits.

### Configuration

Set your target model in `.env`:

```bash
DEFAULT_TARGET_MODEL=claude  # or gpt, gemini
CLAUDE_CONTEXT_WINDOW=200000
GPT_CONTEXT_WINDOW=128000
GEMINI_CONTEXT_WINDOW=1000000
```

### Model-Specific Behavior

**Claude (Anthropic)**:
- ✅ Best with structured, detailed context
- ✅ Excellent at following complex instructions
- ⚡ Prefers explicit task breakdowns

**GPT (OpenAI)**:
- ✅ Works well with conversational context
- ⚠️ May need more explicit formatting
- ⚡ Better with shorter, focused context

**Gemini (Google)**:
- ✅ Handles very large context windows
- ✅ Good with multimodal content
- ⚠️ May need different prompt engineering

**Recommendation**: Standardize on one model per team for consistent experience.

---

## 📊 Performance

### Recommended Resources

| Component | Requirements |
|-----------|-------------|
| **API Server** | 2+ CPU cores, 4GB+ RAM |
| **PostgreSQL** | 4GB+ RAM, SSD storage |
| **Qdrant** | 8GB+ RAM (scales with corpus) |
| **Redis** | 2GB+ RAM (scales with cache) |

### Scaling

- **Horizontal**: API servers behind load balancer
- **PostgreSQL**: Read replicas for read-heavy workloads
- **Qdrant**: Clustering for large-scale vector search
- **Redis**: Clustering for high-availability caching

---

## 🔒 Security

**Before production deployment**:

1. ✅ Change all default passwords in `docker-compose.yml`
2. ✅ Use environment variables for sensitive configuration
3. ✅ Enable TLS/SSL for all service connections
4. ✅ Configure authentication for API endpoints
5. ✅ Use network policies to restrict service access
6. ✅ Regular security updates for all dependencies

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Before submitting a PR**:
- All tests pass
- Code follows style guidelines (black, isort, pylint)
- New features include tests
- Documentation is updated

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Related Projects

- **[OmniMemory](https://github.com/mrtozner/omnimemory)**: Production-ready microservices (13 independent services)
- **Extensions**: LSP integration for enhanced code intelligence ([docs](extensions/lsp/README.md))

---

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Qdrant](https://qdrant.tech/) - Vector similarity search
- [PostgreSQL](https://www.postgresql.org/) - Relational database
- [Redis](https://redis.io/) - In-memory data store
- [NetworkX](https://networkx.org/) - Graph analysis

---

<div align="center">

**[⭐ Star this repo](https://github.com/mrtozner/omn1-ace)** if you find it useful!

**[📖 Read the Docs](docs/)** • **[💬 Discussions](https://github.com/mrtozner/omn1-ace/discussions)** • **[🐛 Report Bug](https://github.com/mrtozner/omn1-ace/issues)**

Made with ❤️ by [Mert Ozoner](https://github.com/mrtozner)

</div>
