# Advanced Document Search Engine

A sophisticated document retrieval and search system built on top of the existing SCM Governance document inventory. This search engine provides natural language query processing, semantic search capabilities, advanced filtering, and comprehensive analytics.

## 🚀 Features

### Core Search Capabilities
- **Natural Language Processing**: Understands natural language queries and extracts intent
- **Semantic Search**: Uses vector embeddings for concept-based document similarity
- **Advanced Filtering**: Filter by document type, category, file extension, and more
- **Faceted Search**: Dynamic facets based on search results
- **Relevance Scoring**: Sophisticated ranking algorithms combining multiple factors

### Document Analysis
- **Content Extraction**: Supports PDF, DOCX, XLSX, HTML, TXT, and more
- **Metadata Enhancement**: Automatically extracts titles, authors, versions, and categories
- **Keyword Extraction**: Identifies important terms and phrases
- **Entity Recognition**: Recognizes systems, departments, and processes

### Analytics & Insights
- **Search Analytics**: Tracks query patterns and user behavior
- **Popular Document Tracking**: Identifies frequently accessed documents
- **Performance Metrics**: Monitors search speed and accuracy
- **Usage Statistics**: Comprehensive search and access analytics

### User Interface
- **Web Interface**: Modern, responsive web application
- **REST API**: Complete API for integration with other systems
- **Command Line Interface**: Interactive CLI for advanced users
- **Document Preview**: Rich document previews with snippets

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB+ RAM (8GB recommended for large document sets)
- 2GB+ free disk space for index storage

### Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install flask PyPDF2 python-docx pandas beautifulsoup4 lxml
pip install nltk sentence-transformers faiss-cpu numpy scikit-learn
```

### NLTK Data
The first run will automatically download required NLTK data:
- punkt (tokenization)
- stopwords (stop words)
- wordnet (lemmatization)

## 🛠️ Installation & Setup

1. **Clone or navigate to the search engine directory**:
   ```bash
   cd search_engine/
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the search engine**:
   ```bash
   python -m search_engine.main --index
   ```

4. **Start the web interface**:
   ```bash
   python -m search_engine.main --web
   ```

5. **Access the search interface**:
   Open your browser and navigate to `http://127.0.0.1:5000`

## 🎯 Usage

### Web Interface

1. **Simple Search**: Enter natural language queries like:
   - "how to process invoices in Oracle Cloud"
   - "payment troubleshooting checklist"
   - "vendor master procedures"

2. **Advanced Search**: Use filters and categories for precise results:
   - Filter by document type (Procedure, Checklist, Guide)
   - Filter by category (Invoice Processing, Payment Processing)
   - Filter by file type (PDF, DOCX, XLSX)

3. **Document View**: Click on any result to view:
   - Document metadata and properties
   - Content preview with highlighted matches
   - Related document suggestions

### Command Line Interface

**Interactive Search**:
```bash
python -m search_engine.main --search
```

**Index Documents**:
```bash
# Index from existing catalog
python -m search_engine.main --index

# Index specific directory
python -m search_engine.main --index /path/to/documents
```

**View Statistics**:
```bash
python -m search_engine.main --stats
```

### API Usage

**Search Documents**:
```bash
curl "http://localhost:5000/api/search?q=invoice%20processing&limit=10"
```

**Get Document Details**:
```bash
curl "http://localhost:5000/api/document/DOCUMENT_ID"
```

**Get Search Suggestions**:
```bash
curl "http://localhost:5000/api/suggestions?q=invoice"
```

**Get Popular Documents**:
```bash
curl "http://localhost:5000/api/popular?limit=5"
```

## 🔧 Configuration

### Search Configuration (`config.py`)

Key settings you can customize:

```python
@dataclass
class SearchConfig:
    # Document processing
    chunk_size: int = 1000          # Size of document chunks for indexing
    chunk_overlap: int = 200        # Overlap between chunks

    # Vector search
    embedding_model: str = "all-MiniLM-L6-v2"  # Model for semantic search
    vector_dimensions: int = 384    # Embedding dimensions

    # Search settings
    default_search_limit: int = 50  # Default number of results
    semantic_search_weight: float = 0.7  # Weight for semantic search
    keyword_search_weight: float = 0.3   # Weight for keyword search
```

### Document Types Supported

- **PDF**: Full text extraction with metadata
- **DOCX**: Word documents with formatting preserved
- **XLSX**: Excel files with sheet content
- **HTML**: Web pages and formatted documents
- **TXT**: Plain text files
- **MD**: Markdown files
- **RTF**: Rich text format

## 📊 Architecture

### Core Components

1. **Document Processor** (`document_processor.py`)
   - Extracts content from various file formats
   - Analyzes document structure and metadata
   - Creates searchable document chunks

2. **NLP Processor** (`nlp_processor.py`)
   - Processes natural language queries
   - Extracts entities and intent
   - Provides query expansion and suggestions

3. **Vector Search** (`vector_search.py`)
   - Creates vector embeddings for semantic search
   - Implements FAISS-based similarity search
   - Manages vector index storage

4. **Search Engine** (`search_engine.py`)
   - Orchestrates all search components
   - Implements ranking and filtering algorithms
   - Tracks analytics and popularity

5. **Integration Layer** (`integration.py`)
   - Connects with existing document catalog
   - Provides seamless migration and synchronization
   - Exposes REST API endpoints

### Data Flow

```
User Query → Query Processing → Semantic Search → Keyword Search → Ranking → Results
    ↓              ↓                    ↓             ↓           ↓         ↓
   NLP       Intent & Entity     Vector Embeddings   Filters   Relevance  Filtered
 Processing    Extraction           Search        Matching    Scoring   Results
```

## 🔍 Search Features

### Natural Language Queries

The search engine understands natural language patterns:

- **Intent Recognition**: Identifies what type of information you need
- **Entity Extraction**: Recognizes systems (Oracle, SAP), processes, departments
- **Query Expansion**: Automatically includes related terms
- **Context Understanding**: Considers document structure and metadata

### Semantic Search

- **Vector Embeddings**: Documents represented as high-dimensional vectors
- **Similarity Matching**: Finds conceptually similar content
- **Cross-language Support**: Works with technical terminology
- **Context Preservation**: Maintains meaning across document chunks

### Advanced Filtering

Filter results by:
- Document type (Procedure, Checklist, Guide, Template)
- Category (Invoice Processing, Payment Processing, etc.)
- File type (PDF, DOCX, XLSX)
- Date ranges
- Author
- Version

### Faceted Search

Dynamic facets based on search results:
- Content types found
- Categories present
- File extensions
- Business units
- Keywords and tags

## 📈 Analytics

### Search Analytics

- Query frequency and patterns
- Popular search terms
- User behavior tracking
- Search success rates

### Document Popularity

- Access frequency tracking
- User engagement metrics
- Trending documents
- Usage patterns

### Performance Metrics

- Search response times
- Index size and growth
- Cache hit rates
- System resource usage

## 🚀 Performance Optimization

### Indexing Strategies

- **Incremental Indexing**: Only reindex changed documents
- **Chunk Optimization**: Balance between context and performance
- **Vector Compression**: Reduce memory usage for large document sets
- **Caching**: Intelligent caching of frequent queries

### Search Optimization

- **Query Processing**: Pre-process and cache common queries
- **Result Ranking**: Multi-factor scoring for relevance
- **Filter Efficiency**: Optimized filtering algorithms
- **Memory Management**: Efficient vector operations

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: For large document sets, increase system memory or reduce chunk size in config

3. **Slow Search**: Check vector index size and consider reindexing with optimized settings

4. **Missing Documents**: Verify file paths in document catalog are accessible

### Debug Mode

Enable debug logging:
```bash
python -m search_engine.main --debug --web
```

### Logs

Logs are written to the console. For persistent logging, modify the logging configuration in `main.py`.

## 🤝 Integration

### Existing Catalog Integration

The search engine seamlessly integrates with your existing `document_catalog.json`:

- **Automatic Migration**: Migrates existing metadata to enhanced format
- **Path Resolution**: Resolves file paths for content indexing
- **Metadata Enhancement**: Adds search-specific metadata fields
- **Synchronization**: Keeps indexes synchronized with catalog updates

### API Integration

Complete REST API for integration with other systems:

- **Search Endpoint**: `/api/search`
- **Document Endpoint**: `/api/document/{id}`
- **Suggestions Endpoint**: `/api/suggestions`
- **Analytics Endpoints**: `/api/popular`, `/api/stats`

## 📝 License

This search engine is part of the SCM Governance document management system.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Verify all dependencies are correctly installed
4. Ensure file paths in the document catalog are accessible

---

**Built with ❤️ for SCM Governance Document Management**