"""
Configuration settings for the Advanced Document Search Engine
"""

import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class SearchConfig:
    """Configuration for search engine components"""

    # Document processing
    supported_extensions: List[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Vector search
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimensions: int = 384
    vector_store_path: str = "data/vector_store"

    # Search settings
    default_search_limit: int = 50
    semantic_search_weight: float = 0.7
    keyword_search_weight: float = 0.3

    # Analytics
    analytics_db_path: str = "data/analytics.db"
    popular_docs_cache_size: int = 100

    # Performance
    cache_ttl: int = 3600  # 1 hour
    max_concurrent_searches: int = 10

    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [
                '.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls',
                '.html', '.htm', '.md', '.rtf'
            ]


@dataclass
class DocumentMetadata:
    """Enhanced document metadata for search indexing"""

    doc_id: str
    file_path: str
    absolute_path: str
    file_name: str
    file_extension: str
    file_size_bytes: int
    file_size_human: str
    date_created: str
    date_modified: str
    date_indexed: str

    # Enhanced fields for search
    title: Optional[str] = None
    content_type: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = None
    business_unit: Optional[str] = None
    document_type: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None

    # Search analytics
    search_count: int = 0
    last_accessed: Optional[str] = None
    popularity_score: float = 0.0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# Global configuration instance
search_config = SearchConfig()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / search_config.vector_store_path
ANALYTICS_DB_PATH = BASE_DIR / search_config.analytics_db_path

# Create directories if they don't exist
for dir_path in [DATA_DIR, VECTOR_STORE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)