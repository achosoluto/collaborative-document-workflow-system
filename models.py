"""
Data models for the Advanced Document Search Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
try:
    from .config import DocumentMetadata, search_config
except ImportError:
    from config import DocumentMetadata, search_config


@dataclass
class DocumentChunk:
    """Represents a chunk of document content for vector search"""

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int

    # Vector embedding
    embedding: Optional[List[float]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'content': self.content,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'token_count': self.token_count,
            'embedding': self.embedding,
            'metadata': self.metadata
        }


@dataclass
class SearchQuery:
    """Represents a search query with all parameters"""

    query_text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    facets: List[str] = field(default_factory=list)
    limit: int = search_config.default_search_limit
    offset: int = 0

    # Search modes
    use_semantic_search: bool = True
    use_keyword_search: bool = True
    semantic_weight: float = search_config.semantic_search_weight
    keyword_weight: float = search_config.keyword_search_weight

    # Advanced options
    fuzzy_matching: bool = False
    phrase_matching: bool = False
    boost_fields: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'query_text': self.query_text,
            'filters': self.filters,
            'facets': self.facets,
            'limit': self.limit,
            'offset': self.offset,
            'use_semantic_search': self.use_semantic_search,
            'use_keyword_search': self.use_keyword_search,
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight,
            'fuzzy_matching': self.fuzzy_matching,
            'phrase_matching': self.phrase_matching,
            'boost_fields': self.boost_fields
        }


@dataclass
class SearchResult:
    """Represents a single search result"""

    doc_id: str
    document: DocumentMetadata
    rank: int
    score: float

    # Relevance scores
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0

    # Matching information
    matched_chunks: List[DocumentChunk] = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)
    snippet: Optional[str] = None

    # Metadata for display
    relevance_explanation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            'doc_id': self.doc_id,
            'document': self.document.__dict__ if hasattr(self.document, '__dict__') else self.document,
            'rank': self.rank,
            'score': self.score,
            'semantic_score': self.semantic_score,
            'keyword_score': self.keyword_score,
            'combined_score': self.combined_score,
            'matched_chunks': [chunk.to_dict() for chunk in self.matched_chunks],
            'highlights': self.highlights,
            'snippet': self.snippet,
            'relevance_explanation': self.relevance_explanation
        }


@dataclass
class SearchResponse:
    """Complete search response with results and metadata"""

    query: SearchQuery
    results: List[SearchResult]
    total_results: int
    search_time: float
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Pagination
    page: int = 1
    total_pages: int = 1
    has_next: bool = False
    has_prev: bool = False

    # Search analytics
    search_id: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            'query': self.query.to_dict(),
            'results': [result.to_dict() for result in self.results],
            'total_results': self.total_results,
            'search_time': self.search_time,
            'facets': self.facets,
            'pagination': {
                'page': self.page,
                'total_pages': self.total_pages,
                'has_next': self.has_next,
                'has_prev': self.has_prev
            },
            'search_id': self.search_id,
            'suggestions': self.suggestions
        }


@dataclass
class SearchAnalytics:
    """Tracks search usage and popular documents"""

    search_id: str
    query_text: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Query metadata
    filters_used: Dict[str, Any] = field(default_factory=dict)
    results_returned: int = 0
    clicked_results: List[str] = field(default_factory=list)

    # Performance metrics
    search_time: float = 0.0
    documents_accessed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'search_id': self.search_id,
            'query_text': self.query_text,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'filters_used': self.filters_used,
            'results_returned': self.results_returned,
            'clicked_results': self.clicked_results,
            'search_time': self.search_time,
            'documents_accessed': self.documents_accessed
        }


@dataclass
class DocumentPopularity:
    """Tracks document popularity and access patterns"""

    doc_id: str
    total_searches: int = 0
    total_clicks: int = 0
    last_accessed: Optional[datetime] = None

    # Weighted scoring
    search_score: float = 0.0
    click_score: float = 0.0
    recency_score: float = 0.0
    popularity_score: float = 0.0

    # Access patterns
    unique_users: int = 0
    avg_session_duration: float = 0.0

    def update_popularity(self, current_time: datetime) -> None:
        """Update popularity scores based on recent activity"""
        # Time decay factor (newer activity weighted higher)
        hours_since_access = (current_time - self.last_accessed).total_seconds() / 3600 if self.last_accessed else 24

        # Recency score (higher for more recent access)
        self.recency_score = max(0, 100 - hours_since_access)

        # Combined popularity score
        self.popularity_score = (
            self.search_score * 0.3 +
            self.click_score * 0.5 +
            self.recency_score * 0.2
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'doc_id': self.doc_id,
            'total_searches': self.total_searches,
            'total_clicks': self.total_clicks,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'search_score': self.search_score,
            'click_score': self.click_score,
            'recency_score': self.recency_score,
            'popularity_score': self.popularity_score,
            'unique_users': self.unique_users,
            'avg_session_duration': self.avg_session_duration
        }