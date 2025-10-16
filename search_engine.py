"""
Main search engine orchestrating all search capabilities
"""

import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from .config import search_config, DocumentMetadata
from .models import SearchQuery, SearchResult, SearchResponse, SearchAnalytics, DocumentPopularity
from .document_processor import DocumentProcessor, ContentAnalyzer
from .nlp_processor import QueryProcessor, TextAnalyzer
from .vector_search import SemanticSearchEngine

logger = logging.getLogger(__name__)


class DocumentIndex:
    """Manages the document index and metadata"""

    def __init__(self, index_path: str = "data/document_index.json"):
        self.index_path = index_path
        self.documents = {}
        self.popularity_data = {}
        self._load_index()

    def _load_index(self):
        """Load document index from disk"""
        try:
            if os.path.exists(self.index_path):
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', {})
                    self.popularity_data = data.get('popularity', {})
        except Exception as e:
            logger.error(f"Error loading document index: {str(e)}")
            self.documents = {}
            self.popularity_data = {}

    def _save_index(self):
        """Save document index to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, 'w') as f:
                json.dump({
                    'documents': self.documents,
                    'popularity': self.popularity_data,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving document index: {str(e)}")

    def add_document(self, metadata: DocumentMetadata):
        """Add document to index"""
        self.documents[metadata.doc_id] = metadata.__dict__
        self._save_index()

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata"""
        if doc_id not in self.documents:
            return None

        data = self.documents[doc_id]
        return DocumentMetadata(**data)

    def update_document(self, doc_id: str, **updates):
        """Update document metadata"""
        if doc_id in self.documents:
            self.documents[doc_id].update(updates)
            self._save_index()

    def remove_document(self, doc_id: str):
        """Remove document from index"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._save_index()

    def search_documents(self, filters: Dict[str, Any] = None) -> List[str]:
        """Search document index with filters"""
        if not filters:
            return list(self.documents.keys())

        matching_docs = []

        for doc_id, doc_data in self.documents.items():
            match = True

            for key, value in filters.items():
                if key not in doc_data:
                    match = False
                    break

                doc_value = doc_data[key]

                # Handle different filter types
                if isinstance(value, list):
                    if doc_value not in value:
                        match = False
                        break
                elif isinstance(value, str):
                    if value.lower() not in str(doc_value).lower():
                        match = False
                        break
                else:
                    if doc_value != value:
                        match = False
                        break

            if match:
                matching_docs.append(doc_id)

        return matching_docs

    def get_popularity(self, doc_id: str) -> DocumentPopularity:
        """Get popularity data for document"""
        if doc_id not in self.popularity_data:
            self.popularity_data[doc_id] = DocumentPopularity(doc_id=doc_id).__dict__

        data = self.popularity_data[doc_id]
        return DocumentPopularity(**data)

    def update_popularity(self, doc_id: str, **updates):
        """Update popularity data for document"""
        if doc_id not in self.popularity_data:
            self.popularity_data[doc_id] = DocumentPopularity(doc_id=doc_id).__dict__

        self.popularity_data[doc_id].update(updates)

        # Update calculated fields
        popularity = DocumentPopularity(**self.popularity_data[doc_id])
        popularity.update_popularity(datetime.now())
        self.popularity_data[doc_id] = popularity.__dict__

        self._save_index()


class AdvancedSearchEngine:
    """Main search engine combining all search capabilities"""

    def __init__(self):
        self.document_index = DocumentIndex()
        self.document_processor = DocumentProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.query_processor = QueryProcessor()
        self.text_analyzer = TextAnalyzer()
        self.semantic_search = SemanticSearchEngine()

        # Analytics storage
        self.analytics_db = {}

    def index_document(self, file_path: str, doc_id: str = None) -> bool:
        """Index a document for search"""
        try:
            if doc_id is None:
                doc_id = str(uuid.uuid4())

            # Extract document metadata
            metadata = self.document_processor.extract_document_metadata(file_path, doc_id)

            # Extract content
            content, content_metadata = self.document_processor.extract_content(file_path)

            # Analyze content
            keywords = self.content_analyzer.extract_keywords(content)
            entities = self.content_analyzer.extract_entities(content)

            # Update metadata with analysis results
            metadata.tags = keywords[:10]  # Top 10 keywords as tags
            if entities:
                metadata.business_unit = entities.get('departments', [''])[0] or None

            # Create document chunks
            chunks = self.document_processor.chunk_content(content, doc_id)

            # Index in vector store
            self.semantic_search.index_document(doc_id, chunks)

            # Add to document index
            self.document_index.add_document(metadata)

            logger.info(f"Successfully indexed document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing document {file_path}: {str(e)}")
            return False

    def search(self, query: str, limit: int = None, filters: Dict[str, Any] = None,
               use_semantic: bool = True, use_keyword: bool = True) -> SearchResponse:
        """Perform advanced search"""
        start_time = time.time()

        if limit is None:
            limit = search_config.default_search_limit

        # Create search query object
        search_query = SearchQuery(
            query_text=query,
            filters=filters or {},
            limit=limit,
            use_semantic_search=use_semantic,
            use_keyword_search=use_keyword
        )

        # Process the query
        query_analysis = self.query_processor.process_query(query)

        # Get candidate documents from filters
        candidate_docs = self.document_index.search_documents(search_query.filters)

        # Perform semantic search
        semantic_results = []
        if use_semantic:
            semantic_results = self.semantic_search.search(query, limit * 2)

        # Combine and rank results
        all_results = self._combine_search_results(semantic_results, candidate_docs, query_analysis)

        # Apply final ranking and limiting
        final_results = self._rank_and_filter_results(all_results, search_query)

        # Generate snippets and highlights
        for result in final_results:
            if result.matched_chunks:
                result.snippet = self.text_analyzer.extract_snippet(
                    result.matched_chunks[0].content, query
                )
                result.highlights = [self.text_analyzer.highlight_matches(
                    result.matched_chunks[0].content, query
                )]

        # Generate facets
        facets = self._generate_facets(final_results)

        # Create response
        search_time = time.time() - start_time

        response = SearchResponse(
            query=search_query,
            results=final_results,
            total_results=len(final_results),
            search_time=search_time,
            facets=facets,
            search_id=str(uuid.uuid4())
        )

        # Track analytics
        self._track_search_analytics(response)

        return response

    def _combine_search_results(self, semantic_results: List[SearchResult],
                              candidate_docs: List[str],
                              query_analysis: Dict[str, Any]) -> List[SearchResult]:
        """Combine results from different search methods"""
        combined_results = {}

        # Process semantic search results
        for result in semantic_results:
            doc_id = result.doc_id

            # Check if document passes filters
            if candidate_docs and doc_id not in candidate_docs:
                continue

            if doc_id not in combined_results:
                # Get document metadata
                document = self.document_index.get_document(doc_id)
                if document:
                    result.document = document
                    combined_results[doc_id] = result

        # Add keyword-based results for documents not found by semantic search
        if len(combined_results) < len(candidate_docs):
            for doc_id in candidate_docs:
                if doc_id not in combined_results:
                    document = self.document_index.get_document(doc_id)
                    if document:
                        # Create basic result for keyword matches
                        result = SearchResult(
                            doc_id=doc_id,
                            document=document,
                            rank=0,
                            score=0.5,  # Lower score for keyword-only matches
                            keyword_score=0.5
                        )
                        combined_results[doc_id] = result

        return list(combined_results.values())

    def _rank_and_filter_results(self, results: List[SearchResult],
                               search_query: SearchQuery) -> List[SearchResult]:
        """Rank and filter search results"""
        if not results:
            return []

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(results[:search_query.limit]):
            result.rank = i + 1

        # Apply limit
        limited_results = results[:search_query.limit]

        # Boost popular documents
        for result in limited_results:
            popularity = self.document_index.get_popularity(result.doc_id)
            if popularity.popularity_score > 0:
                # Boost score by up to 20% based on popularity
                boost = min(popularity.popularity_score / 100.0, 0.2)
                result.score = min(result.score * (1 + boost), 1.0)

        return limited_results

    def _generate_facets(self, results: List[SearchResult]) -> Dict[str, Dict[str, int]]:
        """Generate search facets from results"""
        facets = {
            'content_type': defaultdict(int),
            'category': defaultdict(int),
            'file_extension': defaultdict(int),
            'business_unit': defaultdict(int)
        }

        for result in results:
            if result.document:
                doc = result.document

                if doc.content_type:
                    facets['content_type'][doc.content_type] += 1
                if doc.category:
                    facets['category'][doc.category] += 1
                if doc.file_extension:
                    facets['file_extension'][doc.file_extension] += 1
                if doc.business_unit:
                    facets['business_unit'][doc.business_unit] += 1

        # Convert to regular dicts
        return {k: dict(v) for k, v in facets.items()}

    def _track_search_analytics(self, response: SearchResponse):
        """Track search analytics"""
        analytics = SearchAnalytics(
            search_id=response.search_id,
            query_text=response.query.query_text,
            timestamp=datetime.now(),
            results_returned=len(response.results),
            search_time=response.search_time
        )

        # Track which results were clicked (would be updated when results are accessed)
        self.analytics_db[response.search_id] = analytics

        # Update document popularity
        for result in response.results:
            self._update_document_popularity(result.doc_id, searched=True)

    def _update_document_popularity(self, doc_id: str, searched: bool = False, clicked: bool = False):
        """Update document popularity metrics"""
        popularity = self.document_index.get_popularity(doc_id)

        if searched:
            popularity.total_searches += 1
        if clicked:
            popularity.total_clicks += 1
            popularity.last_accessed = datetime.now()

        self.document_index.update_popularity(doc_id, **popularity.__dict__)

    def get_popular_documents(self, limit: int = 10) -> List[Tuple[str, DocumentPopularity]]:
        """Get most popular documents"""
        popular_docs = []

        for doc_id in self.document_index.documents.keys():
            popularity = self.document_index.get_popularity(doc_id)
            if popularity.total_searches > 0:
                popular_docs.append((doc_id, popularity))

        # Sort by popularity score
        popular_docs.sort(key=lambda x: x[1].popularity_score, reverse=True)

        return popular_docs[:limit]

    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions based on partial query"""
        suggestions = set()

        # Add suggestions based on document titles and content
        query_lower = partial_query.lower()

        for doc_id, doc_data in self.document_index.documents.items():
            # Check title
            title = doc_data.get('title', '')
            if query_lower in title.lower():
                suggestions.add(title)

            # Check tags/keywords
            tags = doc_data.get('tags', [])
            for tag in tags:
                if query_lower in tag.lower():
                    suggestions.add(tag)

        # Add query expansions
        for term, expansions in self.query_processor.query_expansions.items():
            if partial_query.lower() in term.lower():
                suggestions.update(expansions)

        return list(suggestions)[:10]

    def get_document_preview(self, doc_id: str, query: str = None) -> Dict[str, Any]:
        """Get document preview with snippets"""
        document = self.document_index.get_document(doc_id)
        if not document:
            return {}

        # Get document chunks
        chunks = self.semantic_search.vector_store.get_chunks_by_document(doc_id)

        preview = {
            'document': document,
            'chunks': chunks[:3],  # First 3 chunks
            'total_chunks': len(chunks)
        }

        if query:
            # Generate query-specific snippet
            combined_content = '\n'.join([chunk.content for chunk in chunks[:5]])
            preview['snippet'] = self.text_analyzer.extract_snippet(combined_content, query)
            preview['highlights'] = self.text_analyzer.highlight_matches(combined_content, query)

        return preview

    def reindex_all_documents(self, base_path: str = None) -> int:
        """Reindex all documents from the existing catalog"""
        if base_path is None:
            # Use the existing document catalog to find files
            indexed_count = 0

            for doc_id, doc_data in self.document_index.documents.items():
                file_path = doc_data.get('absolute_path')
                if file_path and os.path.exists(file_path):
                    if self.index_document(file_path, doc_id):
                        indexed_count += 1

            return indexed_count
        else:
            # Index all supported files in directory
            indexed_count = 0
            base_dir = Path(base_path)

            for file_path in base_dir.rglob('*'):
                if file_path.is_file() and self.document_processor.can_process(str(file_path)):
                    if self.index_document(str(file_path)):
                        indexed_count += 1

            return indexed_count

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'total_documents': len(self.document_index.documents),
            'vector_store_stats': self.semantic_search.vector_store.get_stats(),
            'total_searches': len(self.analytics_db),
            'popular_documents_count': len([p for p in self.document_index.popularity_data.values() if p['total_searches'] > 0])
        }