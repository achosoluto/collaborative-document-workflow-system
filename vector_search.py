"""
Vector-based semantic search using embeddings
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import faiss

from .config import search_config, VECTOR_STORE_DIR
from .models import DocumentChunk, SearchQuery, SearchResult
from .nlp_processor import TextAnalyzer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Handles text embeddings for semantic search"""

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = search_config.embedding_model

        self.model_name = model_name
        self.model = None
        self.dimensions = search_config.vector_dimensions

        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        return self.encode([text])[0]


class VectorStore:
    """Manages vector storage and similarity search"""

    def __init__(self, store_path: str = None):
        if store_path is None:
            store_path = str(VECTOR_STORE_DIR)

        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model = EmbeddingModel()
        self.text_analyzer = TextAnalyzer()

        # FAISS index for fast similarity search
        self.index = None
        self.chunks = []  # Store chunk metadata
        self.chunk_embeddings = {}

        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index or create new one"""
        index_file = self.store_path / "faiss_index.idx"
        chunks_file = self.store_path / "chunks.pkl"

        if index_file.exists() and chunks_file.exists():
            logger.info("Loading existing vector index...")
            self._load_index()
        else:
            logger.info("Creating new vector index...")
            self._create_index()

    def _create_index(self):
        """Create new FAISS index"""
        # Create FAISS index for cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_model.dimensions)  # Inner product for cosine similarity

        # Save empty index
        self._save_index()

    def _load_index(self):
        """Load existing FAISS index"""
        try:
            self.index = faiss.read_index(str(self.store_path / "faiss_index.idx"))

            with open(self.store_path / "chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)

            logger.info(f"Loaded index with {len(self.chunks)} chunks")

        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            faiss.write_index(self.index, str(self.store_path / "faiss_index.idx"))

            with open(self.store_path / "chunks.pkl", 'wb') as f:
                pickle.dump(self.chunks, f)

        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to vector store"""
        if not chunks:
            return

        logger.info(f"Adding {len(chunks)} chunks to vector store...")

        # Extract texts for embedding
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        start_idx = len(self.chunks)
        self.index.add(embeddings)

        # Store chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()
            chunk.metadata['vector_index'] = start_idx + i
            self.chunks.append(chunk)
            self.chunk_embeddings[chunk.chunk_id] = embeddings[i]

        # Save index
        self._save_index()

        logger.info(f"Added {len(chunks)} chunks. Total chunks: {len(self.chunks)}")

    def search(self, query: str, limit: int = 10, threshold: float = 0.1) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks using semantic similarity"""
        if not query.strip():
            return []

        if len(self.chunks) == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        # Search FAISS index
        scores, indices = self.index.search(query_embedding.reshape(1, -1), limit)

        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold:  # Valid result and above threshold
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))

        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def search_with_query(self, search_query: SearchQuery) -> List[SearchResult]:
        """Perform semantic search with query processing"""
        # Process the query
        query_text = search_query.query_text

        # Get similar chunks
        similar_chunks = self.search(query_text, limit=search_query.limit * 2)  # Get more for ranking

        # Convert to SearchResult objects
        results = []
        for rank, (chunk, score) in enumerate(similar_chunks[:search_query.limit]):
            result = SearchResult(
                doc_id=chunk.doc_id,
                document=None,  # Will be filled by search engine
                rank=rank + 1,
                score=score,
                semantic_score=score,
                matched_chunks=[chunk],
                snippet=self.text_analyzer.extract_snippet(chunk.content, query_text)
            )
            results.append(result)

        return results

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by ID"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_chunks_by_document(self, doc_id: str) -> List[DocumentChunk]:
        """Retrieve all chunks for a document"""
        return [chunk for chunk in self.chunks if chunk.doc_id == doc_id]

    def remove_document(self, doc_id: str):
        """Remove all chunks for a document"""
        # This is a simplified implementation
        # In a production system, you'd need more sophisticated index management
        original_count = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.doc_id != doc_id]

        if len(self.chunks) < original_count:
            logger.info(f"Removed chunks for document {doc_id}")
            self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0,
            'dimensions': self.embedding_model.dimensions,
            'model_name': self.embedding_model.model_name
        }


class SemanticSearchEngine:
    """Main semantic search engine combining vector search with text analysis"""

    def __init__(self):
        self.vector_store = VectorStore()
        self.text_analyzer = TextAnalyzer()

    def search(self, query: str, limit: int = 10, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Perform semantic search"""
        # Create search query object
        search_query = SearchQuery(
            query_text=query,
            limit=limit,
            filters=filters or {}
        )

        return self.vector_store.search_with_query(search_query)

    def hybrid_search(self, search_query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword search"""
        # For now, just do semantic search
        # In a full implementation, this would combine both methods
        return self.vector_store.search_with_query(search_query)

    def index_document(self, doc_id: str, chunks: List[DocumentChunk]):
        """Index document chunks in vector store"""
        self.vector_store.add_chunks(chunks)

    def get_similar_documents(self, doc_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Find documents similar to the given document"""
        # Get chunks for the document
        doc_chunks = self.vector_store.get_chunks_by_document(doc_id)

        if not doc_chunks:
            return []

        # Average embeddings of document chunks
        embeddings = [np.array(chunk.embedding) for chunk in doc_chunks if chunk.embedding]
        if not embeddings:
            return []

        avg_embedding = np.mean(embeddings, axis=0)
        faiss.normalize_L2(avg_embedding.reshape(1, -1))

        # Search for similar documents
        scores, indices = self.vector_store.index.search(avg_embedding.reshape(1, -1), limit * 2)

        # Group by document and calculate average scores
        doc_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.vector_store.chunks):
                chunk = self.vector_store.chunks[idx]
                if chunk.doc_id != doc_id:  # Exclude the same document
                    if chunk.doc_id not in doc_scores:
                        doc_scores[chunk.doc_id] = []
                    doc_scores[chunk.doc_id].append(score)

        # Calculate average scores for each document
        similar_docs = []
        for similar_doc_id, chunk_scores in doc_scores.items():
            avg_score = np.mean(chunk_scores)
            similar_docs.append((similar_doc_id, float(avg_score)))

        # Sort by score and return top results
        similar_docs.sort(key=lambda x: x[1], reverse=True)
        return similar_docs[:limit]

    def get_recommendations(self, user_queries: List[str], limit: int = 5) -> List[str]:
        """Get document recommendations based on user query history"""
        if not user_queries:
            return []

        # Combine recent queries
        combined_query = " ".join(user_queries[-3:])  # Last 3 queries

        # Search for similar content
        results = self.search(combined_query, limit=limit * 2)

        # Extract unique document IDs
        seen_docs = set()
        recommendations = []

        for result in results:
            if result.doc_id not in seen_docs:
                recommendations.append(result.doc_id)
                seen_docs.add(result.doc_id)

                if len(recommendations) >= limit:
                    break

        return recommendations