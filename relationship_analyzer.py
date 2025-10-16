"""
Document Relationship and Knowledge Gap Analysis System
Analyzes relationships between documents and identifies knowledge gaps
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import json

# ML and NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

# Custom imports
from .content_extractor import ContentProcessingPipeline
from .insight_extractor import KeyInsightExtractor
from .content_categorizer import ContentCategorizer

logger = logging.getLogger(__name__)


@dataclass
class DocumentRelationship:
    """Represents a relationship between two documents"""
    source_doc_id: str
    target_doc_id: str
    relationship_type: str
    strength: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'source_doc_id': self.source_doc_id,
            'target_doc_id': self.target_doc_id,
            'relationship_type': self.relationship_type,
            'strength': self.strength,
            'evidence': self.evidence,
            'metadata': self.metadata
        }


@dataclass
class KnowledgeGap:
    """Represents a knowledge gap in the document collection"""
    gap_type: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_areas: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    related_documents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'gap_type': self.gap_type,
            'description': self.description,
            'severity': self.severity,
            'affected_areas': self.affected_areas,
            'recommended_actions': self.recommended_actions,
            'related_documents': self.related_documents,
            'metadata': self.metadata
        }


class RelationshipAnalyzer:
    """Analyzes relationships between documents"""

    def __init__(self):
        self.content_processor = ContentProcessingPipeline()
        self.insight_extractor = KeyInsightExtractor()
        self.content_categorizer = ContentCategorizer()

        # Embedding model for semantic similarity
        self.embedding_model = None
        self._initialize_embedding_model()

        # Relationship type definitions
        self.relationship_types = {
            'semantic_similarity': 'Documents with similar content or topics',
            'process_dependency': 'Documents that depend on each other in a process',
            'reference': 'One document references another',
            'version': 'Documents that are versions of each other',
            'complementary': 'Documents that complement each other',
            'conflicting': 'Documents with conflicting information',
            'hierarchical': 'Documents in a hierarchical relationship'
        }

    def _initialize_embedding_model(self):
        """Initialize sentence embedding model"""
        try:
            # Use a smaller, efficient model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized sentence embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

    def analyze_document_collection(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze relationships in a collection of documents

        Args:
            document_paths: List of document file paths

        Returns:
            Analysis results with relationships and gaps
        """
        start_time = datetime.now()

        try:
            if len(document_paths) < 2:
                return {
                    'success': False,
                    'error': 'Need at least 2 documents for relationship analysis',
                    'relationships': [],
                    'knowledge_gaps': []
                }

            # Process all documents
            processed_docs = []
            for path in document_paths:
                processed = self.content_processor.process_document(path)
                if processed['success']:
                    processed_docs.append({
                        'file_path': path,
                        'doc_id': self._generate_doc_id(path),
                        'processed': processed
                    })

            if len(processed_docs) < 2:
                return {
                    'success': False,
                    'error': 'Insufficient valid documents for analysis',
                    'relationships': [],
                    'knowledge_gaps': []
                }

            # Extract insights from all documents
            all_insights = []
            for doc in processed_docs:
                insights = self.insight_extractor.extract_insights(doc['file_path'])
                if insights['success']:
                    # Add document ID to insights
                    for insight in insights['insights']:
                        insight.metadata['document_id'] = doc['doc_id']
                    all_insights.extend(insights['insights'])

            # Analyze relationships
            relationships = self._find_all_relationships(processed_docs, all_insights)

            # Identify knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps(processed_docs, all_insights)

            # Generate analysis summary
            summary = self._generate_analysis_summary(relationships, knowledge_gaps)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'relationships': relationships,
                'knowledge_gaps': knowledge_gaps,
                'summary': summary,
                'statistics': {
                    'documents_analyzed': len(processed_docs),
                    'total_relationships': len(relationships),
                    'total_gaps': len(knowledge_gaps),
                    'processing_time': processing_time
                },
                'metadata': {
                    'analysis_timestamp': start_time.isoformat(),
                    'method': 'multi_modal_analysis'
                }
            }

        except Exception as e:
            logger.error(f"Relationship analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'relationships': [],
                'knowledge_gaps': []
            }

    def _find_all_relationships(self, processed_docs: List[Dict], all_insights: List) -> List[DocumentRelationship]:
        """Find all types of relationships between documents"""
        relationships = []

        # Get document pairs
        doc_pairs = [(i, j) for i in range(len(processed_docs))
                    for j in range(i+1, len(processed_docs))]

        for i, j in doc_pairs:
            doc1 = processed_docs[i]
            doc2 = processed_docs[j]

            # Find relationships between this pair
            pair_relationships = self._analyze_document_pair(doc1, doc2, all_insights)
            relationships.extend(pair_relationships)

        return relationships

    def _analyze_document_pair(self, doc1: Dict, doc2: Dict, all_insights: List) -> List[DocumentRelationship]:
        """Analyze relationship between two documents"""
        relationships = []

        doc1_id = doc1['doc_id']
        doc2_id = doc2['doc_id']

        # 1. Semantic similarity analysis
        semantic_rel = self._analyze_semantic_similarity(doc1, doc2)
        if semantic_rel:
            relationships.append(semantic_rel)

        # 2. Process dependency analysis
        process_rel = self._analyze_process_dependencies(doc1, doc2, all_insights)
        if process_rel:
            relationships.extend(process_rel)

        # 3. Reference analysis
        reference_rel = self._analyze_references(doc1, doc2)
        if reference_rel:
            relationships.append(reference_rel)

        # 4. Version relationship analysis
        version_rel = self._analyze_version_relationship(doc1, doc2)
        if version_rel:
            relationships.append(version_rel)

        # 5. Complementary analysis
        complementary_rel = self._analyze_complementary_relationship(doc1, doc2)
        if complementary_rel:
            relationships.append(complementary_rel)

        return relationships

    def _analyze_semantic_similarity(self, doc1: Dict, doc2: Dict) -> Optional[DocumentRelationship]:
        """Analyze semantic similarity between documents"""
        try:
            # Extract text content
            text1 = doc1['processed'].get('preprocessed_content', doc1['processed'].get('content', ''))
            text2 = doc2['processed'].get('preprocessed_content', doc2['processed'].get('content', ''))

            if not text1 or not text2 or len(text1) < 50 or len(text2) < 50:
                return None

            # Use embedding model if available
            if self.embedding_model:
                # Create embeddings for key sentences
                sentences1 = sent_tokenize(text1)[:10]  # First 10 sentences
                sentences2 = sent_tokenize(text2)[:10]

                if sentences1 and sentences2:
                    embeddings1 = self.embedding_model.encode(sentences1)
                    embeddings2 = self.embedding_model.encode(sentences2)

                    # Calculate cosine similarity
                    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
                    avg_similarity = np.mean(similarity_matrix)

                    if avg_similarity > 0.3:  # Threshold for relationship
                        return DocumentRelationship(
                            source_doc_id=doc1['doc_id'],
                            target_doc_id=doc2['doc_id'],
                            relationship_type='semantic_similarity',
                            strength=avg_similarity,
                            evidence=[
                                f'Average semantic similarity: {avg_similarity".3f"}',
                                f'Common topics detected between documents'
                            ],
                            metadata={
                                'similarity_method': 'sentence_transformers',
                                'sentences_compared': min(len(sentences1), len(sentences2))
                            }
                        )
            else:
                # Fallback to simple keyword similarity
                similarity = self._calculate_keyword_similarity(text1, text2)
                if similarity > 0.2:
                    return DocumentRelationship(
                        source_doc_id=doc1['doc_id'],
                        target_doc_id=doc2['doc_id'],
                        relationship_type='semantic_similarity',
                        strength=similarity,
                        evidence=['Keyword-based similarity detected'],
                        metadata={'similarity_method': 'keyword_overlap'}
                    )

        except Exception as e:
            logger.warning(f"Semantic similarity analysis failed: {e}")

        return None

    def _calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple keyword-based similarity"""
        # Extract keywords
        keywords1 = set(re.findall(r'\b\w{4,}\b', text1.lower()))
        keywords2 = set(re.findall(r'\b\w{4,}\b', text2.lower()))

        if not keywords1 or not keywords2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))

        return intersection / union if union > 0 else 0.0

    def _analyze_process_dependencies(self, doc1: Dict, doc2: Dict, all_insights: List) -> List[DocumentRelationship]:
        """Analyze process dependencies between documents"""
        relationships = []

        # Get insights for each document
        doc1_insights = [i for i in all_insights if i.metadata.get('document_id') == doc1['doc_id']]
        doc2_insights = [i for i in all_insights if i.metadata.get('document_id') == doc2['doc_id']]

        # Look for dependency indicators
        for insight1 in doc1_insights:
            for insight2 in doc2_insights:
                # Check if insights indicate a dependency
                if self._insights_indicate_dependency(insight1, insight2):
                    strength = self._calculate_dependency_strength(insight1, insight2)

                    relationships.append(DocumentRelationship(
                        source_doc_id=doc1['doc_id'],
                        target_doc_id=doc2['doc_id'],
                        relationship_type='process_dependency',
                        strength=strength,
                        evidence=[
                            f'Dependency indicated between {insight1.insight_type.value} and {insight2.insight_type.value}',
                            f'Process flow suggests {doc1["file_path"]} -> {doc2["file_path"]}'
                        ],
                        metadata={
                            'source_insight': insight1.content[:100],
                            'target_insight': insight2.content[:100]
                        }
                    ))

        return relationships

    def _insights_indicate_dependency(self, insight1, insight2) -> bool:
        """Check if two insights indicate a dependency"""
        # Simple heuristic: look for sequential or conditional language
        dependency_keywords = ['after', 'before', 'next', 'then', 'requires', 'depends']

        combined_content = (insight1.content + ' ' + insight2.content).lower()

        return any(keyword in combined_content for keyword in dependency_keywords)

    def _calculate_dependency_strength(self, insight1, insight2) -> float:
        """Calculate strength of dependency relationship"""
        # Base strength on insight types and content similarity
        type_weight = {
            'process': 0.8,
            'requirement': 0.7,
            'dependency': 0.9,
            'milestone': 0.6
        }

        weight1 = type_weight.get(insight1.insight_type.value, 0.5)
        weight2 = type_weight.get(insight2.insight_type.value, 0.5)

        # Boost if insights are similar
        similarity = self._calculate_keyword_similarity(insight1.content, insight2.content)
        similarity_bonus = similarity * 0.3

        return min(weight1 * weight2 + similarity_bonus, 1.0)

    def _analyze_references(self, doc1: Dict, doc2: Dict) -> Optional[DocumentRelationship]:
        """Analyze if one document references another"""
        try:
            text1 = doc1['processed'].get('content', '').lower()
            text2 = doc2['processed'].get('content', '').lower()

            # Look for file path references
            file1_name = os.path.basename(doc1['file_path']).lower()
            file2_name = os.path.basename(doc2['file_path']).lower()

            # Check if file2 is mentioned in file1
            if (file2_name in text1 or
                os.path.splitext(file2_name)[0] in text1):

                return DocumentRelationship(
                    source_doc_id=doc1['doc_id'],
                    target_doc_id=doc2['doc_id'],
                    relationship_type='reference',
                    strength=0.8,
                    evidence=[f'Document references: {file2_name}'],
                    metadata={'reference_type': 'filename_mention'}
                )

        except Exception as e:
            logger.warning(f"Reference analysis failed: {e}")

        return None

    def _analyze_version_relationship(self, doc1: Dict, doc2: Dict) -> Optional[DocumentRelationship]:
        """Analyze if documents are versions of each other"""
        try:
            # Look for version indicators in filenames
            name1 = os.path.basename(doc1['file_path']).lower()
            name2 = os.path.basename(doc2['file_path']).lower()

            # Check for version patterns
            version_patterns = [
                (r'v?\d+\.\d+', r'v?\d+\.\d+'),  # Version numbers
                (r'rev[ision]*\s*\d+', r'rev[ision]*\s*\d+'),  # Revisions
                (r'update[sd]*', r'update[sd]*'),  # Updates
                (r'final', r'draft'),  # Draft/Final
            ]

            for pattern1, pattern2 in version_patterns:
                if (re.search(pattern1, name1) and re.search(pattern2, name2)) or \
                   (re.search(pattern2, name1) and re.search(pattern1, name2)):
                    return DocumentRelationship(
                        source_doc_id=doc1['doc_id'],
                        target_doc_id=doc2['doc_id'],
                        relationship_type='version',
                        strength=0.7,
                        evidence=['Version indicators found in filenames'],
                        metadata={'version_patterns': [pattern1, pattern2]}
                    )

        except Exception as e:
            logger.warning(f"Version analysis failed: {e}")

        return None

    def _analyze_complementary_relationship(self, doc1: Dict, doc2: Dict) -> Optional[DocumentRelationship]:
        """Analyze if documents complement each other"""
        try:
            # Check if documents have different but related categories
            cat1 = doc1['processed'].get('file_type', '')
            cat2 = doc2['processed'].get('file_type', '')

            # Define complementary type pairs
            complementary_pairs = [
                ('pdf', 'xlsx'),  # Document and data
                ('docx', 'pdf'),  # Editable and published
                ('procedure', 'checklist'),  # Process and verification
            ]

            type1 = os.path.splitext(doc1['file_path'])[1][1:]  # Remove dot
            type2 = os.path.splitext(doc2['file_path'])[1][1:]

            if (type1, type2) in complementary_pairs or (type2, type1) in complementary_pairs:
                return DocumentRelationship(
                    source_doc_id=doc1['doc_id'],
                    target_doc_id=doc2['doc_id'],
                    relationship_type='complementary',
                    strength=0.6,
                    evidence=[f'Complementary file types: {type1} and {type2}'],
                    metadata={'complementary_types': [type1, type2]}
                )

        except Exception as e:
            logger.warning(f"Complementary analysis failed: {e}")

        return None

    def _identify_knowledge_gaps(self, processed_docs: List[Dict], all_insights: List) -> List[KnowledgeGap]:
        """Identify knowledge gaps in the document collection"""
        gaps = []

        # 1. Coverage gaps - missing document types
        coverage_gaps = self._identify_coverage_gaps(processed_docs)
        gaps.extend(coverage_gaps)

        # 2. Process gaps - incomplete processes
        process_gaps = self._identify_process_gaps(all_insights)
        gaps.extend(process_gaps)

        # 3. Currency gaps - outdated information
        currency_gaps = self._identify_currency_gaps(processed_docs)
        gaps.extend(currency_gaps)

        # 4. Quality gaps - inconsistent or conflicting information
        quality_gaps = self._identify_quality_gaps(processed_docs, all_insights)
        gaps.extend(quality_gaps)

        return gaps

    def _identify_coverage_gaps(self, processed_docs: List[Dict]) -> List[KnowledgeGap]:
        """Identify gaps in document coverage"""
        gaps = []

        # Analyze document types present
        doc_types = Counter(doc['processed'].get('file_type', 'unknown') for doc in processed_docs)

        # Define expected document types for business processes
        expected_types = {
            'procedure': 'Process documentation',
            'checklist': 'Verification checklists',
            'template': 'Document templates',
            'policy': 'Policy documents',
            'guide': 'User guides'
        }

        # Check for missing types
        for doc_type, description in expected_types.items():
            if doc_type not in doc_types or doc_types[doc_type] == 0:
                gaps.append(KnowledgeGap(
                    gap_type='missing_document_type',
                    description=f'No {description.lower()} found in document collection',
                    severity='medium',
                    affected_areas=[doc_type],
                    recommended_actions=[
                        f'Create {description.lower()} to document {doc_type} procedures',
                        f'Review existing documentation for {doc_type} coverage'
                    ],
                    metadata={'missing_type': doc_type}
                ))

        return gaps

    def _identify_process_gaps(self, all_insights: List) -> List[KnowledgeGap]:
        """Identify gaps in process documentation"""
        gaps = []

        # Count insight types
        insight_types = Counter(insight.insight_type.value for insight in all_insights)

        # Look for missing process components
        if insight_types.get('process', 0) == 0:
            gaps.append(KnowledgeGap(
                gap_type='missing_process_documentation',
                description='No process documentation found',
                severity='high',
                affected_areas=['process_documentation'],
                recommended_actions=[
                    'Document key business processes',
                    'Create process flow diagrams',
                    'Define process steps and responsibilities'
                ]
            ))

        if insight_types.get('requirement', 0) == 0:
            gaps.append(KnowledgeGap(
                gap_type='missing_requirements',
                description='No requirement specifications found',
                severity='high',
                affected_areas=['requirements'],
                recommended_actions=[
                    'Document business requirements',
                    'Create requirement specifications',
                    'Define acceptance criteria'
                ]
            ))

        return gaps

    def _identify_currency_gaps(self, processed_docs: List[Dict]) -> List[KnowledgeGap]:
        """Identify outdated or missing timestamp information"""
        gaps = []

        outdated_docs = 0
        total_docs = len(processed_docs)

        for doc in processed_docs:
            # Check if document has modification date
            metadata = doc['processed'].get('metadata', {})
            if not metadata or 'modified' not in metadata:
                outdated_docs += 1

        if outdated_docs / total_docs > 0.5:  # More than half lack dates
            gaps.append(KnowledgeGap(
                gap_type='missing_temporal_metadata',
                description=f'{outdated_docs} out of {total_docs} documents lack date information',
                severity='low',
                affected_areas=['metadata', 'version_control'],
                recommended_actions=[
                    'Add creation and modification dates to all documents',
                    'Implement document review cycles',
                    'Track document update history'
                ]
            ))

        return gaps

    def _identify_quality_gaps(self, processed_docs: List[Dict], all_insights: List) -> List[KnowledgeGap]:
        """Identify quality issues in documentation"""
        gaps = []

        # Check for conflicting information
        conflicts = self._detect_conflicts(processed_docs, all_insights)
        if conflicts:
            gaps.append(KnowledgeGap(
                gap_type='conflicting_information',
                description=f'Found {len(conflicts)} potential conflicts in documentation',
                severity='medium',
                affected_areas=['consistency', 'quality'],
                recommended_actions=[
                    'Review conflicting documents',
                    'Standardize terminology and procedures',
                    'Create governance process for documentation'
                ],
                related_documents=[conflict['doc_ids'] for conflict in conflicts]
            ))

        return gaps

    def _detect_conflicts(self, processed_docs: List[Dict], all_insights: List) -> List[Dict]:
        """Detect conflicting information between documents"""
        conflicts = []

        # Simple conflict detection based on contradictory keywords
        contradictory_pairs = [
            ('approved', 'draft'),
            ('current', 'obsolete'),
            ('active', 'inactive'),
            ('required', 'optional')
        ]

        for doc in processed_docs:
            text = doc['processed'].get('content', '').lower()

            for pos, neg in contradictory_pairs:
                if pos in text and neg in text:
                    conflicts.append({
                        'doc_ids': [doc['doc_id']],
                        'conflict_type': 'contradictory_terms',
                        'terms': [pos, neg]
                    })

        return conflicts

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a document ID from file path"""
        import hashlib
        path_str = os.path.abspath(file_path)
        return hashlib.md5(path_str.encode()).hexdigest()[:12]

    def _generate_analysis_summary(self, relationships: List[DocumentRelationship],
                                 knowledge_gaps: List[KnowledgeGap]) -> Dict[str, Any]:
        """Generate summary of analysis results"""
        # Count relationship types
        rel_types = Counter(rel.relationship_type for rel in relationships)

        # Count gap severities
        gap_severities = Counter(gap.severity for gap in knowledge_gaps)

        # Generate insights
        insights = []

        if rel_types.get('semantic_similarity', 0) > 0:
            insights.append(f"Found {rel_types['semantic_similarity']} semantic relationships between documents")

        if rel_types.get('process_dependency', 0) > 0:
            insights.append(f"Identified {rel_types['process_dependency']} process dependencies")

        if gap_severities.get('high', 0) > 0:
            insights.append(f"Found {gap_severities['high']} high-priority knowledge gaps")

        return {
            'total_relationships': len(relationships),
            'total_gaps': len(knowledge_gaps),
            'relationship_types': dict(rel_types),
            'gap_severities': dict(gap_severities),
            'key_insights': insights,
            'analysis_quality': 'good' if len(insights) > 0 else 'limited'
        }


class KnowledgeGapAnalyzer:
    """Specialized analyzer for knowledge gaps"""

    def __init__(self):
        self.relationship_analyzer = RelationshipAnalyzer()

    def analyze_knowledge_architecture(self, document_paths: List[str]) -> Dict[str, Any]:
        """Analyze the knowledge architecture of document collection"""
        # Use relationship analyzer
        results = self.relationship_analyzer.analyze_document_collection(document_paths)

        if not results['success']:
            return results

        # Enhance gap analysis with architectural insights
        enhanced_gaps = self._enhance_gap_analysis(results['knowledge_gaps'], results['relationships'])

        return {
            **results,
            'knowledge_gaps': enhanced_gaps,
            'architecture_insights': self._generate_architecture_insights(results)
        }

    def _enhance_gap_analysis(self, gaps: List[KnowledgeGap], relationships: List[DocumentRelationship]) -> List[KnowledgeGap]:
        """Enhance gap analysis with relationship context"""
        enhanced = []

        for gap in gaps:
            # Add related documents based on relationships
            related_docs = set(gap.related_documents)

            for rel in relationships:
                if (gap.gap_type == 'missing_document_type' and
                    rel.relationship_type in ['semantic_similarity', 'complementary']):
                    related_docs.add(rel.source_doc_id)
                    related_docs.add(rel.target_doc_id)

            gap.related_documents = list(related_docs)
            enhanced.append(gap)

        return enhanced

    def _generate_architecture_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights about knowledge architecture"""
        insights = []

        relationships = results.get('relationships', [])
        gaps = results.get('knowledge_gaps', [])

        # Analyze connectivity
        if relationships:
            avg_strength = sum(rel.strength for rel in relationships) / len(relationships)
            insights.append(f"Average relationship strength: {avg_strength".2f"}")

        # Analyze gap patterns
        if gaps:
            gap_types = Counter(gap.gap_type for gap in gaps)
            most_common_gap = max(gap_types.items(), key=lambda x: x[1])
            insights.append(f"Most common gap type: {most_common_gap[0]} ({most_common_gap[1]} instances)")

        return insights


# Global relationship analyzer instance
relationship_analyzer = RelationshipAnalyzer()
knowledge_gap_analyzer = KnowledgeGapAnalyzer()