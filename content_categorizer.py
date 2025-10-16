"""
Automated Content Categorization and Tagging System
Uses ML and NLP to automatically categorize documents and assign relevant tags
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import json
from datetime import datetime

# ML and NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Custom imports
from .content_extractor import ContentProcessingPipeline

logger = logging.getLogger(__name__)


@dataclass
class Category:
    """Represents a document category"""
    id: str
    name: str
    description: str
    keywords: List[str]
    parent_category: Optional[str] = None
    subcategories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'keywords': self.keywords,
            'parent_category': self.parent_category,
            'subcategories': self.subcategories,
            'metadata': self.metadata
        }


@dataclass
class Tag:
    """Represents a document tag"""
    id: str
    name: str
    category: str
    confidence: float = 1.0
    source: str = "automatic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata
        }


class CategoryManager:
    """Manages document categories and their relationships"""

    def __init__(self):
        self.categories = {}
        self.category_hierarchy = {}
        self._load_default_categories()

    def _load_default_categories(self):
        """Load default category structure"""
        default_categories = {
            'business_process': Category(
                id='business_process',
                name='Business Process',
                description='Documents describing business processes and procedures',
                keywords=['process', 'procedure', 'workflow', 'steps', 'methodology', 'guideline']
            ),
            'invoice_processing': Category(
                id='invoice_processing',
                name='Invoice Processing',
                description='Invoice-related processes and procedures',
                keywords=['invoice', 'billing', 'payment', 'accounts payable', 'vendor payment'],
                parent_category='business_process'
            ),
            'payment_processing': Category(
                id='payment_processing',
                name='Payment Processing',
                description='Payment processing procedures and guidelines',
                keywords=['payment', 'disbursement', 'check', 'wire', 'ach', 'settlement'],
                parent_category='business_process'
            ),
            'vendor_management': Category(
                id='vendor_management',
                name='Vendor Management',
                description='Vendor relationship and management processes',
                keywords=['vendor', 'supplier', 'contractor', 'procurement', 'sourcing'],
                parent_category='business_process'
            ),
            'helpdesk': Category(
                id='helpdesk',
                name='Helpdesk Procedures',
                description='Helpdesk and support procedures',
                keywords=['helpdesk', 'support', 'ticket', 'troubleshooting', 'assistance'],
                parent_category='business_process'
            ),
            'compliance': Category(
                id='compliance',
                name='Compliance',
                description='Compliance and regulatory documents',
                keywords=['compliance', 'regulation', 'audit', 'control', 'governance', 'policy']
            ),
            'training': Category(
                id='training',
                name='Training Materials',
                description='Training documents and materials',
                keywords=['training', 'education', 'learning', 'course', 'tutorial', 'guide']
            ),
            'reference': Category(
                id='reference',
                name='Reference Materials',
                description='Reference documents and resources',
                keywords=['reference', 'documentation', 'manual', 'handbook', 'guide']
            ),
            'template': Category(
                id='template',
                name='Templates',
                description='Document templates and forms',
                keywords=['template', 'form', 'sample', 'example', 'blank']
            ),
            'report': Category(
                id='report',
                name='Reports',
                description='Reports and analysis documents',
                keywords=['report', 'analysis', 'summary', 'findings', 'conclusion']
            )
        }

        self.categories = default_categories

        # Build hierarchy
        for cat_id, category in default_categories.items():
            parent = category.parent_category
            if parent:
                if parent not in self.category_hierarchy:
                    self.category_hierarchy[parent] = []
                self.category_hierarchy[parent].append(cat_id)

    def add_category(self, category: Category) -> bool:
        """Add a new category"""
        if category.id in self.categories:
            return False

        self.categories[category.id] = category

        if category.parent_category:
            if category.parent_category not in self.category_hierarchy:
                self.category_hierarchy[category.parent_category] = []
            self.category_hierarchy[category.parent_category].append(category.id)

        return True

    def get_category_path(self, category_id: str) -> List[str]:
        """Get the full path from root to category"""
        path = [category_id]

        current = category_id
        while current and current in self.categories:
            parent = self.categories[current].parent_category
            if parent:
                path.insert(0, parent)
                current = parent
            else:
                break

        return path

    def find_categories_for_keywords(self, keywords: List[str]) -> List[Tuple[str, float]]:
        """Find categories that match given keywords"""
        matches = []

        for cat_id, category in self.categories.items():
            score = self._calculate_category_match_score(category, keywords)
            if score > 0:
                matches.append((cat_id, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _calculate_category_match_score(self, category: Category, keywords: List[str]) -> float:
        """Calculate how well a category matches given keywords"""
        if not keywords:
            return 0.0

        # Direct keyword matches
        category_keywords = set(category.keywords)
        keyword_matches = sum(1 for kw in keywords if kw.lower() in category_keywords)

        # Partial matches (substring)
        partial_matches = 0
        for kw in keywords:
            for cat_kw in category.keywords:
                if kw.lower() in cat_kw.lower() or cat_kw.lower() in kw.lower():
                    partial_matches += 0.5

        # Calculate score
        total_possible = len(keywords)
        score = (keyword_matches + partial_matches) / total_possible

        return min(score, 1.0)


class TaggingEngine:
    """Engine for automatic tag generation"""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Tag categories and their associated terms
        self.tag_categories = {
            'system': {
                'oracle': ['oracle', 'oracle cloud', 'oracle ebs', 'erp'],
                'sap': ['sap', 'sap system', 'erp system'],
                'maximo': ['maximo', 'asset management', 'maintenance'],
                'servicenow': ['servicenow', 'service now', 'ticket system'],
                'microsoft': ['microsoft', 'excel', 'word', 'powerpoint', 'outlook']
            },
            'process_type': {
                'automated': ['automated', 'automation', 'rpa', 'robotic'],
                'manual': ['manual', 'handbook', 'step-by-step'],
                'hybrid': ['hybrid', 'semi-automated', 'combination']
            },
            'complexity': {
                'simple': ['simple', 'basic', 'easy', 'straightforward'],
                'moderate': ['moderate', 'intermediate', 'standard'],
                'complex': ['complex', 'advanced', 'sophisticated', 'intricate']
            },
            'frequency': {
                'daily': ['daily', 'day-to-day', 'routine'],
                'weekly': ['weekly', 'week', 'periodic'],
                'monthly': ['monthly', 'month-end', 'closing'],
                'quarterly': ['quarterly', 'quarter', 'q1', 'q2', 'q3', 'q4'],
                'annual': ['annual', 'yearly', 'year-end']
            },
            'priority': {
                'high': ['high priority', 'critical', 'urgent', 'important'],
                'medium': ['medium priority', 'normal', 'standard'],
                'low': ['low priority', 'nice-to-have', 'optional']
            }
        }

    def generate_tags(self, text: str, categories: List[str] = None) -> List[Tag]:
        """Generate tags for given text"""
        tags = []

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Generate tags from different sources
            keyword_tags = self._generate_keyword_tags(processed_text)
            category_tags = self._generate_category_tags(processed_text, categories or [])
            system_tags = self._generate_system_tags(processed_text)
            process_tags = self._generate_process_tags(processed_text)

            # Combine all tags
            all_tags = keyword_tags + category_tags + system_tags + process_tags

            # Remove duplicates and low-confidence tags
            filtered_tags = self._filter_and_rank_tags(all_tags)

            return filtered_tags

        except Exception as e:
            logger.error(f"Tag generation error: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for tag generation"""
        # Convert to lowercase
        text = text.lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and token.isalnum() and len(token) > 2:
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)

        return ' '.join(processed_tokens)

    def _generate_keyword_tags(self, text: str) -> List[Tag]:
        """Generate tags from important keywords"""
        tags = []

        # Extract meaningful words
        words = word_tokenize(text)
        word_freq = Counter(words)

        # Filter and create tags for frequent meaningful words
        for word, freq in word_freq.most_common(20):
            if (len(word) > 3 and freq > 1 and
                word not in self.stop_words and word.isalnum()):

                tag = Tag(
                    id=f"keyword_{word}",
                    name=word,
                    category="keyword",
                    confidence=min(freq / 10.0, 1.0),  # Normalize confidence
                    metadata={'frequency': freq, 'type': 'keyword'}
                )
                tags.append(tag)

        return tags

    def _generate_category_tags(self, text: str, categories: List[str]) -> List[Tag]:
        """Generate tags based on document categories"""
        tags = []

        for category in categories:
            tag = Tag(
                id=f"category_{category.lower().replace(' ', '_')}",
                name=category,
                category="category",
                confidence=0.8,
                metadata={'type': 'category'}
            )
            tags.append(tag)

        return tags

    def _generate_system_tags(self, text: str) -> List[Tag]:
        """Generate tags for mentioned systems"""
        tags = []

        for system_category, system_keywords in self.tag_categories['system'].items():
            for keyword in system_keywords:
                if keyword in text:
                    tag = Tag(
                        id=f"system_{system_category}",
                        name=system_category,
                        category="system",
                        confidence=0.9,
                        metadata={'matched_keyword': keyword, 'type': 'system'}
                    )
                    tags.append(tag)

        return tags

    def _generate_process_tags(self, text: str) -> List[Tag]:
        """Generate tags for process characteristics"""
        tags = []

        # Check each tag category
        for category, tag_dict in self.tag_categories.items():
            if category == 'system':
                continue  # Already handled

            for tag_name, keywords in tag_dict.items():
                for keyword in keywords:
                    if keyword in text:
                        tag = Tag(
                            id=f"{category}_{tag_name}",
                            name=tag_name,
                            category=category,
                            confidence=0.7,
                            metadata={'matched_keyword': keyword, 'type': category}
                        )
                        tags.append(tag)

        return tags

    def _filter_and_rank_tags(self, tags: List[Tag]) -> List[Tag]:
        """Filter and rank tags by relevance"""
        # Remove duplicates (same name and category)
        seen = set()
        unique_tags = []

        for tag in tags:
            key = (tag.name, tag.category)
            if key not in seen:
                seen.add(key)
                unique_tags.append(tag)

        # Sort by confidence
        unique_tags.sort(key=lambda x: x.confidence, reverse=True)

        # Return top 15 tags
        return unique_tags[:15]


class ContentCategorizer:
    """Main content categorization system"""

    def __init__(self):
        self.content_processor = ContentProcessingPipeline()
        self.category_manager = CategoryManager()
        self.tagging_engine = TaggingEngine()
        self.nlp_model = None

        # Initialize NLP
        self._initialize_nlp()

        # Training data for ML categorization
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.category_classifier = None

        # Document type signatures
        self.type_signatures = self._create_type_signatures()

    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            self.nlp_model = spacy.load('en_core_web_sm')
        except Exception:
            self.nlp_model = None

    def _create_type_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Create signatures for different document types"""
        return {
            'procedure': {
                'keywords': ['procedure', 'process', 'steps', 'guideline', 'instruction'],
                'patterns': [r'step \d+', r'procedure for', r'how to'],
                'structure_indicators': ['numbered lists', 'bullet points', 'flow diagrams']
            },
            'checklist': {
                'keywords': ['checklist', 'verify', 'validate', 'review', 'audit'],
                'patterns': [r'□', r'☐', r'☑', r'checkbox', r'verify that'],
                'structure_indicators': ['checkboxes', 'yes/no questions']
            },
            'template': {
                'keywords': ['template', 'form', 'sample', 'blank', 'format'],
                'patterns': [r'___+', r'\[.*\]', r'{{.*}}'],
                'structure_indicators': ['fill-in blanks', 'placeholders']
            },
            'policy': {
                'keywords': ['policy', 'standard', 'requirement', 'compliance', 'governance'],
                'patterns': [r'must ', r'shall ', r'should ', r'prohibited'],
                'structure_indicators': ['formal language', 'consequences']
            },
            'guide': {
                'keywords': ['guide', 'manual', 'handbook', 'reference', 'documentation'],
                'patterns': [r'see ', r'refer to', r'for more information'],
                'structure_indicators': ['table of contents', 'index']
            }
        }

    def categorize_document(self, file_path: str) -> Dict[str, Any]:
        """
        Categorize a document and assign tags

        Args:
            file_path: Path to the document

        Returns:
            Dictionary with categories, tags, and metadata
        """
        start_time = datetime.now()

        try:
            # Process document
            processed = self.content_processor.process_document(file_path)

            if not processed['success']:
                return {
                    'success': False,
                    'error': f"Document processing failed: {processed.get('error', 'Unknown error')}",
                    'categories': [],
                    'tags': []
                }

            # Extract text for analysis
            text = processed.get('preprocessed_content', processed.get('content', ''))

            if not text or len(text.strip()) < 50:
                return {
                    'success': False,
                    'error': 'Insufficient content for categorization',
                    'categories': [],
                    'tags': []
                }

            # Perform categorization
            categories = self._categorize_content(text, processed)
            tags = self._generate_tags(text, categories)

            # Analyze document structure
            structure_analysis = self._analyze_document_structure(processed)

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(categories, tags, text)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'categories': categories,
                'tags': tags,
                'structure_analysis': structure_analysis,
                'confidence_scores': confidence_scores,
                'metadata': {
                    'file_path': file_path,
                    'file_type': processed.get('file_type', 'unknown'),
                    'processing_time': processing_time,
                    'text_length': len(text),
                    'categorization_timestamp': start_time.isoformat(),
                    'method': 'ml_nlp_hybrid'
                }
            }

        except Exception as e:
            logger.error(f"Categorization error: {e}")
            return {
                'success': False,
                'error': str(e),
                'categories': [],
                'tags': []
            }

    def _categorize_content(self, text: str, processed: Dict) -> List[Dict[str, Any]]:
        """Categorize content using multiple methods"""
        categories = []

        # Method 1: Keyword-based categorization
        keyword_categories = self._categorize_by_keywords(text)
        categories.extend(keyword_categories)

        # Method 2: Structure-based categorization
        structure_categories = self._categorize_by_structure(processed)
        categories.extend(structure_categories)

        # Method 3: ML-based categorization (if trained)
        if self.category_classifier:
            ml_categories = self._categorize_by_ml(text)
            categories.extend(ml_categories)

        # Remove duplicates and sort by score
        unique_categories = self._deduplicate_categories(categories)
        unique_categories.sort(key=lambda x: x['confidence'], reverse=True)

        return unique_categories[:10]  # Top 10 categories

    def _categorize_by_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Categorize using keyword analysis"""
        categories = []

        # Get keywords from text
        keywords = self._extract_keywords(text)

        # Find matching categories
        category_matches = self.category_manager.find_categories_for_keywords(keywords)

        for cat_id, score in category_matches:
            category = self.category_manager.categories[cat_id]

            categories.append({
                'category_id': cat_id,
                'category_name': category.name,
                'confidence': score,
                'method': 'keyword_matching',
                'matched_keywords': [kw for kw in keywords if kw.lower() in category.keywords],
                'category_path': self.category_manager.get_category_path(cat_id)
            })

        return categories

    def _categorize_by_structure(self, processed: Dict) -> List[Dict[str, Any]]:
        """Categorize based on document structure"""
        categories = []
        file_type = processed.get('file_type', 'unknown')

        # Analyze structure indicators
        content = processed.get('content', '')
        structure_score = {}

        for type_name, signature in self.type_signatures.items():
            score = 0

            # Check keywords
            for keyword in signature['keywords']:
                if keyword in content.lower():
                    score += 0.3

            # Check patterns
            for pattern in signature['patterns']:
                if re.search(pattern, content, re.IGNORECASE):
                    score += 0.4

            # Check structure indicators (basic)
            if any(indicator in content.lower() for indicator in signature['structure_indicators']):
                score += 0.3

            if score > 0.2:  # Minimum threshold
                categories.append({
                    'category_id': f'structure_{type_name}',
                    'category_name': type_name.title(),
                    'confidence': score,
                    'method': 'structure_analysis',
                    'matched_patterns': [p for p in signature['patterns']
                                       if re.search(p, content, re.IGNORECASE)],
                    'category_path': [type_name]
                })

        return categories

    def _categorize_by_ml(self, text: str) -> List[Dict[str, Any]]:
        """Categorize using machine learning"""
        # Placeholder for ML-based categorization
        # In a real implementation, this would use a trained classifier
        return []

    def _generate_tags(self, text: str, categories: List[Dict]) -> List[Dict[str, Any]]:
        """Generate tags for the document"""
        # Get category names for context
        category_names = [cat['category_name'] for cat in categories]

        # Generate tags using tagging engine
        tags = self.tagging_engine.generate_tags(text, category_names)

        # Convert to dictionaries
        return [tag.to_dict() for tag in tags]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Tokenize and filter
        tokens = word_tokenize(text.lower())
        filtered_tokens = [
            token for token in tokens
            if token not in self.tagging_engine.stop_words
            and token.isalnum()
            and len(token) > 3
        ]

        # Get most frequent tokens
        word_freq = Counter(filtered_tokens)
        keywords = [word for word, freq in word_freq.most_common(20) if freq > 1]

        return keywords

    def _analyze_document_structure(self, processed: Dict) -> Dict[str, Any]:
        """Analyze document structure for categorization clues"""
        analysis = {
            'has_tables': False,
            'has_images': False,
            'has_lists': False,
            'has_headers': False,
            'formatting_complexity': 'low'
        }

        # Check for tables (from Excel or Word processing)
        if 'tables' in processed and processed['tables']:
            analysis['has_tables'] = True

        # Check for images (from PDF processing)
        if 'images' in processed and processed['images']:
            analysis['has_images'] = True

        # Analyze content structure
        content = processed.get('content', '')

        # Check for list patterns
        list_patterns = [r'^\s*\d+\.', r'^\s*[•·◦]', r'^\s*[a-zA-Z]\.', r'^\s*[-*+]']
        for pattern in list_patterns:
            if re.search(pattern, content, re.MULTILINE):
                analysis['has_lists'] = True
                break

        # Check for header patterns
        header_patterns = [r'^\s*[A-Z][A-Z\s]{10,}$', r'^\s*\d+\.\s*[A-Z]']
        for pattern in header_patterns:
            if re.search(pattern, content, re.MULTILINE):
                analysis['has_headers'] = True
                break

        # Determine formatting complexity
        complexity_indicators = [
            analysis['has_tables'],
            analysis['has_images'],
            analysis['has_lists'],
            analysis['has_headers']
        ]

        complexity_score = sum(complexity_indicators)
        if complexity_score >= 3:
            analysis['formatting_complexity'] = 'high'
        elif complexity_score >= 2:
            analysis['formatting_complexity'] = 'medium'

        return analysis

    def _calculate_confidence_scores(self, categories: List[Dict], tags: List[Dict], text: str) -> Dict[str, float]:
        """Calculate overall confidence scores"""
        # Category confidence
        if categories:
            avg_category_confidence = sum(cat['confidence'] for cat in categories) / len(categories)
        else:
            avg_category_confidence = 0.0

        # Tag confidence
        if tags:
            avg_tag_confidence = sum(tag['confidence'] for tag in tags) / len(tags)
        else:
            avg_tag_confidence = 0.0

        # Overall confidence based on text quality and analysis results
        text_quality_score = min(len(text) / 1000, 1.0)  # Normalize by length

        overall_confidence = (avg_category_confidence * 0.5 +
                            avg_tag_confidence * 0.3 +
                            text_quality_score * 0.2)

        return {
            'overall': overall_confidence,
            'categories': avg_category_confidence,
            'tags': avg_tag_confidence,
            'text_quality': text_quality_score
        }

    def _deduplicate_categories(self, categories: List[Dict]) -> List[Dict]:
        """Remove duplicate categories"""
        seen = set()
        unique = []

        for cat in categories:
            cat_key = (cat['category_id'], cat['method'])

            if cat_key not in seen:
                seen.add(cat_key)
                unique.append(cat)
            else:
                # Merge scores for duplicates
                for existing in unique:
                    if (existing['category_id'] == cat['category_id'] and
                        existing['method'] == cat['method']):
                        existing['confidence'] = max(existing['confidence'], cat['confidence'])
                        break

        return unique

    def train_categorizer(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the ML categorizer with labeled data

        Args:
            training_data: List of {'text': str, 'category': str} dictionaries

        Returns:
            Training results and metrics
        """
        try:
            if not training_data:
                return {'success': False, 'error': 'No training data provided'}

            # Extract texts and labels
            texts = [item['text'] for item in training_data]
            labels = [item['category'] for item in training_data]

            # Fit vectorizer
            X = self.vectorizer.fit_transform(texts)

            # Train classifier (using K-means for demonstration)
            # In production, you'd use a proper supervised classifier
            n_clusters = min(len(set(labels)), 10)  # Reasonable number of clusters
            self.category_classifier = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = self.category_classifier.fit_predict(X)

            # Map clusters to categories
            cluster_to_category = {}
            for cluster, category in zip(clusters, labels):
                if cluster not in cluster_to_category:
                    cluster_to_category[cluster] = category

            return {
                'success': True,
                'training_samples': len(training_data),
                'categories_found': len(set(labels)),
                'clusters_created': n_clusters,
                'vectorizer_features': len(self.vectorizer.get_feature_names_out()),
                'cluster_mapping': cluster_to_category
            }

        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'success': False, 'error': str(e)}

    def add_custom_category(self, category: Category) -> bool:
        """Add a custom category"""
        return self.category_manager.add_category(category)

    def get_available_categories(self) -> List[Dict[str, Any]]:
        """Get all available categories"""
        return [cat.to_dict() for cat in self.category_manager.categories.values()]

    def get_category_hierarchy(self) -> Dict[str, List[str]]:
        """Get category hierarchy"""
        return self.category_manager.category_hierarchy.copy()


# Global categorizer instance
content_categorizer = ContentCategorizer()