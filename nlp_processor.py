"""
Natural Language Processing for query understanding and text analysis
"""

import re
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processes natural language queries for better search understanding"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

        # Query expansion dictionary for domain-specific terms
        self.query_expansions = {
            'invoice': ['invoice', 'billing', 'bill', 'statement', 'receipt'],
            'payment': ['payment', 'pay', 'remittance', 'disbursement', 'settlement'],
            'vendor': ['vendor', 'supplier', 'provider', 'contractor', 'seller'],
            'oracle': ['oracle', 'oracle cloud', 'oracle ebs', 'erp system'],
            'sap': ['sap', 'sap system', 'erp system'],
            'helpdesk': ['helpdesk', 'help desk', 'support', 'assistance', 'troubleshooting'],
            'procedure': ['procedure', 'process', 'steps', 'guideline', 'protocol'],
            'checklist': ['checklist', 'verification', 'validation', 'review', 'audit'],
            'approval': ['approval', 'approve', 'authorize', 'authorize', 'sanction'],
            'hold': ['hold', 'block', 'stop', 'prevent', 'restrict'],
            'escalation': ['escalation', 'escalate', 'elevate', 'raise', 'urgent'],
            'reconciliation': ['reconciliation', 'reconcile', 'match', 'balance', 'verify'],
        }

    def process_query(self, query: str) -> Dict[str, any]:
        """
        Process a natural language query and extract search parameters

        Returns:
            Dictionary with processed query components
        """
        if not query or not query.strip():
            return {
                'original_query': '',
                'processed_terms': [],
                'entities': {},
                'intent': 'unknown',
                'filters': {},
                'keywords': []
            }

        # Basic text preprocessing
        processed = self._preprocess_text(query)

        # Extract entities and intent
        entities = self._extract_entities(query)
        intent = self._classify_intent(query, entities)

        # Extract potential filters
        filters = self._extract_filters(query)

        # Expand query with related terms
        expanded_terms = self._expand_query(processed)

        # Extract key phrases
        key_phrases = self._extract_key_phrases(query)

        return {
            'original_query': query,
            'processed_terms': processed,
            'expanded_terms': expanded_terms,
            'entities': entities,
            'intent': intent,
            'filters': filters,
            'key_phrases': key_phrases,
            'keywords': self._extract_keywords(query)
        }

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and lemmatize
        processed = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemma = self.lemmatizer.lemmatize(token)
                processed.append(lemma)

        return processed

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities from query"""
        entities = {
            'systems': [],
            'document_types': [],
            'processes': [],
            'departments': []
        }

        query_lower = query.lower()

        # System entities
        if any(system in query_lower for system in ['oracle cloud', 'oracle ebs', 'sap']):
            if 'oracle' in query_lower:
                entities['systems'].append('Oracle')
            if 'sap' in query_lower:
                entities['systems'].append('SAP')

        # Document type entities
        if any(doc_type in query_lower for doc_type in ['procedure', 'checklist', 'guide', 'template']):
            if 'procedure' in query_lower:
                entities['document_types'].append('Procedure')
            if 'checklist' in query_lower:
                entities['document_types'].append('Checklist')
            if 'guide' in query_lower:
                entities['document_types'].append('Guide')
            if 'template' in query_lower:
                entities['document_types'].append('Template')

        # Process entities
        if any(process in query_lower for process in ['invoice', 'payment', 'vendor', 'approval']):
            if 'invoice' in query_lower:
                entities['processes'].append('Invoice Processing')
            if 'payment' in query_lower:
                entities['processes'].append('Payment Processing')
            if 'vendor' in query_lower:
                entities['processes'].append('Vendor Management')
            if 'approval' in query_lower:
                entities['processes'].append('Approval Process')

        return entities

    def _classify_intent(self, query: str, entities: Dict[str, List[str]]) -> str:
        """Classify the intent of the search query"""
        query_lower = query.lower()

        # Intent classification based on keywords and entities
        if any(word in query_lower for word in ['how to', 'steps to', 'procedure for', 'process for']):
            return 'how_to'
        elif any(word in query_lower for word in ['what is', 'definition', 'meaning', 'explain']):
            return 'definition'
        elif any(word in query_lower for word in ['troubleshoot', 'problem', 'issue', 'error', 'fix']):
            return 'troubleshooting'
        elif any(word in query_lower for word in ['checklist', 'verify', 'validate', 'review']):
            return 'checklist'
        elif any(word in query_lower for word in ['template', 'form', 'sample']):
            return 'template'
        elif 'helpdesk' in query_lower or 'support' in query_lower:
            return 'helpdesk'
        else:
            return 'general_search'

    def _extract_filters(self, query: str) -> Dict[str, any]:
        """Extract potential filter criteria from query"""
        filters = {}
        query_lower = query.lower()

        # File type filters
        if any(ext in query_lower for ext in ['.pdf', 'pdf']):
            filters['file_extension'] = '.pdf'
        elif any(ext in query_lower for ext in ['.docx', '.doc']):
            filters['file_extension'] = ['.docx', '.doc']
        elif any(ext in query_lower for ext in ['.xlsx', '.xls']):
            filters['file_extension'] = ['.xlsx', '.xls']

        # Category filters based on keywords
        if 'invoice' in query_lower:
            filters['category'] = 'Invoice Processing'
        elif 'payment' in query_lower:
            filters['category'] = 'Payment Processing'
        elif 'vendor' in query_lower:
            filters['category'] = 'Vendor Management'
        elif 'helpdesk' in query_lower:
            filters['category'] = 'Helpdesk Procedures'

        # Date/temporal filters (basic)
        current_year = str(datetime.now().year)
        if current_year in query:
            filters['year'] = current_year

        return filters

    def _expand_query(self, processed_terms: List[str]) -> List[str]:
        """Expand query with related terms"""
        expanded = set(processed_terms)

        # Add related terms from expansion dictionary
        for term in processed_terms:
            if term in self.query_expansions:
                expanded.update(self.query_expansions[term])

        return list(expanded)

    def _extract_key_phrases(self, query: str) -> List[str]:
        """Extract key phrases from query"""
        phrases = []

        # Look for quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        phrases.extend(quoted_phrases)

        # Look for camelCase or PascalCase terms
        camel_case = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', query)
        phrases.extend(camel_case)

        # Look for technical terms (acronyms followed by words)
        technical_terms = re.findall(r'\b[A-Z]{2,}(?:\s+[A-Z][a-z]+)*\b', query)
        phrases.extend(technical_terms)

        return phrases

    def _extract_keywords(self, query: str) -> List[Tuple[str, float]]:
        """Extract keywords with weights"""
        processed = self._preprocess_text(query)
        word_freq = Counter(processed)

        # Calculate weights based on frequency and position
        keywords = []
        query_words = query.lower().split()

        for word, freq in word_freq.most_common(10):
            # Boost weight if word appears early in query
            try:
                position = query_words.index(word)
                position_boost = max(0, 1.0 - (position / len(query_words)))
            except ValueError:
                position_boost = 0.5

            weight = (freq / len(processed)) * (1 + position_boost)
            keywords.append((word, weight))

        return keywords


class TextAnalyzer:
    """Analyzes text for similarity and relevance scoring"""

    def __init__(self):
        self.query_processor = QueryProcessor()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        # Simple Jaccard similarity for keywords
        keywords1 = set(self.query_processor._preprocess_text(text1))
        keywords2 = set(self.query_processor._preprocess_text(text2))

        if not keywords1 or not keywords2:
            return 0.0

        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))

        return intersection / union if union > 0 else 0.0

    def extract_snippet(self, content: str, query: str, max_length: int = 300) -> str:
        """Extract relevant snippet from content based on query"""
        if not content or not query:
            return content[:max_length] if content else ""

        query_terms = set(self.query_processor._preprocess_text(query))

        # Split content into sentences
        sentences = sent_tokenize(content)

        # Score sentences based on query term matches
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0

            for term in query_terms:
                if term in sentence_lower:
                    score += 1
                # Bonus for exact matches
                if term in sentence:
                    score += 0.5

            if score > 0:
                scored_sentences.append((sentence, score))

        if not scored_sentences:
            return content[:max_length]

        # Sort by score and position
        scored_sentences.sort(key=lambda x: (-x[1], content.find(x[0])))

        # Build snippet from top sentences
        snippet_parts = []
        current_length = 0

        for sentence, _ in scored_sentences[:3]:  # Top 3 sentences
            if current_length + len(sentence) <= max_length:
                snippet_parts.append(sentence)
                current_length += len(sentence)
            else:
                break

        snippet = ' ... '.join(snippet_parts)

        # Truncate if still too long
        if len(snippet) > max_length:
            snippet = snippet[:max_length-3] + '...'

        return snippet

    def highlight_matches(self, text: str, query: str) -> str:
        """Highlight query terms in text"""
        if not text or not query:
            return text

        # Get query terms
        query_terms = self.query_processor._preprocess_text(query)
        query_terms.extend(self.query_processor._extract_key_phrases(query))

        # Highlight terms (case-insensitive)
        highlighted = text

        for term in set(query_terms):
            if len(term) > 2:  # Only highlight meaningful terms
                # Create case-insensitive regex pattern
                pattern = re.compile(f'({re.escape(term)})', re.IGNORECASE)
                highlighted = pattern.sub(r'<mark>\1</mark>', highlighted)

        return highlighted

    def calculate_relevance_score(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query"""
        if not content or not query:
            return 0.0

        score = 0.0

        # Term frequency scoring
        query_terms = set(self.query_processor._preprocess_text(query))
        content_lower = content.lower()

        term_matches = 0
        for term in query_terms:
            if term in content_lower:
                term_matches += 1

        if query_terms:
            score += (term_matches / len(query_terms)) * 0.6

        # Phrase matching bonus
        key_phrases = self.query_processor._extract_key_phrases(query)
        for phrase in key_phrases:
            if phrase.lower() in content_lower:
                score += 0.3

        # Title/heading bonus (first 200 characters)
        first_part = content[:200]
        for term in query_terms:
            if term in first_part.lower():
                score += 0.1

        return min(score, 1.0)  # Cap at 1.0