"""
Advanced Key Insight Extraction System
Extracts processes, requirements, risks, dependencies, and other key insights from documents
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

# NLP and AI libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk import pos_tag
import spacy

# Custom imports
from .content_extractor import ContentProcessingPipeline

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of insights that can be extracted"""
    PROCESS = "process"
    REQUIREMENT = "requirement"
    RISK = "risk"
    DEPENDENCY = "dependency"
    DECISION = "decision"
    ASSUMPTION = "assumption"
    CONSTRAINT = "constraint"
    METRIC = "metric"
    STAKEHOLDER = "stakeholder"
    DELIVERABLE = "deliverable"
    MILESTONE = "milestone"
    ISSUE = "issue"


class ConfidenceLevel(Enum):
    """Confidence levels for extracted insights"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ExtractedInsight:
    """Represents an extracted insight"""
    insight_type: InsightType
    content: str
    confidence: ConfidenceLevel
    location: Dict[str, Any] = field(default_factory=dict)
    context: str = ""
    related_insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'type': self.insight_type.value,
            'content': self.content,
            'confidence': self.confidence.value,
            'location': self.location,
            'context': self.context,
            'related_insights': self.related_insights,
            'metadata': self.metadata
        }


class InsightPattern:
    """Pattern for identifying specific types of insights"""

    def __init__(self, insight_type: InsightType, patterns: List[str], keywords: List[str],
                 context_window: int = 50):
        self.insight_type = insight_type
        self.patterns = patterns
        self.keywords = keywords
        self.context_window = context_window
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def match(self, text: str, position: int) -> Optional[Tuple[str, int, int]]:
        """Check if text matches pattern at given position"""
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                start, end = match.span()
                return text[start:end], start, end
        return None


class KeyInsightExtractor:
    """Main class for extracting key insights from documents"""

    def __init__(self):
        self.content_processor = ContentProcessingPipeline()
        self.nlp_model = None

        # Initialize NLP components
        self._initialize_nlp()

        # Define insight patterns
        self.insight_patterns = self._create_insight_patterns()

        # Context and relationship tracking
        self.context_window = 100  # Characters of context around each insight

    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            # Load spaCy model for advanced NLP
            try:
                self.nlp_model = spacy.load('en_core_web_sm')
            except OSError:
                logger.info("Downloading spaCy model...")
                import os
                os.system('python -m spacy download en_core_web_sm')
                self.nlp_model = spacy.load('en_core_web_sm')

        except Exception as e:
            logger.error(f"Error initializing NLP: {e}")
            self.nlp_model = None

    def _create_insight_patterns(self) -> Dict[InsightType, InsightPattern]:
        """Create patterns for different insight types"""
        patterns = {}

        # Process patterns
        patterns[InsightType.PROCESS] = InsightPattern(
            InsightType.PROCESS,
            [
                r'process[:\s]+([^.]+)',
                r'procedure[:\s]+([^.]+)',
                r'steps?[:\s]+([^.]+)',
                r'workflow[:\s]+([^.]+)',
                r'methodology[:\s]+([^.]+)'
            ],
            ['process', 'procedure', 'workflow', 'steps', 'method', 'approach']
        )

        # Requirement patterns
        patterns[InsightType.REQUIREMENT] = InsightPattern(
            InsightType.REQUIREMENT,
            [
                r'(?:must|shall|should|will)[\s\w]+([^.]+)',
                r'requirement[:\s]+([^.]+)',
                r'required[:\s]+([^.]+)',
                r'needs? to ([^.]+)',
                r'has to ([^.]+)'
            ],
            ['requirement', 'must', 'shall', 'should', 'need', 'required']
        )

        # Risk patterns
        patterns[InsightType.RISK] = InsightPattern(
            InsightType.RISK,
            [
                r'risk[:\s]+([^.]+)',
                r'(?:if|when|should) ([^.]*(?:fail|error|problem|issue)[^.]*[^.]*)',
                r'potential ([^.]*(?:issue|problem|concern|risk)[^.]*[^.]*)',
                r'concern[:\s]+([^.]+)'
            ],
            ['risk', 'issue', 'problem', 'concern', 'challenge', 'threat']
        )

        # Dependency patterns
        patterns[InsightType.DEPENDENCY] = InsightPattern(
            InsightType.DEPENDENCY,
            [
                r'depends? on ([^.]+)',
                r'relies? on ([^.]+)',
                r'requires? ([^.]*(?:completion|delivery|approval)[^.]*[^.]*)',
                r'prerequisite[:\s]+([^.]+)',
                r'before ([^.]+)'
            ],
            ['depends', 'relies', 'requires', 'prerequisite', 'before', 'after']
        )

        # Decision patterns
        patterns[InsightType.DECISION] = InsightPattern(
            InsightType.DECISION,
            [
                r'decision[:\s]+([^.]+)',
                r'decided to ([^.]+)',
                r'approved ([^.]+)',
                r'chosen ([^.]+)',
                r'selected ([^.]+)'
            ],
            ['decision', 'decided', 'approved', 'chosen', 'selected', 'determined']
        )

        # Assumption patterns
        patterns[InsightType.ASSUMPTION] = InsightPattern(
            InsightType.ASSUMPTION,
            [
                r'assum(?:e|ing)[:\s]+([^.]+)',
                r'assumption[:\s]+([^.]+)',
                r'presum(?:e|ing)[:\s]+([^.]+)',
                r'expected that ([^.]+)'
            ],
            ['assume', 'assuming', 'assumption', 'presume', 'expected']
        )

        # Constraint patterns
        patterns[InsightType.CONSTRAINT] = InsightPattern(
            InsightType.CONSTRAINT,
            [
                r'constraint[:\s]+([^.]+)',
                r'limitation[:\s]+([^.]+)',
                r'restricted to ([^.]+)',
                r'cannot ([^.]+)',
                r'limited to ([^.]+)'
            ],
            ['constraint', 'limitation', 'restricted', 'cannot', 'limited']
        )

        # Metric patterns
        patterns[InsightType.METRIC] = InsightPattern(
            InsightType.METRIC,
            [
                r'(?:target|goal)[:\s]+([^.]+)',
                r'(?:measure|metric)[:\s]+([^.]+)',
                r'kpi[:\s]+([^.]+)',
                r'(?:\d+%|\d+x|\$[\d,]+) ([^.]+)'
            ],
            ['target', 'goal', 'measure', 'metric', 'kpi', 'objective']
        )

        # Stakeholder patterns
        patterns[InsightType.STAKEHOLDER] = InsightPattern(
            InsightType.STAKEHOLDER,
            [
                r'stakeholder[:\s]+([^.]+)',
                r'responsible[:\s]+([^.]+)',
                r'owner[:\s]+([^.]+)',
                r'accountable[:\s]+([^.]+)'
            ],
            ['stakeholder', 'responsible', 'owner', 'accountable', 'team']
        )

        # Deliverable patterns
        patterns[InsightType.DELIVERABLE] = InsightPattern(
            InsightType.DELIVERABLE,
            [
                r'deliverable[:\s]+([^.]+)',
                r'output[:\s]+([^.]+)',
                r'produce[:\s]+([^.]+)',
                r'create[:\s]+([^.]+)'
            ],
            ['deliverable', 'output', 'produce', 'create', 'generate']
        )

        return patterns

    def extract_insights(self, file_path: str) -> Dict[str, Any]:
        """
        Extract all types of insights from a document

        Args:
            file_path: Path to the document

        Returns:
            Dictionary containing extracted insights and metadata
        """
        start_time = datetime.now()

        try:
            # Process document
            processed = self.content_processor.process_document(file_path)

            if not processed['success']:
                return {
                    'success': False,
                    'error': f"Document processing failed: {processed.get('error', 'Unknown error')}",
                    'insights': [],
                    'statistics': {}
                }

            # Extract text content
            text = processed.get('preprocessed_content', processed.get('content', ''))

            if not text or len(text.strip()) < 50:
                return {
                    'success': False,
                    'error': 'Insufficient content for insight extraction',
                    'insights': [],
                    'statistics': {}
                }

            # Extract insights by type
            all_insights = []
            insights_by_type = {insight_type: [] for insight_type in InsightType}

            for insight_type, pattern in self.insight_patterns.items():
                type_insights = self._extract_insights_by_type(text, insight_type, pattern, processed)
                all_insights.extend(type_insights)
                insights_by_type[insight_type] = type_insights

            # Analyze relationships between insights
            relationships = self._analyze_insight_relationships(all_insights)

            # Calculate statistics
            statistics = self._calculate_insight_statistics(all_insights, text)

            # Create summary
            summary = self._create_insights_summary(all_insights)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'insights': all_insights,
                'insights_by_type': {k.value: v for k, v in insights_by_type.items()},
                'relationships': relationships,
                'summary': summary,
                'statistics': statistics,
                'metadata': {
                    'file_path': file_path,
                    'file_type': processed.get('file_type', 'unknown'),
                    'processing_time': processing_time,
                    'text_length': len(text),
                    'extraction_timestamp': start_time.isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Insight extraction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'insights': [],
                'statistics': {}
            }

    def _extract_insights_by_type(self, text: str, insight_type: InsightType,
                                pattern: InsightPattern, processed: Dict) -> List[ExtractedInsight]:
        """Extract insights of a specific type"""
        insights = []

        try:
            # Method 1: Pattern-based extraction
            pattern_insights = self._extract_with_patterns(text, insight_type, pattern)
            insights.extend(pattern_insights)

            # Method 2: Keyword-based extraction
            keyword_insights = self._extract_with_keywords(text, insight_type, pattern)
            insights.extend(keyword_insights)

            # Method 3: NLP-based extraction (if available)
            if self.nlp_model:
                nlp_insights = self._extract_with_nlp(text, insight_type, pattern)
                insights.extend(nlp_insights)

            # Remove duplicates and low-confidence insights
            filtered_insights = self._filter_insights(insights)

            # Add context and metadata
            for insight in filtered_insights:
                self._enhance_insight(insight, text, processed)

            return filtered_insights

        except Exception as e:
            logger.error(f"Error extracting {insight_type.value} insights: {e}")
            return []

    def _extract_with_patterns(self, text: str, insight_type: InsightType,
                             pattern: InsightPattern) -> List[ExtractedInsight]:
        """Extract insights using regex patterns"""
        insights = []

        for compiled_pattern in pattern.compiled_patterns:
            matches = list(compiled_pattern.finditer(text))

            for match in matches:
                content = match.group(0).strip()
                start_pos = match.start()
                end_pos = match.end()

                # Calculate confidence based on match quality
                confidence = self._calculate_pattern_confidence(content, pattern)

                # Extract context
                context_start = max(0, start_pos - self.context_window)
                context_end = min(len(text), end_pos + self.context_window)
                context = text[context_start:context_end].strip()

                insight = ExtractedInsight(
                    insight_type=insight_type,
                    content=content,
                    confidence=confidence,
                    location={
                        'start': start_pos,
                        'end': end_pos,
                        'context_start': context_start,
                        'context_end': context_end
                    },
                    context=context,
                    metadata={
                        'extraction_method': 'pattern_matching',
                        'pattern': compiled_pattern.pattern
                    }
                )

                insights.append(insight)

        return insights

    def _extract_with_keywords(self, text: str, insight_type: InsightType,
                             pattern: InsightPattern) -> List[ExtractedInsight]:
        """Extract insights using keyword analysis"""
        insights = []

        # Split text into sentences
        sentences = sent_tokenize(text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Check if sentence contains relevant keywords
            keyword_matches = [kw for kw in pattern.keywords if kw in sentence_lower]

            if keyword_matches:
                # Calculate confidence based on keyword matches
                confidence_score = min(len(keyword_matches) / 3.0, 1.0)  # Normalize to 0-1
                confidence = self._score_to_confidence(confidence_score)

                # Find position in original text
                start_pos = text.find(sentence)
                if start_pos != -1:
                    end_pos = start_pos + len(sentence)

                    insight = ExtractedInsight(
                        insight_type=insight_type,
                        content=sentence.strip(),
                        confidence=confidence,
                        location={
                            'start': start_pos,
                            'end': end_pos
                        },
                        context=sentence.strip(),
                        metadata={
                            'extraction_method': 'keyword_matching',
                            'matched_keywords': keyword_matches
                        }
                    )

                    insights.append(insight)

        return insights

    def _extract_with_nlp(self, text: str, insight_type: InsightType,
                         pattern: InsightPattern) -> List[ExtractedInsight]:
        """Extract insights using NLP techniques"""
        insights = []

        try:
            # Process text with spaCy
            doc = self.nlp_model(text[:10000])  # Limit for performance

            # Look for specific patterns based on insight type
            if insight_type == InsightType.REQUIREMENT:
                insights.extend(self._extract_requirements_nlp(doc))
            elif insight_type == InsightType.RISK:
                insights.extend(self._extract_risks_nlp(doc))
            elif insight_type == InsightType.STAKEHOLDER:
                insights.extend(self._extract_stakeholders_nlp(doc))

        except Exception as e:
            logger.warning(f"NLP extraction failed for {insight_type.value}: {e}")

        return insights

    def _extract_requirements_nlp(self, doc) -> List[ExtractedInsight]:
        """Extract requirements using NLP"""
        insights = []

        for sent in doc.sents:
            # Look for modal verbs and requirement indicators
            modal_verbs = ['must', 'shall', 'should', 'will', 'can', 'may']
            has_modal = any(token.lemma_.lower() in modal_verbs for token in sent)

            # Look for requirement keywords
            req_keywords = ['requirement', 'required', 'needs', 'necessary']
            has_req_keyword = any(keyword in sent.text.lower() for keyword in req_keywords)

            if has_modal or has_req_keyword:
                confidence = ConfidenceLevel.HIGH if has_modal else ConfidenceLevel.MEDIUM

                insight = ExtractedInsight(
                    insight_type=InsightType.REQUIREMENT,
                    content=sent.text.strip(),
                    confidence=confidence,
                    context=sent.text.strip(),
                    metadata={
                        'extraction_method': 'nlp_modal_verbs',
                        'has_modal_verb': has_modal,
                        'has_requirement_keyword': has_req_keyword
                    }
                )

                insights.append(insight)

        return insights

    def _extract_risks_nlp(self, doc) -> List[ExtractedInsight]:
        """Extract risks using NLP"""
        insights = []

        for sent in doc.sents:
            # Look for negative sentiment and risk indicators
            risk_keywords = ['risk', 'issue', 'problem', 'concern', 'threat', 'challenge']
            has_risk_keyword = any(keyword in sent.text.lower() for keyword in risk_keywords)

            # Look for conditional or negative structures
            conditional_words = ['if', 'when', 'should']
            has_conditional = any(word in sent.text.lower() for word in conditional_words)

            if has_risk_keyword or has_conditional:
                confidence = ConfidenceLevel.HIGH if has_risk_keyword else ConfidenceLevel.MEDIUM

                insight = ExtractedInsight(
                    insight_type=InsightType.RISK,
                    content=sent.text.strip(),
                    confidence=confidence,
                    context=sent.text.strip(),
                    metadata={
                        'extraction_method': 'nlp_risk_indicators',
                        'has_risk_keyword': has_risk_keyword,
                        'has_conditional': has_conditional
                    }
                )

                insights.append(insight)

        return insights

    def _extract_stakeholders_nlp(self, doc) -> List[ExtractedInsight]:
        """Extract stakeholders using NLP"""
        insights = []

        for sent in doc.sents:
            # Look for named entities that might be stakeholders
            for ent in sent.ents:
                if ent.label_ in ['PERSON', 'ORG']:
                    # Check if entity is mentioned in stakeholder context
                    stakeholder_keywords = ['responsible', 'owner', 'team', 'stakeholder', 'manager']
                    has_stakeholder_context = any(
                        keyword in sent.text.lower() for keyword in stakeholder_keywords
                    )

                    if has_stakeholder_context:
                        insight = ExtractedInsight(
                            insight_type=InsightType.STAKEHOLDER,
                            content=f"{ent.text} ({ent.label_})",
                            confidence=ConfidenceLevel.HIGH,
                            context=sent.text.strip(),
                            metadata={
                                'extraction_method': 'nlp_named_entities',
                                'entity_type': ent.label_,
                                'stakeholder_context': True
                            }
                        )

                        insights.append(insight)

        return insights

    def _calculate_pattern_confidence(self, content: str, pattern: InsightPattern) -> ConfidenceLevel:
        """Calculate confidence level for pattern match"""
        # Base confidence on content length and keyword matches
        content_length = len(content)
        keyword_matches = sum(1 for keyword in pattern.keywords if keyword.lower() in content.lower())

        # Longer, more specific content gets higher confidence
        if content_length > 50 and keyword_matches > 1:
            return ConfidenceLevel.HIGH
        elif content_length > 20 and keyword_matches > 0:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level"""
        if score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _filter_insights(self, insights: List[ExtractedInsight]) -> List[ExtractedInsight]:
        """Filter and deduplicate insights"""
        filtered = []

        for insight in insights:
            # Skip very short or generic insights
            if len(insight.content.strip()) < 10:
                continue

            # Skip low confidence insights unless they're very specific
            if insight.confidence == ConfidenceLevel.LOW and len(insight.content) < 30:
                continue

            # Check for duplicates
            is_duplicate = False
            for existing in filtered:
                # Simple duplicate detection based on content similarity
                if self._calculate_similarity(insight.content, existing.content) > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(insight)

        return filtered

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _enhance_insight(self, insight: ExtractedInsight, full_text: str, processed: Dict):
        """Add context and metadata to insight"""
        # Add surrounding context
        if 'start' in insight.location:
            start = insight.location['start']
            context_start = max(0, start - self.context_window)
            context_end = min(len(full_text), start + len(insight.content) + self.context_window)
            insight.context = full_text[context_start:context_end].strip()

        # Add document metadata
        insight.metadata.update({
            'document_type': processed.get('file_type', 'unknown'),
            'extraction_timestamp': datetime.now().isoformat()
        })

    def _analyze_insight_relationships(self, insights: List[ExtractedInsight]) -> Dict[str, Any]:
        """Analyze relationships between insights"""
        relationships = {
            'dependencies': [],
            'related_insights': [],
            'clusters': []
        }

        # Find dependency relationships
        for i, insight1 in enumerate(insights):
            for j, insight2 in enumerate(insights[i+1:], i+1):
                # Check if insights are related
                similarity = self._calculate_similarity(insight1.content, insight2.content)

                if similarity > 0.3:  # Similarity threshold
                    relationships['related_insights'].append({
                        'insight1_id': i,
                        'insight2_id': j,
                        'similarity': similarity,
                        'types': [insight1.insight_type.value, insight2.insight_type.value]
                    })

                    # Check for specific dependency patterns
                    if (insight1.insight_type == InsightType.DEPENDENCY or
                        insight2.insight_type == InsightType.DEPENDENCY):
                        relationships['dependencies'].append({
                            'from_insight': i,
                            'to_insight': j,
                            'relationship_type': 'depends_on'
                        })

        return relationships

    def _calculate_insight_statistics(self, insights: List[ExtractedInsight], text: str) -> Dict[str, Any]:
        """Calculate statistics about extracted insights"""
        if not insights:
            return {'total_insights': 0}

        # Count by type
        type_counts = {}
        confidence_counts = {'high': 0, 'medium': 0, 'low': 0}

        for insight in insights:
            insight_type = insight.insight_type.value
            type_counts[insight_type] = type_counts.get(insight_type, 0) + 1
            confidence_counts[insight.confidence.value] += 1

        # Calculate coverage
        text_length = len(text)
        total_insight_length = sum(len(insight.content) for insight in insights)
        coverage_ratio = total_insight_length / text_length if text_length > 0 else 0

        return {
            'total_insights': len(insights),
            'type_distribution': type_counts,
            'confidence_distribution': confidence_counts,
            'coverage_ratio': coverage_ratio,
            'avg_insight_length': total_insight_length / len(insights),
            'insight_density': len(insights) / (text_length / 1000)  # Insights per 1000 characters
        }

    def _create_insights_summary(self, insights: List[ExtractedInsight]) -> Dict[str, Any]:
        """Create a summary of all insights"""
        if not insights:
            return {'overview': 'No insights extracted'}

        # Group insights by type for summary
        by_type = {}
        for insight in insights:
            insight_type = insight.insight_type.value
            if insight_type not in by_type:
                by_type[insight_type] = []
            by_type[insight_type].append(insight.content)

        # Create overview
        overview = f"Extracted {len(insights)} insights from document"

        # Find most significant insights (high confidence)
        high_confidence = [i for i in insights if i.confidence == ConfidenceLevel.HIGH]

        return {
            'overview': overview,
            'total_insights': len(insights),
            'high_confidence_insights': len(high_confidence),
            'insight_types_found': list(by_type.keys()),
            'top_insights': [i.content for i in high_confidence[:5]],  # Top 5 high confidence
            'summary_by_type': {k: len(v) for k, v in by_type.items()}
        }


# Global insight extractor instance
insight_extractor = KeyInsightExtractor()