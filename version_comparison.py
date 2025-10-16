"""
Automated Version Detection and Comparison Engine
Provides sophisticated analysis of document changes and version differences
"""

import difflib
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path

from .version_control import version_manager, DocumentVersion
from .nlp_processor import nlp_processor

@dataclass
class ChangeDetail:
    """Detailed information about a specific change"""
    change_type: str  # addition, deletion, modification, move
    element_type: str  # paragraph, sentence, word, table, image, etc.
    location: Dict[str, Any]  # start/end positions, page numbers, etc.
    old_content: str
    new_content: str
    significance: float  # 0.0 to 1.0
    description: str

@dataclass
class ComparisonResult:
    """Result of comparing two document versions"""
    version1_id: str
    version2_id: str
    overall_similarity: float  # 0.0 to 1.0
    change_summary: Dict[str, int]  # Count of each change type
    detailed_changes: List[ChangeDetail]
    semantic_changes: List[Dict[str, Any]]
    structural_changes: List[Dict[str, Any]]
    metadata_changes: List[Dict[str, Any]]

class DocumentComparator:
    """Advanced document comparison engine"""

    def __init__(self):
        self.change_patterns = self._initialize_change_patterns()

    def _initialize_change_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting different types of changes"""
        return {
            'critical_terms': [
                'shall', 'must', 'required', 'mandatory', 'critical', 'essential',
                'prohibited', 'forbidden', 'not permitted', 'strictly prohibited'
            ],
            'procedural_terms': [
                'process', 'procedure', 'step', 'workflow', 'guideline', 'policy',
                'standard', 'protocol', 'method', 'approach'
            ],
            'data_terms': [
                'data', 'information', 'record', 'document', 'file', 'report',
                'database', 'system', 'application', 'software'
            ],
            'temporal_terms': [
                'date', 'time', 'deadline', 'schedule', 'frequency', 'timing',
                'duration', 'period', 'cycle', 'timeline'
            ]
        }

    def compare_documents(self, version1_id: str, version2_id: str) -> ComparisonResult:
        """Compare two document versions comprehensively"""
        v1 = version_manager.db.get_version(version1_id)
        v2 = version_manager.db.get_version(version2_id)

        if not v1 or not v2:
            raise ValueError("Version not found")

        result = ComparisonResult(
            version1_id=version1_id,
            version2_id=version2_id,
            overall_similarity=0.0,
            change_summary={},
            detailed_changes=[],
            semantic_changes=[],
            structural_changes=[],
            metadata_changes=[]
        )

        # Compare content if available
        if v1.content_snapshot and v2.content_snapshot:
            content_comparison = self._compare_content(v1.content_snapshot, v2.content_snapshot)
            result.detailed_changes = content_comparison['detailed_changes']
            result.overall_similarity = content_comparison['similarity']

        # Compare metadata
        result.metadata_changes = self._compare_metadata(v1, v2)

        # Analyze semantic changes
        result.semantic_changes = self._analyze_semantic_changes(v1, v2)

        # Analyze structural changes
        result.structural_changes = self._analyze_structural_changes(v1, v2)

        # Generate change summary
        result.change_summary = self._generate_change_summary(result)

        return result

    def _compare_content(self, content1: str, content2: str) -> Dict[str, Any]:
        """Compare document content and return detailed changes"""
        lines1 = content1.splitlines()
        lines2 = content2.splitlines()

        # Basic line-by-line comparison
        differ = difflib.Differ()
        diff = list(differ.compare(lines1, lines2))

        # Analyze differences
        changes = []
        current_change = None

        for i, line in enumerate(diff):
            marker = line[0] if line else ' '

            if marker == '+':
                # Addition
                if current_change and current_change['change_type'] == 'addition':
                    current_change['new_content'] += '\n' + line[1:]
                else:
                    current_change = {
                        'change_type': 'addition',
                        'element_type': 'line',
                        'location': {'line_number': i},
                        'old_content': '',
                        'new_content': line[1:],
                        'significance': self._calculate_significance(line[1:]),
                        'description': 'Content added'
                    }
                    changes.append(current_change)

            elif marker == '-':
                # Deletion
                if current_change and current_change['change_type'] == 'deletion':
                    current_change['old_content'] += '\n' + line[1:]
                else:
                    current_change = {
                        'change_type': 'deletion',
                        'element_type': 'line',
                        'location': {'line_number': i},
                        'old_content': line[1:],
                        'new_content': '',
                        'significance': self._calculate_significance(line[1:]),
                        'description': 'Content removed'
                    }
                    changes.append(current_change)

            elif marker == '?':
                # Context line, attach to previous change if exists
                if current_change:
                    current_change['context'] = line[1:]

            else:
                # Unchanged line
                current_change = None

        # Calculate overall similarity
        similarity = difflib.SequenceMatcher(None, content1, content2).ratio()

        return {
            'similarity': similarity,
            'detailed_changes': changes,
            'total_lines_v1': len(lines1),
            'total_lines_v2': len(lines2),
            'changed_lines': len([c for c in changes if c['change_type'] != 'addition'])
        }

    def _compare_metadata(self, v1: DocumentVersion, v2: DocumentVersion) -> List[Dict[str, Any]]:
        """Compare metadata between versions"""
        changes = []

        # Compare file size
        if v1.file_size != v2.file_size:
            changes.append({
                'field': 'file_size',
                'old_value': v1.file_size,
                'new_value': v2.file_size,
                'change_type': 'size_change',
                'significance': self._calculate_size_change_significance(v1.file_size, v2.file_size)
            })

        # Compare file hash
        if v1.file_hash != v2.file_hash:
            changes.append({
                'field': 'content_hash',
                'old_value': v1.file_hash[:16] + '...',
                'new_value': v2.file_hash[:16] + '...',
                'change_type': 'content_change',
                'significance': 1.0
            })

        # Compare change type
        if v1.change_type != v2.change_type:
            changes.append({
                'field': 'change_type',
                'old_value': v1.change_type,
                'new_value': v2.change_type,
                'change_type': 'metadata_change',
                'significance': 0.3
            })

        # Compare lifecycle status
        if v1.lifecycle_status != v2.lifecycle_status:
            changes.append({
                'field': 'lifecycle_status',
                'old_value': v1.lifecycle_status,
                'new_value': v2.lifecycle_status,
                'change_type': 'status_change',
                'significance': 0.8
            })

        return changes

    def _analyze_semantic_changes(self, v1: DocumentVersion, v2: DocumentVersion) -> List[Dict[str, Any]]:
        """Analyze semantic changes between versions"""
        semantic_changes = []

        if not v1.content_snapshot or not v2.content_snapshot:
            return semantic_changes

        # Extract key terms and concepts
        terms1 = self._extract_key_terms(v1.content_snapshot)
        terms2 = self._extract_key_terms(v2.content_snapshot)

        # Find added and removed terms
        added_terms = terms2 - terms1
        removed_terms = terms1 - terms2

        if added_terms:
            semantic_changes.append({
                'change_type': 'semantic_addition',
                'description': 'New concepts or terms added',
                'added_terms': list(added_terms),
                'significance': min(0.8, len(added_terms) * 0.1)
            })

        if removed_terms:
            semantic_changes.append({
                'change_type': 'semantic_removal',
                'description': 'Concepts or terms removed',
                'removed_terms': list(removed_terms),
                'significance': min(0.8, len(removed_terms) * 0.1)
            })

        # Analyze sentiment changes if applicable
        sentiment1 = self._analyze_sentiment(v1.content_snapshot)
        sentiment2 = self._analyze_sentiment(v2.content_snapshot)

        if abs(sentiment1 - sentiment2) > 0.1:
            semantic_changes.append({
                'change_type': 'tone_change',
                'description': 'Document tone or sentiment changed',
                'old_sentiment': sentiment1,
                'new_sentiment': sentiment2,
                'significance': 0.4
            })

        return semantic_changes

    def _analyze_structural_changes(self, v1: DocumentVersion, v2: DocumentVersion) -> List[Dict[str, Any]]:
        """Analyze structural changes between versions"""
        structural_changes = []

        if not v1.content_snapshot or not v2.content_snapshot:
            return structural_changes

        # Analyze document structure
        structure1 = self._analyze_document_structure(v1.content_snapshot)
        structure2 = self._analyze_document_structure(v2.content_snapshot)

        # Compare section counts
        if structure1['section_count'] != structure2['section_count']:
            structural_changes.append({
                'change_type': 'section_count_change',
                'description': 'Number of sections changed',
                'old_count': structure1['section_count'],
                'new_count': structure2['section_count'],
                'significance': 0.3
            })

        # Compare paragraph counts
        if structure1['paragraph_count'] != structure2['paragraph_count']:
            structural_changes.append({
                'change_type': 'paragraph_count_change',
                'description': 'Number of paragraphs changed',
                'old_count': structure1['paragraph_count'],
                'new_count': structure2['paragraph_count'],
                'significance': 0.2
            })

        # Compare document length
        if structure1['word_count'] != structure2['word_count']:
            old_words = structure1['word_count']
            new_words = structure2['word_count']
            change_percent = abs(new_words - old_words) / max(old_words, new_words)

            structural_changes.append({
                'change_type': 'length_change',
                'description': 'Document length changed',
                'old_words': old_words,
                'new_words': new_words,
                'change_percent': change_percent,
                'significance': min(0.5, change_percent)
            })

        return structural_changes

    def _extract_key_terms(self, content: str) -> Set[str]:
        """Extract key terms from document content"""
        # Simple keyword extraction based on patterns
        key_terms = set()

        # Extract capitalized words (potential proper nouns)
        capitalized_words = set(re.findall(r'\b[A-Z][a-z]+\b', content))
        key_terms.update(capitalized_words)

        # Extract terms from patterns
        for category, patterns in self.change_patterns.items():
            for pattern in patterns:
                if pattern.lower() in content.lower():
                    key_terms.add(pattern.lower())

        return key_terms

    def _analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment of document content (simplified)"""
        # Simple sentiment analysis based on positive/negative words
        positive_words = {'good', 'excellent', 'improved', 'better', 'success', 'positive', 'benefit'}
        negative_words = {'bad', 'poor', 'worse', 'failure', 'negative', 'problem', 'issue', 'error'}

        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.5  # Neutral

        return positive_count / total_sentiment_words

    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure"""
        lines = content.splitlines()

        # Count sections (lines starting with numbers or bullets)
        section_count = len(re.findall(r'^\s*[\d\w]+\.', '\n'.join(lines)))

        # Count paragraphs (blocks of text separated by empty lines)
        paragraphs = [p for p in re.split(r'\n\s*\n', content) if p.strip()]
        paragraph_count = len(paragraphs)

        # Count words
        words = content.split()
        word_count = len(words)

        # Estimate reading time (words per minute)
        reading_time_minutes = word_count / 200  # Average reading speed

        return {
            'section_count': section_count,
            'paragraph_count': paragraph_count,
            'word_count': word_count,
            'character_count': len(content),
            'line_count': len(lines),
            'reading_time_minutes': reading_time_minutes
        }

    def _calculate_significance(self, content: str) -> float:
        """Calculate significance of a content change"""
        if not content.strip():
            return 0.0

        # Higher significance for critical terms
        significance = 0.2  # Base significance

        content_lower = content.lower()
        for category, patterns in self.change_patterns.items():
            for pattern in patterns:
                if pattern.lower() in content_lower:
                    significance += 0.2

        # Higher significance for longer content
        significance += min(0.3, len(content) / 1000)

        # Higher significance for structured content (bullet points, numbers)
        if re.search(r'^\s*[\d\-\*\+]', content):
            significance += 0.1

        return min(1.0, significance)

    def _calculate_size_change_significance(self, old_size: int, new_size: int) -> float:
        """Calculate significance of file size change"""
        if old_size == 0:
            return 1.0

        change_percent = abs(new_size - old_size) / old_size
        return min(1.0, change_percent)

    def _generate_change_summary(self, result: ComparisonResult) -> Dict[str, int]:
        """Generate summary of changes"""
        summary = {}

        # Count change types
        for change in result.detailed_changes:
            change_type = change['change_type']
            summary[change_type] = summary.get(change_type, 0) + 1

        # Add metadata changes
        for change in result.metadata_changes:
            change_type = change.get('change_type', 'metadata_change')
            summary[change_type] = summary.get(change_type, 0) + 1

        # Add semantic changes
        for change in result.semantic_changes:
            change_type = change.get('change_type', 'semantic_change')
            summary[change_type] = summary.get(change_type, 0) + 1

        # Add structural changes
        for change in result.structural_changes:
            change_type = change.get('change_type', 'structural_change')
            summary[change_type] = summary.get(change_type, 0) + 1

        return summary

class ContentAnalyzer:
    """Advanced content analysis for document comparison"""

    def __init__(self):
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> Set[str]:
        """Load common stop words"""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'could', 'should', 'may',
            'might', 'must', 'shall', 'can', 'this', 'these', 'those'
        }

    def extract_key_phrases(self, content: str, max_phrases: int = 10) -> List[Tuple[str, float]]:
        """Extract key phrases from document content"""
        # Simple phrase extraction based on noun phrases and important terms
        sentences = re.split(r'[.!?]+', content)

        phrases = []
        for sentence in sentences:
            # Extract capitalized phrases (likely proper nouns or titles)
            capitalized_phrases = re.findall(r'\b[A-Z][a-zA-Z\s]+\b', sentence)
            for phrase in capitalized_phrases:
                if len(phrase.split()) > 1 and len(phrase) > 3:
                    score = self._score_phrase(phrase, content)
                    phrases.append((phrase.strip(), score))

        # Sort by score and return top phrases
        phrases.sort(key=lambda x: x[1], reverse=True)
        return phrases[:max_phrases]

    def _score_phrase(self, phrase: str, content: str) -> float:
        """Score a phrase based on importance indicators"""
        score = 0.0

        # Frequency in document
        frequency = content.lower().count(phrase.lower())
        score += min(0.3, frequency * 0.05)

        # Length bonus (longer phrases are often more specific)
        score += min(0.2, len(phrase) / 50)

        # Position bonus (earlier phrases might be more important)
        position = content.lower().find(phrase.lower())
        if position > 0:
            score += 0.1 * (1 - position / len(content))

        return score

    def detect_content_type_changes(self, content1: str, content2: str) -> List[Dict[str, Any]]:
        """Detect changes in content types (e.g., text to table, etc.)"""
        changes = []

        # Analyze content structure
        structure1 = self._analyze_content_structure(content1)
        structure2 = self._analyze_content_structure(content2)

        # Compare structure types
        if structure1['dominant_type'] != structure2['dominant_type']:
            changes.append({
                'change_type': 'content_type_change',
                'old_type': structure1['dominant_type'],
                'new_type': structure2['dominant_type'],
                'significance': 0.6,
                'description': f'Document type changed from {structure1["dominant_type"]} to {structure2["dominant_type"]}'
            })

        return changes

    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure of document content"""
        # Count different content elements
        table_count = content.count('|') + content.count('\t')  # Simple table detection
        list_count = len(re.findall(r'^\s*[\d\-\*\+]', content, re.MULTILINE))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])

        # Determine dominant type
        if table_count > list_count and table_count > paragraph_count:
            dominant_type = 'tabular'
        elif list_count > paragraph_count:
            dominant_type = 'list'
        else:
            dominant_type = 'prose'

        return {
            'dominant_type': dominant_type,
            'table_elements': table_count,
            'list_elements': list_count,
            'paragraph_elements': paragraph_count
        }

class VersionDetectionEngine:
    """Engine for automatically detecting when new versions should be created"""

    def __init__(self):
        self.comparator = DocumentComparator()
        self.analyzer = ContentAnalyzer()
        self.significance_thresholds = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8
        }

    def should_create_version(self, file_path: str, current_version: DocumentVersion = None) -> Tuple[bool, str]:
        """
        Determine if a new version should be created for a document
        Returns: (should_create, reason)
        """

        try:
            # Read current file content
            if not os.path.exists(file_path):
                return False, "File does not exist"

            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()

            # If no current version, always create
            if not current_version:
                return True, "Initial version"

            # If no content snapshot, create version
            if not current_version.content_snapshot:
                return True, "No previous content snapshot"

            # Compare content
            comparison = self.comparator._compare_content(
                current_version.content_snapshot,
                current_content
            )

            # Check if changes exceed threshold
            if comparison['similarity'] < 0.9:  # Less than 90% similar
                max_significance = max(
                    (change.get('significance', 0) for change in comparison['detailed_changes']),
                    default=0
                )

                if max_significance >= self.significance_thresholds['medium']:
                    return True, f"Significant content changes detected (similarity: {comparison['similarity']".2f"})"

            # Check file metadata changes
            current_stat = os.stat(file_path)
            if (current_stat.st_size != current_version.file_size or
                current_stat.st_mtime != current_version.created_at.timestamp()):
                return True, "File metadata changed"

            return False, "No significant changes detected"

        except Exception as e:
            return True, f"Error during analysis: {str(e)}"

    def analyze_change_impact(self, doc_id: str, new_content: str = None) -> Dict[str, Any]:
        """Analyze the potential impact of document changes"""
        # Get current version
        versions = version_manager.db.get_document_versions(doc_id)
        current_version = versions[0] if versions else None

        if not current_version or not new_content:
            return {'impact_level': 'unknown', 'reason': 'No baseline for comparison'}

        # Compare with current version
        comparison = self.comparator.compare_documents(current_version.version_id, current_version.version_id)

        # Assess impact based on changes
        high_impact_keywords = ['critical', 'security', 'compliance', 'legal', 'safety']
        impact_score = 0.0

        # Check for high-impact changes
        for change in comparison.detailed_changes:
            content = (change.get('old_content', '') + ' ' + change.get('new_content', '')).lower()
            if any(keyword in content for keyword in high_impact_keywords):
                impact_score += 0.3

        # Check semantic changes
        for change in comparison.semantic_changes:
            if 'critical' in change.get('description', '').lower():
                impact_score += 0.4

        # Determine impact level
        if impact_score >= 0.7:
            impact_level = 'high'
        elif impact_score >= 0.4:
            impact_level = 'medium'
        else:
            impact_level = 'low'

        return {
            'impact_level': impact_level,
            'impact_score': impact_score,
            'change_summary': comparison.change_summary,
            'recommendations': self._generate_impact_recommendations(impact_level, comparison)
        }

    def _generate_impact_recommendations(self, impact_level: str, comparison: ComparisonResult) -> List[str]:
        """Generate recommendations based on impact analysis"""
        recommendations = []

        if impact_level == 'high':
            recommendations.append("Review changes with stakeholders before approval")
            recommendations.append("Consider impact on dependent processes")
            recommendations.append("Update related documentation")

        if comparison.change_summary.get('content_type_change', 0) > 0:
            recommendations.append("Verify that content type changes don't break existing integrations")

        if comparison.change_summary.get('semantic_removal', 0) > 0:
            recommendations.append("Ensure removed concepts are properly archived or communicated")

        return recommendations

# Global instances
document_comparator = DocumentComparator()
content_analyzer = ContentAnalyzer()
version_detection_engine = VersionDetectionEngine()