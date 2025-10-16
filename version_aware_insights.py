"""
Version-Aware Insights System
Generates insights for specific document versions and tracks insight evolution
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

# Custom imports
from .version_control import version_manager, DocumentVersion
from .insight_extractor import insight_extractor, ExtractedInsight
from .summarization_engine import summarization_engine, SummarizationConfig, SummarizationMethod, DetailLevel
from .content_categorizer import content_categorizer
from .export_manager import export_manager

logger = logging.getLogger(__name__)


@dataclass
class VersionInsight:
    """Represents insights for a specific document version"""
    version_id: str
    document_id: str
    version_number: str

    # Generated insights
    insights: List[ExtractedInsight] = field(default_factory=list)
    summary: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    insight_count: int = 0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'version_id': self.version_id,
            'document_id': self.document_id,
            'version_number': self.version_number,
            'insights': [insight.to_dict() for insight in self.insights],
            'summary': self.summary,
            'categories': self.categories,
            'tags': self.tags,
            'generated_at': self.generated_at.isoformat(),
            'processing_time': self.processing_time,
            'insight_count': self.insight_count,
            'confidence_distribution': self.confidence_distribution
        }


@dataclass
class InsightEvolution:
    """Tracks how insights evolve across document versions"""
    document_id: str
    evolution_timeline: List[Dict[str, Any]] = field(default_factory=list)
    insight_changes: Dict[str, List[str]] = field(default_factory=dict)
    stability_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'document_id': self.document_id,
            'evolution_timeline': self.evolution_timeline,
            'insight_changes': self.insight_changes,
            'stability_metrics': self.stability_metrics
        }


class VersionAwareInsightEngine:
    """Engine for generating version-aware insights"""

    def __init__(self):
        self.insight_cache = {}
        self.evolution_tracker = {}

    def generate_version_insights(self, version_id: str,
                                force_refresh: bool = False) -> Optional[VersionInsight]:
        """
        Generate insights for a specific document version

        Args:
            version_id: ID of the document version
            force_refresh: Force regeneration of insights

        Returns:
            VersionInsight object with insights for the version
        """
        # Check cache first
        cache_key = f"insights_{version_id}"
        if not force_refresh and cache_key in self.insight_cache:
            return self.insight_cache[cache_key]

        try:
            # Get version information
            version = version_manager.db.get_version(version_id)
            if not version:
                logger.error(f"Version not found: {version_id}")
                return None

            # Check if we can access the file
            if not os.path.exists(version.file_path):
                logger.warning(f"Version file not found: {version.file_path}")
                # Try to use content snapshot if available
                if version.content_snapshot:
                    return self._generate_insights_from_content(
                        version, version.content_snapshot
                    )
                else:
                    logger.error("No content available for version")
                    return None

            # Generate insights from file
            return self._generate_insights_from_file(version)

        except Exception as e:
            logger.error(f"Error generating version insights: {e}")
            return None

    def _generate_insights_from_file(self, version: DocumentVersion) -> Optional[VersionInsight]:
        """Generate insights from version file"""
        start_time = datetime.now()

        try:
            # Extract insights
            insights_result = insight_extractor.extract_insights(version.file_path)

            if not insights_result['success']:
                logger.error(f"Failed to extract insights: {insights_result['error']}")
                return None

            # Generate summary
            summary_config = SummarizationConfig(
                method=SummarizationMethod.HYBRID,
                detail_level=DetailLevel.CONCISE
            )
            summary_result = summarization_engine.summarize_document(
                version.file_path, summary_config
            )

            # Categorize content
            categorization_result = content_categorizer.categorize_document(version.file_path)

            # Calculate confidence distribution
            confidence_dist = {}
            for insight in insights_result['insights']:
                conf = insight.confidence.value
                confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

            # Create version insight object
            version_insight = VersionInsight(
                version_id=version.version_id,
                document_id=version.doc_id,
                version_number=version.version_number,
                insights=insights_result['insights'],
                summary=summary_result.summary if summary_result.success else None,
                categories=[cat.get('category_name', '') for cat in categorization_result.get('categories', [])],
                tags=[tag.get('name', '') for tag in categorization_result.get('tags', [])],
                processing_time=(datetime.now() - start_time).total_seconds(),
                insight_count=len(insights_result['insights']),
                confidence_distribution=confidence_dist
            )

            # Cache results
            cache_key = f"insights_{version.version_id}"
            self.insight_cache[cache_key] = version_insight

            return version_insight

        except Exception as e:
            logger.error(f"Error in file-based insight generation: {e}")
            return None

    def _generate_insights_from_content(self, version: DocumentVersion,
                                      content: str) -> Optional[VersionInsight]:
        """Generate insights from content snapshot"""
        start_time = datetime.now()

        try:
            # For content-based generation, we need to adapt our approach
            # since some processors expect file paths

            # Generate summary from content
            summary_config = SummarizationConfig(
                method=SummarizationMethod.HYBRID,
                detail_level=DetailLevel.CONCISE
            )
            summary_result = summarization_engine.summarize_text(content, summary_config)

            # For insights, we'll need to create a temporary file or adapt the extractor
            # For now, create basic insights from content analysis
            insights = self._extract_insights_from_content(content)

            # Basic categorization
            categories = self._categorize_content_basic(content)
            tags = self._extract_tags_from_content(content)

            # Calculate confidence distribution
            confidence_dist = {}
            for insight in insights:
                conf = insight.confidence.value
                confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

            version_insight = VersionInsight(
                version_id=version.version_id,
                document_id=version.doc_id,
                version_number=version.version_number,
                insights=insights,
                summary=summary_result.summary if summary_result.success else None,
                categories=categories,
                tags=tags,
                processing_time=(datetime.now() - start_time).total_seconds(),
                insight_count=len(insights),
                confidence_distribution=confidence_dist
            )

            # Cache results
            cache_key = f"insights_{version.version_id}"
            self.insight_cache[cache_key] = version_insight

            return version_insight

        except Exception as e:
            logger.error(f"Error in content-based insight generation: {e}")
            return None

    def _extract_insights_from_content(self, content: str) -> List[ExtractedInsight]:
        """Extract basic insights from content text"""
        insights = []

        # Simple keyword-based insight extraction
        content_lower = content.lower()

        # Define insight patterns
        insight_patterns = {
            'process': ['process', 'procedure', 'steps', 'workflow'],
            'requirement': ['must', 'shall', 'should', 'required'],
            'risk': ['risk', 'issue', 'problem', 'concern'],
            'dependency': ['depends', 'requires', 'prerequisite']
        }

        for insight_type, keywords in insight_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    from .insight_extractor import InsightType, ConfidenceLevel

                    insight = ExtractedInsight(
                        insight_type=InsightType(insight_type),
                        content=f"Found {insight_type} indicator: {keyword}",
                        confidence=ConfidenceLevel.MEDIUM,
                        context=f"Content contains {insight_type}-related keywords",
                        metadata={
                            'extraction_method': 'content_keyword_analysis',
                            'matched_keyword': keyword
                        }
                    )
                    insights.append(insight)

        return insights

    def _categorize_content_basic(self, content: str) -> List[str]:
        """Basic content categorization"""
        categories = []
        content_lower = content.lower()

        # Simple categorization based on keywords
        if any(word in content_lower for word in ['invoice', 'payment', 'billing']):
            categories.append('Financial')
        if any(word in content_lower for word in ['procedure', 'process', 'steps']):
            categories.append('Procedural')
        if any(word in content_lower for word in ['oracle', 'sap', 'system']):
            categories.append('Technical')

        return categories

    def _extract_tags_from_content(self, content: str) -> List[str]:
        """Extract basic tags from content"""
        tags = []

        # Extract meaningful words as tags
        import re
        words = re.findall(r'\b[A-Z][a-z]+\b', content)  # Capitalized words

        # Filter and limit tags
        meaningful_tags = [word for word in words if len(word) > 3][:10]
        tags.extend(meaningful_tags)

        return tags

    def analyze_insight_evolution(self, document_id: str) -> Optional[InsightEvolution]:
        """
        Analyze how insights evolve across document versions

        Args:
            document_id: Document ID to analyze

        Returns:
            InsightEvolution object with evolution analysis
        """
        try:
            # Get all versions for the document
            versions = version_manager.db.get_document_versions(document_id)

            if len(versions) < 2:
                logger.warning(f"Need at least 2 versions for evolution analysis. Found: {len(versions)}")
                return None

            # Get insights for each version
            version_insights = []
            for version in versions:
                insight = self.generate_version_insights(version.version_id)
                if insight:
                    version_insights.append(insight)

            if len(version_insights) < 2:
                logger.warning("Insufficient version insights for evolution analysis")
                return None

            # Analyze evolution
            evolution = InsightEvolution(document_id=document_id)

            # Build timeline
            for insight in version_insights:
                evolution.evolution_timeline.append({
                    'version_id': insight.version_id,
                    'version_number': insight.version_number,
                    'insight_count': insight.insight_count,
                    'confidence_distribution': insight.confidence_distribution,
                    'categories': insight.categories,
                    'generated_at': insight.generated_at.isoformat()
                })

            # Analyze changes
            evolution.insight_changes = self._analyze_insight_changes(version_insights)

            # Calculate stability metrics
            evolution.stability_metrics = self._calculate_stability_metrics(version_insights)

            # Cache evolution data
            self.evolution_tracker[document_id] = evolution

            return evolution

        except Exception as e:
            logger.error(f"Error analyzing insight evolution: {e}")
            return None

    def _analyze_insight_changes(self, version_insights: List[VersionInsight]) -> Dict[str, List[str]]:
        """Analyze changes in insights between versions"""
        changes = {
            'new_insights': [],
            'removed_insights': [],
            'modified_insights': [],
            'stable_insights': []
        }

        if len(version_insights) < 2:
            return changes

        # Sort by version number
        sorted_insights = sorted(version_insights, key=lambda x: x.version_number)

        # Simple change analysis based on insight counts and types
        for i in range(1, len(sorted_insights)):
            current = sorted_insights[i]
            previous = sorted_insights[i-1]

            current_count = current.insight_count
            previous_count = previous.insight_count

            if current_count > previous_count:
                changes['new_insights'].append(
                    f"Version {current.version_number}: {current_count - previous_count} new insights"
                )
            elif current_count < previous_count:
                changes['removed_insights'].append(
                    f"Version {current.version_number}: {previous_count - current_count} insights removed"
                )

        return changes

    def _calculate_stability_metrics(self, version_insights: List[VersionInsight]) -> Dict[str, float]:
        """Calculate stability metrics for insights across versions"""
        metrics = {}

        if len(version_insights) < 2:
            return metrics

        # Calculate insight count stability
        insight_counts = [insight.insight_count for insight in version_insights]
        if len(insight_counts) > 1:
            # Coefficient of variation for insight counts
            mean_count = sum(insight_counts) / len(insight_counts)
            if mean_count > 0:
                std_dev = (sum((x - mean_count) ** 2 for x in insight_counts) / len(insight_counts)) ** 0.5
                metrics['insight_stability'] = 1.0 / (1.0 + (std_dev / mean_count))
            else:
                metrics['insight_stability'] = 0.0

        # Calculate confidence stability
        confidence_scores = []
        for insight in version_insights:
            total_confidence = sum(
                insight.confidence_distribution.get(conf, 0) * {'low': 1, 'medium': 2, 'high': 3}.get(conf, 1)
                for conf in insight.confidence_distribution
            )
            total_insights = insight.insight_count
            if total_insights > 0:
                avg_confidence = total_confidence / total_insights
                confidence_scores.append(avg_confidence)

        if confidence_scores:
            metrics['confidence_stability'] = 1.0 / (1.0 + (sum((s - sum(confidence_scores)/len(confidence_scores))**2
                                                              for s in confidence_scores) / len(confidence_scores)) ** 0.5)

        return metrics

    def compare_version_insights(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """
        Compare insights between two document versions

        Args:
            version1_id: First version ID
            version2_id: Second version ID

        Returns:
            Comparison results
        """
        try:
            # Get insights for both versions
            insight1 = self.generate_version_insights(version1_id)
            insight2 = self.generate_version_insights(version2_id)

            if not insight1 or not insight2:
                return {'error': 'Could not generate insights for one or both versions'}

            # Get version information
            version1 = version_manager.db.get_version(version1_id)
            version2 = version_manager.db.get_version(version2_id)

            if not version1 or not version2:
                return {'error': 'Could not retrieve version information'}

            # Compare insights
            comparison = {
                'version1': {
                    'version_id': version1.version_id,
                    'version_number': version1.version_number,
                    'insight_count': insight1.insight_count,
                    'confidence_distribution': insight1.confidence_distribution,
                    'categories': insight1.categories
                },
                'version2': {
                    'version_id': version2.version_id,
                    'version_number': version2.version_number,
                    'insight_count': insight2.insight_count,
                    'confidence_distribution': insight2.confidence_distribution,
                    'categories': insight2.categories
                },
                'differences': self._calculate_insight_differences(insight1, insight2),
                'evolution_indicators': self._calculate_evolution_indicators(insight1, insight2)
            }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing version insights: {e}")
            return {'error': str(e)}

    def _calculate_insight_differences(self, insight1: VersionInsight,
                                     insight2: VersionInsight) -> Dict[str, Any]:
        """Calculate differences between two version insights"""
        differences = {
            'insight_count_diff': insight2.insight_count - insight1.insight_count,
            'category_changes': [],
            'confidence_changes': {}
        }

        # Category changes
        categories1 = set(insight1.categories)
        categories2 = set(insight2.categories)

        added_categories = categories2 - categories1
        removed_categories = categories1 - categories2

        if added_categories:
            differences['category_changes'].append(f"Added categories: {list(added_categories)}")
        if removed_categories:
            differences['category_changes'].append(f"Removed categories: {list(removed_categories)}")

        # Confidence distribution changes
        for conf_level in ['high', 'medium', 'low']:
            count1 = insight1.confidence_distribution.get(conf_level, 0)
            count2 = insight2.confidence_distribution.get(conf_level, 0)
            diff = count2 - count1
            if diff != 0:
                differences['confidence_changes'][conf_level] = diff

        return differences

    def _calculate_evolution_indicators(self, insight1: VersionInsight,
                                      insight2: VersionInsight) -> Dict[str, Any]:
        """Calculate evolution indicators between versions"""
        indicators = {
            'insight_growth_rate': 0.0,
            'confidence_improvement': 0.0,
            'stability_score': 0.0
        }

        # Insight growth rate
        if insight1.insight_count > 0:
            indicators['insight_growth_rate'] = (
                (insight2.insight_count - insight1.insight_count) / insight1.insight_count
            )

        # Confidence improvement
        def calculate_avg_confidence(insight):
            total_weighted = sum(
                insight.confidence_distribution.get(conf, 0) * {'low': 1, 'medium': 2, 'high': 3}.get(conf, 1)
                for conf in insight.confidence_distribution
            )
            total_insights = insight.insight_count
            return total_weighted / total_insights if total_insights > 0 else 0

        avg_conf1 = calculate_avg_confidence(insight1)
        avg_conf2 = calculate_avg_confidence(insight2)

        if avg_conf1 > 0:
            indicators['confidence_improvement'] = (avg_conf2 - avg_conf1) / avg_conf1

        # Stability score (based on similarity of insight types and categories)
        categories1 = set(insight1.categories)
        categories2 = set(insight2.categories)

        if categories1 or categories2:
            intersection = len(categories1.intersection(categories2))
            union = len(categories1.union(categories2))
            indicators['stability_score'] = intersection / union if union > 0 else 0.0

        return indicators

    def get_document_insight_history(self, document_id: str) -> Dict[str, Any]:
        """
        Get complete insight history for a document

        Args:
            document_id: Document ID

        Returns:
            Complete insight history across all versions
        """
        try:
            # Get all versions
            versions = version_manager.db.get_document_versions(document_id)

            if not versions:
                return {'error': 'No versions found for document'}

            # Get insights for each version
            version_insights = []
            for version in versions:
                insight = self.generate_version_insights(version.version_id)
                if insight:
                    version_insights.append(insight)

            # Generate evolution analysis
            evolution = self.analyze_insight_evolution(document_id)

            # Compile history
            history = {
                'document_id': document_id,
                'total_versions': len(versions),
                'versions_with_insights': len(version_insights),
                'version_insights': [vi.to_dict() for vi in version_insights],
                'evolution_analysis': evolution.to_dict() if evolution else None,
                'summary': {
                    'total_insights_across_versions': sum(vi.insight_count for vi in version_insights),
                    'average_insights_per_version': sum(vi.insight_count for vi in version_insights) / len(version_insights) if version_insights else 0,
                    'most_insightful_version': max(version_insights, key=lambda x: x.insight_count).version_number if version_insights else None
                }
            }

            return history

        except Exception as e:
            logger.error(f"Error getting insight history: {e}")
            return {'error': str(e)}

    def export_version_insights(self, version_id: str, export_format: str,
                              output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export insights for a specific version

        Args:
            version_id: Version ID
            export_format: Export format
            output_path: Custom output path

        Returns:
            Export results
        """
        try:
            # Generate insights if not cached
            insight = self.generate_version_insights(version_id)

            if not insight:
                return {'success': False, 'error': 'Could not generate insights for version'}

            # Prepare export data
            export_data = {
                'type': 'version_insights',
                'data': insight,
                'filename': f"version_insights_{version_id}.{export_format}"
            }

            # Use export manager
            return export_manager.batch_export([export_data], export_format, output_path)

        except Exception as e:
            return {'success': False, 'error': str(e)}


class VersionAwareInsightAPI:
    """API for version-aware insight operations"""

    def __init__(self):
        self.insight_engine = VersionAwareInsightEngine()

    def get_version_insights(self, version_id: str) -> Dict[str, Any]:
        """Get insights for a specific version"""
        insight = self.insight_engine.generate_version_insights(version_id)

        if insight:
            return {
                'success': True,
                'version_insights': insight.to_dict()
            }
        else:
            return {
                'success': False,
                'error': 'Could not generate insights for version'
            }

    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Compare insights between two versions"""
        return self.insight_engine.compare_version_insights(version1_id, version2_id)

    def get_insight_evolution(self, document_id: str) -> Dict[str, Any]:
        """Get insight evolution for a document"""
        evolution = self.insight_engine.analyze_insight_evolution(document_id)

        if evolution:
            return {
                'success': True,
                'evolution': evolution.to_dict()
            }
        else:
            return {
                'success': False,
                'error': 'Could not analyze insight evolution'
            }

    def get_insight_history(self, document_id: str) -> Dict[str, Any]:
        """Get complete insight history for a document"""
        return self.insight_engine.get_document_insight_history(document_id)

    def export_version_insights(self, version_id: str, export_format: str = 'json') -> Dict[str, Any]:
        """Export version insights"""
        return self.insight_engine.export_version_insights(version_id, export_format)


# Global version-aware insight instances
version_insight_engine = VersionAwareInsightEngine()
version_insight_api = VersionAwareInsightAPI()