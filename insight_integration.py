"""
Integration Module for Advanced Document Summarization and Insight Generation System
Connects new capabilities with existing search engine and vector store
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Custom imports
from .summarization_engine import summarization_engine, SummarizationConfig, SummarizationMethod, DetailLevel
from .insight_extractor import insight_extractor
from .content_categorizer import content_categorizer
from .relationship_analyzer import relationship_analyzer
from .export_manager import export_manager
from .insight_dashboard import dashboard_generator

# Existing search engine imports
from .search_engine import AdvancedSearchEngine
from .models import DocumentMetadata
from .vector_search import SemanticSearchEngine

logger = logging.getLogger(__name__)


class InsightEnhancedDocumentProcessor:
    """Enhanced document processor with insight capabilities"""

    def __init__(self, search_engine: AdvancedSearchEngine):
        self.search_engine = search_engine
        self.insight_cache = {}

    def process_document_with_insights(self, file_path: str, doc_id: str = None) -> Dict[str, Any]:
        """
        Process document with full insight extraction and analysis

        Args:
            file_path: Path to document
            doc_id: Document ID (optional)

        Returns:
            Complete processing results with insights
        """
        try:
            # Generate document ID if not provided
            if doc_id is None:
                doc_id = self._generate_document_id(file_path)

            # Step 1: Basic search engine processing
            success = self.search_engine.index_document(file_path, doc_id)

            if not success:
                return {
                    'success': False,
                    'error': 'Failed to index document in search engine'
                }

            # Step 2: Extract insights
            insights_result = insight_extractor.extract_insights(file_path)

            # Step 3: Categorize and tag
            categorization_result = content_categorizer.categorize_document(file_path)

            # Step 4: Generate summary
            summary_config = SummarizationConfig(
                method=SummarizationMethod.HYBRID,
                detail_level=DetailLevel.CONCISE
            )
            summary_result = summarization_engine.summarize_document(file_path, summary_config)

            # Step 5: Combine all results
            combined_result = {
                'success': True,
                'document_id': doc_id,
                'file_path': file_path,
                'processing_timestamp': datetime.now().isoformat(),
                'search_engine_indexed': True,
                'insights': insights_result,
                'categorization': categorization_result,
                'summary': summary_result.to_dict() if summary_result.success else None,
                'metadata': {
                    'file_size': os.path.getsize(file_path),
                    'file_extension': Path(file_path).suffix,
                    'processing_time': 0  # Would calculate actual time
                }
            }

            # Cache results
            self.insight_cache[doc_id] = combined_result

            return combined_result

        except Exception as e:
            logger.error(f"Enhanced document processing error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_document_id(self, file_path: str) -> str:
        """Generate consistent document ID"""
        import hashlib
        path_str = os.path.abspath(file_path)
        return hashlib.md5(path_str.encode()).hexdigest()[:12]


class InsightSearchEngine:
    """Search engine enhanced with insight capabilities"""

    def __init__(self, base_search_engine: AdvancedSearchEngine):
        self.base_engine = base_search_engine
        self.insight_processor = InsightEnhancedDocumentProcessor(base_search_engine)

    def search_with_insights(self, query: str, include_insights: bool = True,
                           include_summaries: bool = False, **search_kwargs) -> Dict[str, Any]:
        """
        Enhanced search with insight integration

        Args:
            query: Search query
            include_insights: Whether to include insights in results
            include_summaries: Whether to include summaries in results
            **search_kwargs: Additional search parameters

        Returns:
            Enhanced search results
        """
        # Perform base search
        base_results = self.base_engine.search(query, **search_kwargs)

        if not include_insights and not include_summaries:
            return base_results.to_dict() if hasattr(base_results, 'to_dict') else base_results

        # Enhance results with insights
        enhanced_results = []

        for result in base_results.results:
            enhanced_result = result.to_dict() if hasattr(result, 'to_dict') else result.copy()

            doc_id = result.doc_id

            # Add insights if requested
            if include_insights:
                insights = self._get_document_insights(doc_id)
                if insights:
                    enhanced_result['insights'] = insights

            # Add summary if requested
            if include_summaries:
                summary = self._get_document_summary(doc_id)
                if summary:
                    enhanced_result['summary'] = summary

            enhanced_results.append(enhanced_result)

        # Create enhanced response
        enhanced_response = base_results.to_dict() if hasattr(base_results, 'to_dict') else base_results.copy()
        enhanced_response['results'] = enhanced_results
        enhanced_response['enhanced_with_insights'] = include_insights
        enhanced_response['enhanced_with_summaries'] = include_summaries

        return enhanced_response

    def _get_document_insights(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached insights for document"""
        # Check cache first
        if doc_id in self.insight_processor.insight_cache:
            cached = self.insight_processor.insight_cache[doc_id]
            return cached.get('insights')

        # Extract insights if not cached
        # This would need the file path - for now return None
        return None

    def _get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached summary for document"""
        if doc_id in self.insight_processor.insight_cache:
            cached = self.insight_processor.insight_cache[doc_id]
            return cached.get('summary')

        return None

    def get_document_insights_dashboard(self, doc_id: str) -> Dict[str, Any]:
        """Get comprehensive insights dashboard for a document"""
        # Get cached data
        if doc_id in self.insight_processor.insight_cache:
            cached = self.insight_processor.insight_cache[doc_id]

            # Generate dashboard for single document
            dashboard_data = {
                'document_id': doc_id,
                'insights': cached.get('insights', {}),
                'categorization': cached.get('categorization', {}),
                'summary': cached.get('summary', {}),
                'metadata': cached.get('metadata', {})
            }

            return dashboard_generator.generate_insight_dashboard(dashboard_data)

        return {'error': 'Document not found in insight cache'}

    def analyze_document_collection(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze entire document collection with insights

        Args:
            document_paths: List of document paths to analyze

        Returns:
            Collection analysis results
        """
        collection_results = []

        for file_path in document_paths:
            try:
                # Process each document with insights
                result = self.insight_processor.process_document_with_insights(file_path)
                collection_results.append(result)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                collection_results.append({
                    'success': False,
                    'file_path': file_path,
                    'error': str(e)
                })

        # Generate collection-wide analysis
        collection_insights = self._generate_collection_insights(collection_results)

        # Generate relationship analysis
        relationship_analysis = relationship_analyzer.analyze_document_collection(document_paths)

        return {
            'success': True,
            'collection_analysis': collection_insights,
            'relationship_analysis': relationship_analysis,
            'individual_results': collection_results,
            'summary': {
                'total_documents': len(document_paths),
                'successfully_processed': len([r for r in collection_results if r.get('success')]),
                'total_insights': sum(len(r.get('insights', {}).get('insights', []))
                                   for r in collection_results if r.get('success')),
                'total_relationships': len(relationship_analysis.get('relationships', [])),
                'total_gaps': len(relationship_analysis.get('knowledge_gaps', []))
            }
        }

    def _generate_collection_insights(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate insights about the entire collection"""
        successful_results = [r for r in results if r.get('success')]

        if not successful_results:
            return {'error': 'No successfully processed documents'}

        # Aggregate insights by type
        all_insights = []
        for result in successful_results:
            insights = result.get('insights', {}).get('insights', [])
            all_insights.extend(insights)

        # Aggregate categories
        all_categories = []
        for result in successful_results:
            categories = result.get('categorization', {}).get('categories', [])
            all_categories.extend(categories)

        # Generate collection-level insights
        from collections import Counter

        insight_types = Counter(insight.insight_type.value for insight in all_insights)
        category_names = Counter(cat.get('category_name', '') for cat in all_categories)

        return {
            'total_insights': len(all_insights),
            'insight_type_distribution': dict(insight_types),
            'category_distribution': dict(category_names),
            'documents_with_insights': len(successful_results),
            'average_insights_per_document': len(all_insights) / len(successful_results)
        }


class InsightAPI:
    """API interface for insight operations"""

    def __init__(self, search_engine: AdvancedSearchEngine):
        self.search_engine = InsightSearchEngine(search_engine)

    def summarize_document(self, file_path: str, method: str = 'hybrid',
                          detail_level: str = 'concise', **kwargs) -> Dict[str, Any]:
        """
        API endpoint for document summarization

        Args:
            file_path: Path to document
            method: Summarization method
            detail_level: Level of detail
            **kwargs: Additional configuration

        Returns:
            Summarization results
        """
        try:
            # Convert string parameters to enums
            method_enum = SummarizationMethod(method.lower())
            detail_enum = DetailLevel(detail_level.lower())

            # Create configuration
            config = SummarizationConfig(
                method=method_enum,
                detail_level=detail_enum,
                **kwargs
            )

            # Generate summary
            result = summarization_engine.summarize_document(file_path, config)

            return {
                'success': result.success,
                'summary': result.summary if result.success else None,
                'metadata': result.to_dict() if result.success else None,
                'error': result.error_message if not result.success else None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def extract_insights(self, file_path: str) -> Dict[str, Any]:
        """
        API endpoint for insight extraction

        Args:
            file_path: Path to document

        Returns:
            Extracted insights
        """
        try:
            result = insight_extractor.extract_insights(file_path)

            return {
                'success': result['success'],
                'insights': result['insights'] if result['success'] else [],
                'summary': result['summary'] if result['success'] else None,
                'statistics': result['statistics'] if result['success'] else None,
                'error': result['error'] if not result['success'] else None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def categorize_document(self, file_path: str) -> Dict[str, Any]:
        """
        API endpoint for document categorization

        Args:
            file_path: Path to document

        Returns:
            Categorization results
        """
        try:
            result = content_categorizer.categorize_document(file_path)

            return {
                'success': result['success'],
                'categories': result['categories'] if result['success'] else [],
                'tags': result['tags'] if result['success'] else [],
                'confidence_scores': result['confidence_scores'] if result['success'] else None,
                'error': result['error'] if not result['success'] else None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def analyze_relationships(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        API endpoint for relationship analysis

        Args:
            document_paths: List of document paths

        Returns:
            Relationship analysis results
        """
        try:
            result = relationship_analyzer.analyze_document_collection(document_paths)

            return {
                'success': result['success'],
                'relationships': result['relationships'] if result['success'] else [],
                'knowledge_gaps': result['knowledge_gaps'] if result['success'] else [],
                'summary': result['summary'] if result['success'] else None,
                'error': result['error'] if not result['success'] else None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def export_results(self, data: Any, export_format: str, data_type: str,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        API endpoint for exporting results

        Args:
            data: Data to export
            export_format: Export format (pdf, json, csv, etc.)
            data_type: Type of data (summary, insights, dashboard)
            output_path: Custom output path (optional)

        Returns:
            Export results
        """
        try:
            if data_type == 'summary' and hasattr(data, 'to_dict'):
                # Handle SummarizationResult
                result = export_manager.export_summary(data, export_format, output_path)
            elif data_type == 'insights':
                # Handle insights list
                result = export_manager.export_insights(data, export_format, output_path)
            elif data_type == 'dashboard':
                # Handle dashboard data
                result = export_manager.export_dashboard(data, export_format, output_path)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported data type: {data_type}'
                }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_dashboard(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        API endpoint for generating dashboard

        Args:
            document_paths: List of document paths

        Returns:
            Dashboard data
        """
        try:
            # Analyze collection
            analysis = self.search_engine.analyze_document_collection(document_paths)

            if not analysis['success']:
                return analysis

            # Generate dashboard
            dashboard = dashboard_generator.generate_insight_dashboard(analysis)

            return {
                'success': True,
                'dashboard': dashboard,
                'document_count': len(document_paths)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class InsightIntegrationManager:
    """Main integration manager for the insight system"""

    def __init__(self, search_engine: AdvancedSearchEngine = None):
        # Initialize base search engine if not provided
        if search_engine is None:
            search_engine = AdvancedSearchEngine()

        self.search_engine = search_engine
        self.insight_api = InsightAPI(search_engine)

        # Integration status
        self.integration_status = {
            'search_engine_connected': True,
            'vector_store_connected': True,
            'insight_system_ready': True,
            'caching_enabled': False
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all integrated systems"""
        return {
            'integration_status': self.integration_status,
            'available_features': [
                'document_summarization',
                'insight_extraction',
                'content_categorization',
                'relationship_analysis',
                'knowledge_gap_analysis',
                'dashboard_generation',
                'multi_format_export'
            ],
            'supported_formats': export_manager.get_supported_formats(),
            'available_methods': summarization_engine.get_available_methods(),
            'available_detail_levels': summarization_engine.get_available_detail_levels()
        }

    def process_document_collection(self, document_paths: List[str],
                                  generate_insights: bool = True,
                                  generate_summaries: bool = True,
                                  analyze_relationships: bool = True) -> Dict[str, Any]:
        """
        Process entire document collection with all insight capabilities

        Args:
            document_paths: List of document paths
            generate_insights: Whether to generate insights
            generate_summaries: Whether to generate summaries
            analyze_relationships: Whether to analyze relationships

        Returns:
            Complete analysis results
        """
        start_time = datetime.now()

        try:
            results = {
                'success': True,
                'processing_summary': {
                    'total_documents': len(document_paths),
                    'start_time': start_time.isoformat(),
                    'processing_time': 0
                },
                'documents': [],
                'collection_analysis': None,
                'relationship_analysis': None,
                'dashboard': None
            }

            # Process individual documents
            for file_path in document_paths:
                doc_result = self.search_engine.insight_processor.process_document_with_insights(file_path)
                results['documents'].append(doc_result)

            # Generate collection-wide analysis
            if analyze_relationships:
                results['relationship_analysis'] = self.insight_api.analyze_relationships(document_paths)

            # Generate dashboard
            results['dashboard'] = self.insight_api.get_dashboard(document_paths)

            # Update timing
            results['processing_summary']['processing_time'] = (
                datetime.now() - start_time
            ).total_seconds()

            return results

        except Exception as e:
            logger.error(f"Collection processing error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Global integration instances
def initialize_insight_integration(search_engine: AdvancedSearchEngine = None) -> InsightIntegrationManager:
    """Initialize the complete insight integration system"""
    return InsightIntegrationManager(search_engine)


# Convenience functions for easy access
def summarize_document(file_path: str, method: str = 'hybrid', detail_level: str = 'concise') -> Dict[str, Any]:
    """Convenience function for document summarization"""
    manager = initialize_insight_integration()
    return manager.insight_api.summarize_document(file_path, method, detail_level)


def extract_insights(file_path: str) -> Dict[str, Any]:
    """Convenience function for insight extraction"""
    manager = initialize_insight_integration()
    return manager.insight_api.extract_insights(file_path)


def analyze_document_collection(document_paths: List[str]) -> Dict[str, Any]:
    """Convenience function for collection analysis"""
    manager = initialize_insight_integration()
    return manager.search_engine.analyze_document_collection(document_paths)