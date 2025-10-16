"""
Comprehensive API for Document Summarization and Insight Services
Provides RESTful API endpoints for all insight and summarization functionality
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file

# Custom imports
from .insight_integration import initialize_insight_integration, InsightIntegrationManager
from .summarization_engine import summarization_engine, SummarizationConfig, SummarizationMethod, DetailLevel
from .insight_extractor import insight_extractor
from .content_categorizer import content_categorizer
from .relationship_analyzer import relationship_analyzer
from .export_manager import export_manager
from .insight_dashboard import dashboard_generator
from .version_aware_insights import version_insight_api
from .compliance_insights import compliance_insight_api

logger = logging.getLogger(__name__)


class InsightAPI:
    """Main API class for insight services"""

    def __init__(self):
        self.integration_manager = initialize_insight_integration()
        self.api_blueprint = Blueprint('insight_api', __name__)

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register API routes"""

        @self.api_blueprint.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'services': self.integration_manager.get_system_status()
            })

        @self.api_blueprint.route('/summarize', methods=['POST'])
        def summarize_document():
            """Document summarization endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                file_path = data.get('file_path')
                if not file_path:
                    return jsonify({'error': 'file_path is required'}), 400

                # Optional parameters
                method = data.get('method', 'hybrid')
                detail_level = data.get('detail_level', 'concise')
                target_length = data.get('target_length')
                target_sentences = data.get('target_sentences')

                # Generate summary
                result = self.integration_manager.insight_api.summarize_document(
                    file_path, method, detail_level,
                    target_length=target_length,
                    target_sentences=target_sentences
                )

                return jsonify(result)

            except Exception as e:
                logger.error(f"Summarization API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/insights', methods=['POST'])
        def extract_insights():
            """Insight extraction endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                file_path = data.get('file_path')
                if not file_path:
                    return jsonify({'error': 'file_path is required'}), 400

                result = self.integration_manager.insight_api.extract_insights(file_path)
                return jsonify(result)

            except Exception as e:
                logger.error(f"Insight extraction API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/categorize', methods=['POST'])
        def categorize_document():
            """Document categorization endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                file_path = data.get('file_path')
                if not file_path:
                    return jsonify({'error': 'file_path is required'}), 400

                result = self.integration_manager.insight_api.categorize_document(file_path)
                return jsonify(result)

            except Exception as e:
                logger.error(f"Categorization API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/relationships', methods=['POST'])
        def analyze_relationships():
            """Relationship analysis endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                document_paths = data.get('document_paths', [])
                if not document_paths:
                    return jsonify({'error': 'document_paths is required'}), 400

                if len(document_paths) < 2:
                    return jsonify({'error': 'At least 2 documents required for relationship analysis'}), 400

                result = self.integration_manager.insight_api.analyze_relationships(document_paths)
                return jsonify(result)

            except Exception as e:
                logger.error(f"Relationship analysis API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/dashboard', methods=['POST'])
        def generate_dashboard():
            """Dashboard generation endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                document_paths = data.get('document_paths', [])
                if not document_paths:
                    return jsonify({'error': 'document_paths is required'}), 400

                result = self.integration_manager.insight_api.get_dashboard(document_paths)
                return jsonify(result)

            except Exception as e:
                logger.error(f"Dashboard generation API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/export', methods=['POST'])
        def export_results():
            """Export results endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                export_data = data.get('data')
                export_format = data.get('format', 'json')
                data_type = data.get('data_type', 'dashboard')
                output_path = data.get('output_path')

                if not export_data:
                    return jsonify({'error': 'data is required'}), 400

                # Handle different data types
                if data_type == 'summary' and hasattr(export_data, 'to_dict'):
                    result = export_manager.export_summary(export_data, export_format, output_path)
                elif data_type == 'insights':
                    result = export_manager.export_insights(export_data, export_format, output_path)
                elif data_type == 'dashboard':
                    result = export_manager.export_dashboard(export_data, export_format, output_path)
                else:
                    return jsonify({'error': f'Unsupported data_type: {data_type}'}), 400

                return jsonify(result)

            except Exception as e:
                logger.error(f"Export API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/version-insights', methods=['POST'])
        def get_version_insights():
            """Version-aware insights endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                version_id = data.get('version_id')
                if not version_id:
                    return jsonify({'error': 'version_id is required'}), 400

                result = version_insight_api.get_version_insights(version_id)
                return jsonify(result)

            except Exception as e:
                logger.error(f"Version insights API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/version-compare', methods=['POST'])
        def compare_versions():
            """Version comparison endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                version1_id = data.get('version1_id')
                version2_id = data.get('version2_id')

                if not version1_id or not version2_id:
                    return jsonify({'error': 'version1_id and version2_id are required'}), 400

                result = version_insight_api.compare_versions(version1_id, version2_id)
                return jsonify(result)

            except Exception as e:
                logger.error(f"Version comparison API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/compliance-insights', methods=['POST'])
        def get_compliance_insights():
            """Compliance insights endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                file_path = data.get('file_path')
                if not file_path:
                    return jsonify({'error': 'file_path is required'}), 400

                result = compliance_insight_api.get_compliance_insights(file_path)
                return jsonify(result)

            except Exception as e:
                logger.error(f"Compliance insights API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/compliance-report', methods=['POST'])
        def generate_compliance_report():
            """Compliance report generation endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                file_path = data.get('file_path')
                output_format = data.get('format', 'pdf')

                if not file_path:
                    return jsonify({'error': 'file_path is required'}), 400

                result = compliance_insight_api.generate_compliance_report(file_path, output_format)
                return jsonify(result)

            except Exception as e:
                logger.error(f"Compliance report API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/analyze-collection', methods=['POST'])
        def analyze_collection():
            """Collection analysis endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                document_paths = data.get('document_paths', [])
                if not document_paths:
                    return jsonify({'error': 'document_paths is required'}), 400

                # Optional analysis flags
                generate_insights = data.get('generate_insights', True)
                generate_summaries = data.get('generate_summaries', True)
                analyze_relationships = data.get('analyze_relationships', True)

                result = self.integration_manager.process_document_collection(
                    document_paths,
                    generate_insights=generate_insights,
                    generate_summaries=generate_summaries,
                    analyze_relationships=analyze_relationships
                )

                return jsonify(result)

            except Exception as e:
                logger.error(f"Collection analysis API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/system-status', methods=['GET'])
        def get_system_status():
            """System status endpoint"""
            return jsonify(self.integration_manager.get_system_status())

        @self.api_blueprint.route('/available-methods', methods=['GET'])
        def get_available_methods():
            """Get available summarization methods"""
            return jsonify({
                'summarization_methods': summarization_engine.get_available_methods(),
                'detail_levels': summarization_engine.get_available_detail_levels(),
                'supported_formats': export_manager.get_supported_formats()
            })

        @self.api_blueprint.route('/batch-summarize', methods=['POST'])
        def batch_summarize():
            """Batch summarization endpoint"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                document_paths = data.get('document_paths', [])
                if not document_paths:
                    return jsonify({'error': 'document_paths is required'}), 400

                # Summarization config
                method = data.get('method', 'hybrid')
                detail_level = data.get('detail_level', 'concise')
                export_format = data.get('export_format', 'json')

                results = []
                for file_path in document_paths:
                    try:
                        summary_result = self.integration_manager.insight_api.summarize_document(
                            file_path, method, detail_level
                        )

                        if summary_result['success']:
                            # Export each summary
                            export_data = {
                                'type': 'summary',
                                'data': summary_result.get('metadata'),  # SummarizationResult object
                                'filename': f"summary_{Path(file_path).stem}"
                            }

                            export_result = export_manager.batch_export([export_data], export_format)
                            summary_result['export'] = export_result

                        results.append({
                            'file_path': file_path,
                            'result': summary_result
                        })

                    except Exception as e:
                        results.append({
                            'file_path': file_path,
                            'result': {'success': False, 'error': str(e)}
                        })

                return jsonify({
                    'success': True,
                    'batch_results': results,
                    'total_processed': len(results),
                    'successful_summaries': len([r for r in results if r['result'].get('success')])
                })

            except Exception as e:
                logger.error(f"Batch summarization API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.api_blueprint.route('/search-enhanced', methods=['POST'])
        def search_enhanced():
            """Enhanced search with insights"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                query = data.get('query')
                if not query:
                    return jsonify({'error': 'query is required'}), 400

                # Optional parameters
                limit = data.get('limit', 10)
                include_insights = data.get('include_insights', False)
                include_summaries = data.get('include_summaries', False)
                filters = data.get('filters', {})

                result = self.integration_manager.search_engine.search_with_insights(
                    query,
                    include_insights=include_insights,
                    include_summaries=include_summaries,
                    limit=limit,
                    filters=filters
                )

                return jsonify(result)

            except Exception as e:
                logger.error(f"Enhanced search API error: {e}")
                return jsonify({'error': str(e)}), 500

    def get_blueprint(self):
        """Get the Flask blueprint for registration"""
        return self.api_blueprint


class InsightAPIClient:
    """Client for accessing insight API endpoints"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')

    def _make_request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to API"""
        import requests

        url = f"{self.base_url}{endpoint}"

        try:
            if method == 'GET':
                response = requests.get(url)
            elif method == 'POST':
                response = requests.post(url, json=data)
            else:
                return {'error': f'Unsupported method: {method}'}

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {'error': f'API request failed: {e}'}

    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return self._make_request('/api/health')

    def summarize_document(self, file_path: str, method: str = 'hybrid',
                          detail_level: str = 'concise', **kwargs) -> Dict[str, Any]:
        """Summarize document"""
        data = {
            'file_path': file_path,
            'method': method,
            'detail_level': detail_level,
            **kwargs
        }
        return self._make_request('/api/summarize', 'POST', data)

    def extract_insights(self, file_path: str) -> Dict[str, Any]:
        """Extract insights"""
        data = {'file_path': file_path}
        return self._make_request('/api/insights', 'POST', data)

    def categorize_document(self, file_path: str) -> Dict[str, Any]:
        """Categorize document"""
        data = {'file_path': file_path}
        return self._make_request('/api/categorize', 'POST', data)

    def analyze_relationships(self, document_paths: List[str]) -> Dict[str, Any]:
        """Analyze relationships"""
        data = {'document_paths': document_paths}
        return self._make_request('/api/relationships', 'POST', data)

    def generate_dashboard(self, document_paths: List[str]) -> Dict[str, Any]:
        """Generate dashboard"""
        data = {'document_paths': document_paths}
        return self._make_request('/api/dashboard', 'POST', data)

    def export_results(self, data: Any, export_format: str, data_type: str,
                      output_path: str = None) -> Dict[str, Any]:
        """Export results"""
        export_data = {
            'data': data,
            'format': export_format,
            'data_type': data_type
        }
        if output_path:
            export_data['output_path'] = output_path

        return self._make_request('/api/export', 'POST', export_data)

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self._make_request('/api/system-status')

    def get_available_methods(self) -> Dict[str, Any]:
        """Get available methods and formats"""
        return self._make_request('/api/available-methods')


def register_insight_api(app, url_prefix: str = '/api'):
    """
    Register insight API with Flask application

    Args:
        app: Flask application instance
        url_prefix: URL prefix for API routes
    """
    api = InsightAPI()
    app.register_blueprint(api.get_blueprint(), url_prefix=url_prefix)


def create_insight_api_server(host: str = '127.0.0.1', port: int = 5000,
                             debug: bool = False):
    """
    Create and run standalone insight API server

    Args:
        host: Server host
        port: Server port
        debug: Debug mode
    """
    from flask import Flask

    app = Flask(__name__)
    register_insight_api(app)

    logger.info(f"Starting Insight API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


# Global API instances
insight_api_server = InsightAPI()
insight_api_client = InsightAPIClient()


def initialize_api():
    """Initialize the insight API system"""
    logger.info("Insight API system initialized")
    return insight_api_server


# Convenience functions for direct API access
def api_summarize(file_path: str, method: str = 'hybrid', detail_level: str = 'concise') -> Dict[str, Any]:
    """Direct API access for document summarization"""
    return insight_api_server.integration_manager.insight_api.summarize_document(file_path, method, detail_level)


def api_extract_insights(file_path: str) -> Dict[str, Any]:
    """Direct API access for insight extraction"""
    return insight_api_server.integration_manager.insight_api.extract_insights(file_path)


def api_generate_dashboard(document_paths: List[str]) -> Dict[str, Any]:
    """Direct API access for dashboard generation"""
    return insight_api_server.integration_manager.insight_api.get_dashboard(document_paths)