"""
Integration Module for Compliance Monitoring System
Integrates compliance monitoring with existing document management systems
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from .compliance_models import ComplianceDataManager
    from .compliance_scanner import ComplianceScannerManager
    from .compliance_dashboard import ComplianceDashboard
    from .compliance_reporting import ComplianceReportingManager
    from .compliance_workflows import ComplianceViolationManager
    from .compliance_analytics import ComplianceAnalyticsEngine
except ImportError:
    from compliance_models import ComplianceDataManager
    from compliance_scanner import ComplianceScannerManager
    from compliance_dashboard import ComplianceDashboard
    from compliance_reporting import ComplianceReportingManager
    from compliance_workflows import ComplianceViolationManager
    from compliance_analytics import ComplianceAnalyticsEngine

logger = logging.getLogger(__name__)


class ComplianceMetadataIntegrator:
    """Integrates compliance data with document metadata"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()

    def enhance_document_metadata(self, doc_id: str, existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance existing document metadata with compliance information"""
        try:
            enhanced_metadata = existing_metadata.copy()

            # Get compliance assessments for this document
            assessments = self.data_manager.load_assessments(doc_id)

            if assessments:
                # Get latest assessment
                latest_assessment = max(assessments, key=lambda a: a.assessed_at)

                # Add compliance information to metadata
                enhanced_metadata['compliance'] = {
                    'overall_status': latest_assessment.overall_status,
                    'compliance_score': latest_assessment.compliance_score,
                    'risk_level': latest_assessment.risk_level,
                    'last_assessed': latest_assessment.assessed_at.isoformat(),
                    'total_assessments': len(assessments),
                    'active_violations': len([
                        a for a in assessments
                        if any(v for v in a.violations if v.get('status') == 'open')
                    ])
                }

                # Add compliance badges for search results
                if latest_assessment.overall_status == 'compliant':
                    enhanced_metadata['compliance_badge'] = '✅ Compliant'
                elif latest_assessment.overall_status == 'non_compliant':
                    enhanced_metadata['compliance_badge'] = '❌ Non-Compliant'
                else:
                    enhanced_metadata['compliance_badge'] = '⚠️ Partial Compliance'

            # Get violations for this document
            violations = self.data_manager.load_violations(doc_id)
            if violations:
                open_violations = [v for v in violations if v.status == 'open']
                enhanced_metadata['violation_count'] = len(violations)
                enhanced_metadata['open_violations'] = len(open_violations)

                if open_violations:
                    highest_severity = max(open_violations, key=lambda v: self._severity_rank(v.severity))
                    enhanced_metadata['highest_violation_severity'] = highest_severity.severity

            return enhanced_metadata

        except Exception as e:
            logger.error(f"Error enhancing metadata for {doc_id}: {e}")
            return existing_metadata

    def _severity_rank(self, severity: str) -> int:
        """Get numeric rank for violation severity"""
        ranks = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        return ranks.get(severity, 0)


class ComplianceSearchIntegrator:
    """Integrates compliance data with search functionality"""

    def __init__(self):
        self.metadata_integrator = ComplianceMetadataIntegrator()

    def add_compliance_filters(self) -> Dict[str, Any]:
        """Add compliance-related search filters"""
        return {
            'compliance_status': {
                'type': 'select',
                'options': ['compliant', 'non_compliant', 'partial', 'not_assessed'],
                'label': 'Compliance Status'
            },
            'risk_level': {
                'type': 'select',
                'options': ['low', 'medium', 'high', 'critical'],
                'label': 'Risk Level'
            },
            'has_violations': {
                'type': 'boolean',
                'label': 'Has Open Violations'
            },
            'compliance_score_min': {
                'type': 'number',
                'label': 'Minimum Compliance Score',
                'min': 0,
                'max': 100
            },
            'last_assessed_days': {
                'type': 'number',
                'label': 'Assessed Within Days',
                'min': 1,
                'max': 365
            }
        }

    def apply_compliance_filtering(self, documents: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply compliance filters to search results"""
        if not filters:
            return documents

        filtered_documents = []

        for doc in documents:
            doc_id = doc.get('doc_id')
            if not doc_id:
                continue

            # Enhance document with compliance data
            enhanced_doc = self.metadata_integrator.enhance_document_metadata(doc_id, doc)

            # Apply filters
            include_doc = True

            # Compliance status filter
            if 'compliance_status' in filters:
                doc_status = enhanced_doc.get('compliance', {}).get('overall_status', 'not_assessed')
                if doc_status != filters['compliance_status']:
                    include_doc = False

            # Risk level filter
            if 'risk_level' in filters and include_doc:
                doc_risk = enhanced_doc.get('compliance', {}).get('risk_level', 'unknown')
                if doc_risk != filters['risk_level']:
                    include_doc = False

            # Has violations filter
            if 'has_violations' in filters and include_doc:
                open_violations = enhanced_doc.get('open_violations', 0)
                has_violations = open_violations > 0
                if has_violations != filters['has_violations']:
                    include_doc = False

            # Compliance score filter
            if 'compliance_score_min' in filters and include_doc:
                compliance_score = enhanced_doc.get('compliance', {}).get('compliance_score', 0)
                if compliance_score < filters['compliance_score_min']:
                    include_doc = False

            # Last assessed filter
            if 'last_assessed_days' in filters and include_doc:
                last_assessed = enhanced_doc.get('compliance', {}).get('last_assessed')
                if last_assessed:
                    days_since = (datetime.now() - datetime.fromisoformat(last_assessed)).days
                    if days_since > filters['last_assessed_days']:
                        include_doc = False

            if include_doc:
                filtered_documents.append(enhanced_doc)

        return filtered_documents

    def add_compliance_facets(self, documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Add compliance-related facets to search results"""
        facets = {
            'compliance_status': {},
            'risk_level': {},
            'has_violations': {},
            'compliance_score_range': {}
        }

        for doc in documents:
            compliance_info = doc.get('compliance', {})

            # Compliance status facet
            status = compliance_info.get('overall_status', 'not_assessed')
            facets['compliance_status'][status] = facets['compliance_status'].get(status, 0) + 1

            # Risk level facet
            risk = compliance_info.get('risk_level', 'unknown')
            facets['risk_level'][risk] = facets['risk_level'].get(risk, 0) + 1

            # Has violations facet
            has_violations = 'yes' if doc.get('open_violations', 0) > 0 else 'no'
            facets['has_violations'][has_violations] = facets['has_violations'].get(has_violations, 0) + 1

            # Compliance score range facet
            score = compliance_info.get('compliance_score', 0)
            if score >= 90:
                score_range = '90-100'
            elif score >= 70:
                score_range = '70-89'
            elif score >= 50:
                score_range = '50-69'
            else:
                score_range = '0-49'

            facets['compliance_score_range'][score_range] = facets['compliance_score_range'].get(score_range, 0) + 1

        return facets


class ComplianceVersionIntegrator:
    """Integrates compliance monitoring with version control"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()

    def check_version_compliance_impact(self, doc_id: str, new_version_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance impact of a new document version"""
        try:
            # Get current compliance status
            current_assessments = self.data_manager.load_assessments(doc_id)

            if not current_assessments:
                return {
                    'impact_level': 'unknown',
                    'message': 'No existing compliance data for comparison'
                }

            # Analyze potential compliance changes
            # This would involve comparing the new version content with compliance requirements
            # For now, return a basic analysis

            latest_assessment = max(current_assessments, key=lambda a: a.assessed_at)

            return {
                'current_compliance_score': latest_assessment.compliance_score,
                'current_risk_level': latest_assessment.risk_level,
                'impact_level': 'medium',  # Would be calculated based on content analysis
                'recommended_actions': [
                    'Re-run compliance assessment after version update',
                    'Review changes for compliance implications'
                ],
                'assessment_required': True
            }

        except Exception as e:
            logger.error(f"Error checking version compliance impact for {doc_id}: {e}")
            return {
                'impact_level': 'error',
                'message': str(e)
            }

    def trigger_compliance_scan_on_version_change(self, doc_id: str, version_info: Dict[str, Any]) -> bool:
        """Trigger compliance scan when document version changes"""
        try:
            # Get file path for the document
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if not catalog_path.exists():
                return False

            with open(catalog_path, 'r') as f:
                catalog = json.load(f)

            if doc_id not in catalog:
                return False

            file_path = catalog[doc_id].get('file_path')
            if not file_path or not Path(file_path).exists():
                return False

            # Trigger compliance scan
            scanner_manager = ComplianceScannerManager()
            result = scanner_manager.scan_document_by_id(doc_id)

            if result.get('success'):
                logger.info(f"Compliance scan triggered for version change of {doc_id}")
                return True
            else:
                logger.error(f"Failed to scan document {doc_id} after version change: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Error triggering compliance scan for {doc_id}: {e}")
            return False


class ComplianceLifecycleIntegrator:
    """Integrates compliance with document lifecycle management"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()

    def get_compliance_requirements_for_lifecycle_stage(self, stage: str) -> List[str]:
        """Get compliance requirements for a specific lifecycle stage"""
        requirements = []

        try:
            all_requirements = self.data_manager.load_requirements()

            for requirement in all_requirements:
                # Map lifecycle stages to requirement categories
                if stage == 'draft' and 'draft' in requirement.description.lower():
                    requirements.append(requirement.requirement_id)
                elif stage == 'review' and 'review' in requirement.description.lower():
                    requirements.append(requirement.requirement_id)
                elif stage == 'approved' and 'approval' in requirement.description.lower():
                    requirements.append(requirement.requirement_id)
                elif stage == 'published' and 'published' in requirement.description.lower():
                    requirements.append(requirement.requirement_id)

        except Exception as e:
            logger.error(f"Error getting compliance requirements for stage {stage}: {e}")

        return requirements

    def validate_lifecycle_transition(self, doc_id: str, from_stage: str, to_stage: str) -> Dict[str, Any]:
        """Validate if a lifecycle transition is compliant"""
        try:
            # Get current compliance status
            assessments = self.data_manager.load_assessments(doc_id)

            if not assessments:
                return {
                    'valid': False,
                    'reason': 'No compliance assessments found',
                    'blocking_violations': []
                }

            latest_assessment = max(assessments, key=lambda a: a.assessed_at)

            # Check for blocking violations
            blocking_violations = []
            for violation_data in latest_assessment.violations:
                if violation_data.get('severity') in ['critical', 'high']:
                    blocking_violations.append(violation_data)

            # Determine if transition is valid
            if to_stage in ['approved', 'published'] and blocking_violations:
                return {
                    'valid': False,
                    'reason': f'Cannot transition to {to_stage} with {len(blocking_violations)} critical/high violations',
                    'blocking_violations': blocking_violations,
                    'required_actions': ['Resolve all critical and high violations before proceeding']
                }

            return {
                'valid': True,
                'reason': 'Transition is compliant',
                'current_compliance_score': latest_assessment.compliance_score,
                'recommendations': self._get_transition_recommendations(to_stage, latest_assessment)
            }

        except Exception as e:
            logger.error(f"Error validating lifecycle transition for {doc_id}: {e}")
            return {
                'valid': False,
                'reason': f'Error during validation: {str(e)}'
            }

    def _get_transition_recommendations(self, to_stage: str, assessment) -> List[str]:
        """Get recommendations for lifecycle transitions"""
        recommendations = []

        if to_stage == 'approved':
            if assessment.compliance_score < 80:
                recommendations.append('Consider additional compliance review before approval')
        elif to_stage == 'published':
            if assessment.risk_level in ['high', 'critical']:
                recommendations.append('High-risk document - consider additional controls')

        return recommendations


class ComplianceIntegrationManager:
    """Main integration manager for all compliance systems"""

    def __init__(self):
        self.metadata_integrator = ComplianceMetadataIntegrator()
        self.search_integrator = ComplianceSearchIntegrator()
        self.version_integrator = ComplianceVersionIntegrator()
        self.lifecycle_integrator = ComplianceLifecycleIntegrator()

        # Core compliance components
        self.scanner_manager = ComplianceScannerManager()
        self.dashboard = ComplianceDashboard()
        self.reporting_manager = ComplianceReportingManager()
        self.violation_manager = ComplianceViolationManager()
        self.analytics_engine = ComplianceAnalyticsEngine()

    def initialize_compliance_integration(self) -> bool:
        """Initialize compliance integration with existing systems"""
        try:
            logger.info("Initializing compliance monitoring integration...")

            # Create compliance data directory
            compliance_dir = Path("data/compliance")
            compliance_dir.mkdir(parents=True, exist_ok=True)

            # Initialize default compliance requirements (skip if in test mode)
            try:
                self._initialize_default_requirements()
            except ImportError:
                # Skip initialization if modules not available (test mode)
                pass

            logger.info("Compliance monitoring integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing compliance integration: {e}")
            return False

    def _initialize_default_requirements(self) -> None:
        """Initialize default compliance requirements"""
        try:
            from .compliance_config import all_requirements

            # Save default requirements if not already present
            existing_requirements = self.metadata_integrator.data_manager.load_requirements()

            if not existing_requirements:
                success = self.metadata_integrator.data_manager.save_requirements(all_requirements)
                if success:
                    logger.info(f"Initialized {len(all_requirements)} default compliance requirements")
                else:
                    logger.error("Failed to save default compliance requirements")

        except Exception as e:
            logger.error(f"Error initializing default requirements: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all integrated systems"""
        try:
            return {
                'compliance_system': {
                    'status': 'active',
                    'components': {
                        'scanner': self.scanner_manager.get_scan_status(),
                        'dashboard': {'status': 'active'},
                        'reporting': {'status': 'active'},
                        'workflows': {'status': 'active'},
                        'analytics': {'status': 'active'}
                    }
                },
                'integration_points': {
                    'metadata_integration': 'active',
                    'search_integration': 'active',
                    'version_control_integration': 'active',
                    'lifecycle_integration': 'active'
                },
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'status': 'error', 'message': str(e)}

    def process_document_ingestion(self, doc_id: str, file_path: str) -> Dict[str, Any]:
        """Process a newly ingested document for compliance"""
        try:
            # Run compliance scan
            scan_result = self.scanner_manager.scan_document_by_id(doc_id)

            if scan_result.get('success'):
                # Process any violations found
                assessments = scan_result.get('assessments', [])
                for assessment_data in assessments:
                    # Extract violations and process workflows
                    for violation_data in assessment_data.get('violations', []):
                        # This would create violation objects and trigger workflows
                        pass

                return {
                    'success': True,
                    'assessments_performed': len(assessments),
                    'message': f'Document {doc_id} processed for compliance'
                }
            else:
                return {
                    'success': False,
                    'error': scan_result.get('error', 'Unknown error during compliance scan')
                }

        except Exception as e:
            logger.error(f"Error processing document ingestion for {doc_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def enhance_search_results(self, search_results: List[Dict[str, Any]], filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Enhance search results with compliance information"""
        try:
            # Enhance each result with compliance data
            enhanced_results = []
            for result in search_results:
                doc_id = result.get('doc_id')
                if doc_id:
                    enhanced_result = self.metadata_integrator.enhance_document_metadata(doc_id, result)
                    enhanced_results.append(enhanced_result)
                else:
                    enhanced_results.append(result)

            # Apply compliance filters if provided
            if filters:
                enhanced_results = self.search_integrator.apply_compliance_filtering(enhanced_results, filters)

            return enhanced_results

        except Exception as e:
            logger.error(f"Error enhancing search results: {e}")
            return search_results

    def add_compliance_search_facets(self, documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Add compliance facets to search results"""
        return self.search_integrator.add_compliance_facets(documents)

    def validate_version_change(self, doc_id: str, version_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance impact of version change"""
        return self.version_integrator.check_version_compliance_impact(doc_id, version_info)

    def validate_lifecycle_transition(self, doc_id: str, from_stage: str, to_stage: str) -> Dict[str, Any]:
        """Validate compliance for lifecycle transition"""
        return self.lifecycle_integrator.validate_lifecycle_transition(doc_id, from_stage, to_stage)


# Global compliance integration manager
compliance_integration_manager = ComplianceIntegrationManager()

# Auto-initialize compliance integration
compliance_integration_manager.initialize_compliance_integration()