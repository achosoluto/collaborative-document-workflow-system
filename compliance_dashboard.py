"""
Compliance Dashboard with Real-time Monitoring
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from flask import Blueprint, render_template, jsonify, request

try:
    from .compliance_models import ComplianceDataManager, ComplianceAssessment, ComplianceViolation
    from .compliance_scanner import ComplianceScannerManager
    from .compliance_tracker import ComplianceTracker
except ImportError:
    from compliance_models import ComplianceDataManager, ComplianceAssessment, ComplianceViolation
    from compliance_scanner import ComplianceScannerManager
    from compliance_tracker import ComplianceTracker

logger = logging.getLogger(__name__)


class ComplianceDashboard:
    """Compliance dashboard with real-time monitoring"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()
        self.scanner_manager = ComplianceScannerManager()
        self.tracker = ComplianceTracker()

        # Dashboard settings
        self.refresh_interval = 30  # seconds
        self.max_recent_items = 50

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Overall compliance statistics
            overall_stats = self._get_overall_statistics()

            # Recent assessments
            recent_assessments = self._get_recent_assessments()

            # Active violations
            active_violations = self._get_active_violations()

            # Compliance trends
            compliance_trends = self._get_compliance_trends()

            # Upcoming deadlines
            upcoming_deadlines = self._get_upcoming_deadlines()

            # System status
            system_status = self._get_system_status()

            return {
                'overall_statistics': overall_stats,
                'recent_assessments': recent_assessments,
                'active_violations': active_violations,
                'compliance_trends': compliance_trends,
                'upcoming_deadlines': upcoming_deadlines,
                'system_status': system_status,
                'last_updated': datetime.now().isoformat(),
                'refresh_interval': self.refresh_interval
            }

        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }

    def _get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall compliance statistics"""
        try:
            # Load all assessments
            assessments = self.data_manager.load_assessments()

            if not assessments:
                return {
                    'total_documents': 0,
                    'assessed_documents': 0,
                    'compliance_rate': 0.0,
                    'total_violations': 0,
                    'critical_violations': 0,
                    'high_violations': 0,
                    'medium_violations': 0,
                    'low_violations': 0
                }

            # Calculate statistics
            total_assessments = len(assessments)
            compliant_assessments = len([a for a in assessments if a.overall_status == 'compliant'])
            non_compliant_assessments = len([a for a in assessments if a.overall_status == 'non_compliant'])
            partial_assessments = len([a for a in assessments if a.overall_status == 'partial'])

            compliance_rate = (compliant_assessments / total_assessments * 100) if total_assessments > 0 else 0

            # Load violations for statistics
            violations = self.data_manager.load_violations()
            critical_violations = len([v for v in violations if v.severity == 'critical' and v.status == 'open'])
            high_violations = len([v for v in violations if v.severity == 'high' and v.status == 'open'])
            medium_violations = len([v for v in violations if v.severity == 'medium' and v.status == 'open'])
            low_violations = len([v for v in violations if v.severity == 'low' and v.status == 'open'])

            # Get document count from catalog
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            total_documents = 0
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    catalog = json.load(f)
                total_documents = len(catalog)

            return {
                'total_documents': total_documents,
                'assessed_documents': total_assessments,
                'compliance_rate': round(compliance_rate, 2),
                'total_violations': len(violations),
                'open_violations': len([v for v in violations if v.status == 'open']),
                'resolved_violations': len([v for v in violations if v.status == 'resolved']),
                'critical_violations': critical_violations,
                'high_violations': high_violations,
                'medium_violations': medium_violations,
                'low_violations': low_violations,
                'compliant_assessments': compliant_assessments,
                'non_compliant_assessments': non_compliant_assessments,
                'partial_assessments': partial_assessments
            }

        except Exception as e:
            logger.error(f"Error getting overall statistics: {e}")
            return {}

    def _get_recent_assessments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent compliance assessments"""
        try:
            assessments = self.data_manager.load_assessments()

            # Sort by assessment date (newest first)
            recent_assessments = sorted(
                assessments,
                key=lambda a: a.assessed_at,
                reverse=True
            )[:limit]

            result = []
            for assessment in recent_assessments:
                # Get document title from catalog
                doc_title = "Unknown Document"
                catalog_path = Path(__file__).parent.parent / "document_catalog.json"
                if catalog_path.exists():
                    with open(catalog_path, 'r') as f:
                        catalog = json.load(f)
                    if assessment.doc_id in catalog:
                        doc_title = catalog[assessment.doc_id].get('file_name', assessment.doc_id)

                result.append({
                    'assessment_id': assessment.assessment_id,
                    'doc_id': assessment.doc_id,
                    'doc_title': doc_title,
                    'requirement_id': assessment.requirement_id,
                    'overall_status': assessment.overall_status,
                    'compliance_score': round(assessment.compliance_score, 2),
                    'risk_level': assessment.risk_level,
                    'assessed_at': assessment.assessed_at.isoformat(),
                    'violations_count': len(assessment.violations),
                    'warnings_count': len(assessment.warnings)
                })

            return result

        except Exception as e:
            logger.error(f"Error getting recent assessments: {e}")
            return []

    def _get_active_violations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get active compliance violations"""
        try:
            violations = self.data_manager.load_violations(status='open')

            # Sort by severity and creation date
            def sort_key(v):
                severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                return (severity_order.get(v.severity, 4), v.created_at)

            sorted_violations = sorted(violations, key=sort_key)[:limit]

            result = []
            for violation in sorted_violations:
                # Get document title
                doc_title = "Unknown Document"
                catalog_path = Path(__file__).parent.parent / "document_catalog.json"
                if catalog_path.exists():
                    with open(catalog_path, 'r') as f:
                        catalog = json.load(f)
                    if violation.doc_id in catalog:
                        doc_title = catalog[violation.doc_id].get('file_name', violation.doc_id)

                result.append({
                    'violation_id': violation.violation_id,
                    'doc_id': violation.doc_id,
                    'doc_title': doc_title,
                    'violation_type': violation.violation_type,
                    'severity': violation.severity,
                    'title': violation.title,
                    'description': violation.description,
                    'status': violation.status,
                    'created_at': violation.created_at.isoformat(),
                    'days_open': (datetime.now() - violation.created_at).days
                })

            return result

        except Exception as e:
            logger.error(f"Error getting active violations: {e}")
            return []

    def _get_compliance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get compliance trends over time"""
        try:
            assessments = self.data_manager.load_assessments()

            # Filter assessments from the last N days
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_assessments = [
                a for a in assessments
                if a.assessed_at >= cutoff_date
            ]

            if not recent_assessments:
                return {
                    'daily_scores': [],
                    'trend_direction': 'no_data',
                    'avg_score': 0.0
                }

            # Group by day and calculate daily averages
            daily_scores = {}
            for assessment in recent_assessments:
                day_key = assessment.assessed_at.date().isoformat()
                if day_key not in daily_scores:
                    daily_scores[day_key] = []
                daily_scores[day_key].append(assessment.compliance_score)

            # Calculate daily averages
            daily_averages = []
            for day, scores in sorted(daily_scores.items()):
                avg_score = sum(scores) / len(scores)
                daily_averages.append({
                    'date': day,
                    'average_score': round(avg_score, 2),
                    'assessment_count': len(scores)
                })

            # Calculate trend direction
            if len(daily_averages) >= 2:
                first_half = daily_averages[:len(daily_averages)//2]
                second_half = daily_averages[len(daily_averages)//2:]

                first_avg = sum(d['average_score'] for d in first_half) / len(first_half)
                second_avg = sum(d['average_score'] for d in second_half) / len(second_half)

                if second_avg > first_avg + 2:
                    trend_direction = 'improving'
                elif second_avg < first_avg - 2:
                    trend_direction = 'declining'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'insufficient_data'

            overall_avg = sum(d['average_score'] for d in daily_averages) / len(daily_averages)

            return {
                'daily_scores': daily_averages,
                'trend_direction': trend_direction,
                'avg_score': round(overall_avg, 2),
                'period_days': days
            }

        except Exception as e:
            logger.error(f"Error getting compliance trends: {e}")
            return {
                'daily_scores': [],
                'trend_direction': 'error',
                'avg_score': 0.0
            }

    def _get_upcoming_deadlines(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """Get upcoming compliance deadlines"""
        return self.tracker.get_deadlines(days_ahead)

    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        try:
            return {
                'scanner_status': self.scanner_manager.get_scan_status(),
                'data_manager_status': 'active',
                'last_scan': self._get_last_scan_time(),
                'database_size': self._get_database_size(),
                'error_count': self._get_error_count()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'status': 'error', 'message': str(e)}

    def _get_last_scan_time(self) -> Optional[str]:
        """Get the last scan time"""
        try:
            assessments = self.data_manager.load_assessments()
            if assessments:
                latest = max(assessments, key=lambda a: a.assessed_at)
                return latest.assessed_at.isoformat()
            return None
        except:
            return None

    def _get_database_size(self) -> Dict[str, int]:
        """Get database size information"""
        try:
            sizes = {}

            # Check compliance data files
            data_dir = Path("data/compliance")
            if data_dir.exists():
                for file_path in data_dir.glob("*.json"):
                    try:
                        sizes[file_path.name] = file_path.stat().st_size
                    except:
                        pass

            return sizes
        except:
            return {}

    def _get_error_count(self) -> int:
        """Get recent error count"""
        # This would integrate with logging system
        # For now, return 0
        return 0

    def get_document_details(self, doc_id: str) -> Dict[str, Any]:
        """Get detailed compliance information for a specific document"""
        try:
            # Get document metadata
            doc_metadata = None
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    catalog = json.load(f)
                if doc_id in catalog:
                    doc_metadata = catalog[doc_id]

            # Get assessments for this document
            assessments = self.data_manager.load_assessments(doc_id)

            # Get violations for this document
            violations = self.data_manager.load_violations(doc_id)

            # Calculate document compliance score
            if assessments:
                avg_score = sum(a.compliance_score for a in assessments) / len(assessments)
                latest_assessment = max(assessments, key=lambda a: a.assessed_at)
                overall_status = latest_assessment.overall_status
            else:
                avg_score = 0.0
                overall_status = 'not_assessed'

            return {
                'doc_id': doc_id,
                'document_metadata': doc_metadata,
                'compliance_score': round(avg_score, 2),
                'overall_status': overall_status,
                'total_assessments': len(assessments),
                'assessments': [a.to_dict() for a in assessments],
                'total_violations': len(violations),
                'open_violations': len([v for v in violations if v.status == 'open']),
                'violations': [v.to_dict() for v in violations],
                'last_assessed': max((a.assessed_at for a in assessments), default=None)
            }

        except Exception as e:
            logger.error(f"Error getting document details for {doc_id}: {e}")
            return {'error': str(e)}

    def get_violation_details(self, violation_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific violation"""
        try:
            violations = self.data_manager.load_violations()
            violation = next((v for v in violations if v.violation_id == violation_id), None)

            if not violation:
                return {'error': f'Violation {violation_id} not found'}

            # Get related assessment
            assessments = self.data_manager.load_assessments()
            assessment = next((a for a in assessments if a.assessment_id == violation.assessment_id), None)

            # Get document metadata
            doc_metadata = None
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    catalog = json.load(f)
                if violation.doc_id in catalog:
                    doc_metadata = catalog[violation.doc_id]

            return {
                'violation': violation.to_dict(),
                'assessment': assessment.to_dict() if assessment else None,
                'document_metadata': doc_metadata,
                'remediation_suggestions': self._get_remediation_suggestions(violation)
            }

        except Exception as e:
            logger.error(f"Error getting violation details for {violation_id}: {e}")
            return {'error': str(e)}

    def _get_remediation_suggestions(self, violation) -> List[str]:
        """Get remediation suggestions for a violation"""
        suggestions = []

        if violation.violation_type == 'missing_content':
            suggestions.append("Add the required content to the document")
            suggestions.append("Update document sections to include missing information")
        elif violation.violation_type == 'prohibited_content':
            suggestions.append("Remove or replace the prohibited content")
            suggestions.append("Review document for compliance with content policies")
        elif violation.violation_type == 'structural_issue':
            suggestions.append("Improve document structure and organization")
            suggestions.append("Add required sections and subsections")
        else:
            suggestions.append("Review document for compliance with requirements")
            suggestions.append("Consult compliance team for guidance")

        return suggestions


# Flask blueprint for dashboard routes
compliance_dashboard_bp = Blueprint('compliance_dashboard', __name__)

@compliance_dashboard_bp.route('/api/compliance/dashboard')
def api_dashboard():
    """API endpoint for dashboard data"""
    dashboard = ComplianceDashboard()
    data = dashboard.get_dashboard_data()
    return jsonify(data)

@compliance_dashboard_bp.route('/api/compliance/document/<doc_id>')
def api_document_details(doc_id):
    """API endpoint for document compliance details"""
    dashboard = ComplianceDashboard()
    data = dashboard.get_document_details(doc_id)
    return jsonify(data)

@compliance_dashboard_bp.route('/api/compliance/violation/<violation_id>')
def api_violation_details(violation_id):
    """API endpoint for violation details"""
    dashboard = ComplianceDashboard()
    data = dashboard.get_violation_details(violation_id)
    return jsonify(data)

@compliance_dashboard_bp.route('/api/compliance/scan/<doc_id>', methods=['POST'])
def api_scan_document(doc_id):
    """API endpoint to trigger document scan"""
    scanner_manager = ComplianceScannerManager()
    result = scanner_manager.scan_document_by_id(doc_id)
    return jsonify(result)

@compliance_dashboard_bp.route('/api/compliance/requirements')
def api_requirements():
    """API endpoint for compliance requirements"""
    tracker = ComplianceTracker()
    data = tracker.track_all_requirements()
    return jsonify(data)

@compliance_dashboard_bp.route('/api/compliance/deadlines')
def api_deadlines():
    """API endpoint for upcoming deadlines"""
    tracker = ComplianceTracker()
    days = request.args.get('days', default=30, type=int)
    data = tracker.get_deadlines(days)
    return jsonify(data)