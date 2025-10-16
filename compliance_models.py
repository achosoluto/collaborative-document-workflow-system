"""
Data models for the Compliance Monitoring System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import uuid


@dataclass
class ComplianceRequirement:
    """Represents a regulatory or compliance requirement"""

    requirement_id: str
    title: str
    description: str
    category: str  # 'regulatory', 'policy', 'procedural', 'security'
    subcategory: str  # More specific classification
    severity: str  # 'critical', 'high', 'medium', 'low'

    # Requirement details
    applicable_systems: List[str] = field(default_factory=list)
    applicable_document_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Compliance criteria
    must_contain_keywords: List[str] = field(default_factory=list)
    must_not_contain_keywords: List[str] = field(default_factory=list)
    required_sections: List[str] = field(default_factory=list)
    prohibited_content: List[str] = field(default_factory=list)

    # Metadata
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    created_by: str = ""

    # Status
    is_active: bool = True
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'requirement_id': self.requirement_id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'subcategory': self.subcategory,
            'severity': self.severity,
            'applicable_systems': self.applicable_systems,
            'applicable_document_types': self.applicable_document_types,
            'keywords': self.keywords,
            'must_contain_keywords': self.must_contain_keywords,
            'must_not_contain_keywords': self.must_not_contain_keywords,
            'required_sections': self.required_sections,
            'prohibited_content': self.prohibited_content,
            'effective_date': self.effective_date.isoformat() if self.effective_date else None,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'last_updated': self.last_updated.isoformat(),
            'created_by': self.created_by,
            'is_active': self.is_active,
            'version': self.version
        }


@dataclass
class ComplianceAssessment:
    """Results of compliance assessment for a document"""

    assessment_id: str
    doc_id: str
    requirement_id: str

    # Assessment results
    overall_status: str  # 'compliant', 'non_compliant', 'partial', 'not_applicable'
    compliance_score: float  # 0.0 to 100.0
    risk_level: str  # 'low', 'medium', 'high', 'critical'

    # Detailed findings
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Assessment metadata
    assessed_at: datetime = field(default_factory=datetime.now)
    assessed_by: str = "system"
    assessment_version: str = "1.0"

    # Document context
    document_version: Optional[str] = None
    document_sections_analyzed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'assessment_id': self.assessment_id,
            'doc_id': self.doc_id,
            'requirement_id': self.requirement_id,
            'overall_status': self.overall_status,
            'compliance_score': self.compliance_score,
            'risk_level': self.risk_level,
            'violations': self.violations,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'assessed_at': self.assessed_at.isoformat(),
            'assessed_by': self.assessed_by,
            'assessment_version': self.assessment_version,
            'document_version': self.document_version,
            'document_sections_analyzed': self.document_sections_analyzed
        }


@dataclass
class ComplianceViolation:
    """Represents a specific compliance violation"""

    violation_id: str
    assessment_id: str
    doc_id: str
    requirement_id: str

    # Violation details
    violation_type: str  # 'missing_content', 'prohibited_content', 'structural_issue', 'metadata_issue'
    severity: str  # 'critical', 'high', 'medium', 'low'
    title: str
    description: str

    # Location in document
    section: Optional[str] = None
    page_number: Optional[int] = None
    line_number: Optional[int] = None
    content_snippet: Optional[str] = None

    # Status and remediation
    status: str = 'open'  # 'open', 'acknowledged', 'in_progress', 'resolved', 'accepted_risk'
    assigned_to: Optional[str] = None
    remediation_plan: Optional[str] = None
    remediation_deadline: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'violation_id': self.violation_id,
            'assessment_id': self.assessment_id,
            'doc_id': self.doc_id,
            'requirement_id': self.requirement_id,
            'violation_type': self.violation_type,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'section': self.section,
            'page_number': self.page_number,
            'line_number': self.line_number,
            'content_snippet': self.content_snippet,
            'status': self.status,
            'assigned_to': self.assigned_to,
            'remediation_plan': self.remediation_plan,
            'remediation_deadline': self.remediation_deadline.isoformat() if self.remediation_deadline else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class ComplianceReport:
    """Compliance report for a document or system"""

    report_id: str
    report_type: str  # 'document', 'system', 'periodic', 'ad_hoc'
    title: str
    description: str

    # Scope
    scope_type: str  # 'single_document', 'document_type', 'system', 'all'
    scope_value: str  # Document ID, document type, system name, or 'all'

    # Report data
    generated_at: datetime = field(default_factory=datetime.now)
    generated_by: str = "system"
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    # Summary statistics
    total_documents_assessed: int = 0
    compliant_documents: int = 0
    non_compliant_documents: int = 0
    partial_compliance_documents: int = 0

    # Risk summary
    critical_violations: int = 0
    high_risk_violations: int = 0
    medium_risk_violations: int = 0
    low_risk_violations: int = 0

    # Details
    assessments: List[str] = field(default_factory=list)  # Assessment IDs
    violations: List[str] = field(default_factory=list)  # Violation IDs
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type,
            'title': self.title,
            'description': self.description,
            'scope_type': self.scope_type,
            'scope_value': self.scope_value,
            'generated_at': self.generated_at.isoformat(),
            'generated_by': self.generated_by,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
            'total_documents_assessed': self.total_documents_assessed,
            'compliant_documents': self.compliant_documents,
            'non_compliant_documents': self.non_compliant_documents,
            'partial_compliance_documents': self.partial_compliance_documents,
            'critical_violations': self.critical_violations,
            'high_risk_violations': self.high_risk_violations,
            'medium_risk_violations': self.medium_risk_violations,
            'low_risk_violations': self.low_risk_violations,
            'assessments': self.assessments,
            'violations': self.violations,
            'recommendations': self.recommendations
        }


@dataclass
class ComplianceAlert:
    """Alert for compliance issues"""

    alert_id: str
    alert_type: str  # 'violation_detected', 'deadline_approaching', 'status_change', 'system_issue'
    severity: str  # 'info', 'warning', 'error', 'critical'

    # Alert details
    title: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    # Related entities
    related_doc_id: Optional[str] = None
    related_requirement_id: Optional[str] = None
    related_violation_id: Optional[str] = None
    related_assessment_id: Optional[str] = None

    # Status and handling
    status: str = 'active'  # 'active', 'acknowledged', 'resolved', 'dismissed'
    assigned_to: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'title': self.title,
            'message': self.message,
            'details': self.details,
            'related_doc_id': self.related_doc_id,
            'related_requirement_id': self.related_requirement_id,
            'related_violation_id': self.related_violation_id,
            'related_assessment_id': self.related_assessment_id,
            'status': self.status,
            'assigned_to': self.assigned_to,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class ComplianceAnalytics:
    """Analytics data for compliance trends and insights"""

    analytics_id: str
    period_start: datetime
    period_end: datetime

    # Document compliance trends
    total_documents: int = 0
    assessed_documents: int = 0
    compliance_rate: float = 0.0
    avg_compliance_score: float = 0.0

    # Violation trends
    total_violations: int = 0
    new_violations: int = 0
    resolved_violations: int = 0
    overdue_violations: int = 0

    # Risk distribution
    risk_distribution: Dict[str, int] = field(default_factory=dict)

    # Top violation types
    top_violation_types: List[Tuple[str, int]] = field(default_factory=list)

    # Compliance by category
    compliance_by_category: Dict[str, float] = field(default_factory=dict)
    compliance_by_system: Dict[str, float] = field(default_factory=dict)

    # Trends over time
    daily_compliance_scores: List[Tuple[str, float]] = field(default_factory=list)
    weekly_violation_counts: List[Tuple[str, int]] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'analytics_id': self.analytics_id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_documents': self.total_documents,
            'assessed_documents': self.assessed_documents,
            'compliance_rate': self.compliance_rate,
            'avg_compliance_score': self.avg_compliance_score,
            'total_violations': self.total_violations,
            'new_violations': self.new_violations,
            'resolved_violations': self.resolved_violations,
            'overdue_violations': self.overdue_violations,
            'risk_distribution': self.risk_distribution,
            'top_violation_types': self.top_violation_types,
            'compliance_by_category': self.compliance_by_category,
            'compliance_by_system': self.compliance_by_system,
            'daily_compliance_scores': self.daily_compliance_scores,
            'weekly_violation_counts': self.weekly_violation_counts,
            'generated_at': self.generated_at.isoformat()
        }


class ComplianceDataManager:
    """Manages compliance data storage and retrieval"""

    def __init__(self, data_dir: str = "data/compliance"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths for different data types
        self.requirements_file = self.data_dir / "requirements.json"
        self.assessments_file = self.data_dir / "assessments.json"
        self.violations_file = self.data_dir / "violations.json"
        self.reports_file = self.data_dir / "reports.json"
        self.alerts_file = self.data_dir / "alerts.json"
        self.analytics_file = self.data_dir / "analytics.json"

        # In-memory caches
        self._requirements_cache: Dict[str, ComplianceRequirement] = {}
        self._assessments_cache: Dict[str, ComplianceAssessment] = {}
        self._violations_cache: Dict[str, ComplianceViolation] = {}

    def generate_id(self, prefix: str = "comp") -> str:
        """Generate a unique ID for compliance entities"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def save_requirements(self, requirements: List[ComplianceRequirement]) -> bool:
        """Save compliance requirements to file"""
        try:
            data = [req.to_dict() for req in requirements]
            with open(self.requirements_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Update cache
            self._requirements_cache = {req.requirement_id: req for req in requirements}
            return True
        except Exception as e:
            print(f"Error saving requirements: {e}")
            return False

    def load_requirements(self) -> List[ComplianceRequirement]:
        """Load compliance requirements from file"""
        try:
            if not self.requirements_file.exists():
                return []

            with open(self.requirements_file, 'r') as f:
                data = json.load(f)

            requirements = []
            for item in data:
                req = ComplianceRequirement(
                    requirement_id=item['requirement_id'],
                    title=item['title'],
                    description=item['description'],
                    category=item['category'],
                    subcategory=item['subcategory'],
                    severity=item['severity'],
                    applicable_systems=item.get('applicable_systems', []),
                    applicable_document_types=item.get('applicable_document_types', []),
                    keywords=item.get('keywords', []),
                    must_contain_keywords=item.get('must_contain_keywords', []),
                    must_not_contain_keywords=item.get('must_not_contain_keywords', []),
                    required_sections=item.get('required_sections', []),
                    prohibited_content=item.get('prohibited_content', []),
                    effective_date=datetime.fromisoformat(item['effective_date']) if item.get('effective_date') else None,
                    expiry_date=datetime.fromisoformat(item['expiry_date']) if item.get('expiry_date') else None,
                    last_updated=datetime.fromisoformat(item['last_updated']),
                    created_by=item.get('created_by', ''),
                    is_active=item.get('is_active', True),
                    version=item.get('version', '1.0')
                )
                requirements.append(req)

            # Update cache
            self._requirements_cache = {req.requirement_id: req for req in requirements}
            return requirements

        except Exception as e:
            print(f"Error loading requirements: {e}")
            return []

    def get_requirement(self, requirement_id: str) -> Optional[ComplianceRequirement]:
        """Get a specific requirement by ID"""
        if requirement_id in self._requirements_cache:
            return self._requirements_cache[requirement_id]

        # Load from file if not in cache
        requirements = self.load_requirements()
        return self._requirements_cache.get(requirement_id)

    def save_assessment(self, assessment: ComplianceAssessment) -> bool:
        """Save a compliance assessment"""
        try:
            # Load existing assessments
            assessments = self.load_assessments()
            assessments.append(assessment)

            # Save to file
            data = [assess.to_dict() for assess in assessments]
            with open(self.assessments_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Update cache
            self._assessments_cache[assessment.assessment_id] = assessment
            return True

        except Exception as e:
            print(f"Error saving assessment: {e}")
            return False

    def load_assessments(self, doc_id: str = None) -> List[ComplianceAssessment]:
        """Load compliance assessments, optionally filtered by document ID"""
        try:
            if not self.assessments_file.exists():
                return []

            with open(self.assessments_file, 'r') as f:
                data = json.load(f)

            assessments = []
            for item in data:
                assessment = ComplianceAssessment(
                    assessment_id=item['assessment_id'],
                    doc_id=item['doc_id'],
                    requirement_id=item['requirement_id'],
                    overall_status=item['overall_status'],
                    compliance_score=item['compliance_score'],
                    risk_level=item['risk_level'],
                    violations=item.get('violations', []),
                    warnings=item.get('warnings', []),
                    recommendations=item.get('recommendations', []),
                    assessed_at=datetime.fromisoformat(item['assessed_at']),
                    assessed_by=item.get('assessed_by', 'system'),
                    assessment_version=item.get('assessment_version', '1.0'),
                    document_version=item.get('document_version'),
                    document_sections_analyzed=item.get('document_sections_analyzed', [])
                )
                assessments.append(assessment)

            # Filter by document ID if specified
            if doc_id:
                assessments = [a for a in assessments if a.doc_id == doc_id]

            # Update cache
            self._assessments_cache = {a.assessment_id: a for a in assessments}
            return assessments

        except Exception as e:
            print(f"Error loading assessments: {e}")
            return []

    def save_violation(self, violation: ComplianceViolation) -> bool:
        """Save a compliance violation"""
        try:
            # Load existing violations
            violations = self.load_violations()
            violations.append(violation)

            # Save to file
            data = [viol.to_dict() for viol in violations]
            with open(self.violations_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Update cache
            self._violations_cache[violation.violation_id] = violation
            return True

        except Exception as e:
            print(f"Error saving violation: {e}")
            return False

    def load_violations(self, doc_id: str = None, status: str = None) -> List[ComplianceViolation]:
        """Load compliance violations, optionally filtered"""
        try:
            if not self.violations_file.exists():
                return []

            with open(self.violations_file, 'r') as f:
                data = json.load(f)

            violations = []
            for item in data:
                violation = ComplianceViolation(
                    violation_id=item['violation_id'],
                    assessment_id=item['assessment_id'],
                    doc_id=item['doc_id'],
                    requirement_id=item['requirement_id'],
                    violation_type=item['violation_type'],
                    severity=item['severity'],
                    title=item['title'],
                    description=item['description'],
                    section=item.get('section'),
                    page_number=item.get('page_number'),
                    line_number=item.get('line_number'),
                    content_snippet=item.get('content_snippet'),
                    status=item.get('status', 'open'),
                    assigned_to=item.get('assigned_to'),
                    remediation_plan=item.get('remediation_plan'),
                    remediation_deadline=datetime.fromisoformat(item['remediation_deadline']) if item.get('remediation_deadline') else None,
                    resolved_at=datetime.fromisoformat(item['resolved_at']) if item.get('resolved_at') else None,
                    resolved_by=item.get('resolved_by'),
                    created_at=datetime.fromisoformat(item['created_at']),
                    updated_at=datetime.fromisoformat(item['updated_at'])
                )
                violations.append(violation)

            # Apply filters
            if doc_id:
                violations = [v for v in violations if v.doc_id == doc_id]
            if status:
                violations = [v for v in violations if v.status == status]

            # Update cache
            self._violations_cache = {v.violation_id: v for v in violations}
            return violations

        except Exception as e:
            print(f"Error loading violations: {e}")
            return []