"""
Configuration for the Compliance Monitoring System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
try:
    from .compliance_models import ComplianceRequirement
except ImportError:
    from compliance_models import ComplianceRequirement


@dataclass
class ComplianceConfig:
    """Configuration for compliance monitoring"""

    # System settings
    enable_real_time_monitoring: bool = True
    enable_automated_reporting: bool = True
    enable_alert_notifications: bool = True

    # Scanning settings
    scan_interval_hours: int = 24
    scan_batch_size: int = 100
    enable_incremental_scanning: bool = True

    # Assessment settings
    default_assessment_timeout: int = 300  # seconds
    max_content_size_mb: int = 50
    enable_parallel_assessment: bool = True

    # Alert settings
    alert_cooldown_minutes: int = 60
    max_alerts_per_day: int = 1000
    critical_alert_recipients: List[str] = field(default_factory=list)

    # Reporting settings
    daily_report_time: str = "08:00"
    weekly_report_day: str = "monday"
    monthly_report_date: int = 1

    # Retention settings
    assessment_retention_days: int = 2555  # 7 years
    violation_retention_days: int = 2555
    report_retention_days: int = 2555
    alert_retention_days: int = 365

    # Risk thresholds
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'critical': 95.0,
        'high': 85.0,
        'medium': 70.0,
        'low': 50.0
    })

    # Integration settings
    integrate_with_version_control: bool = True
    integrate_with_search_engine: bool = True
    integrate_with_approval_workflow: bool = True


@dataclass
class ComplianceRule:
    """Individual compliance rule configuration"""

    rule_id: str
    name: str
    description: str
    rule_type: str  # 'content', 'structure', 'metadata', 'format'

    # Rule parameters
    enabled: bool = True
    severity: str = 'medium'
    weight: float = 1.0

    # Rule logic
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)

    # Applicability
    document_types: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    def to_requirement(self) -> ComplianceRequirement:
        """Convert rule to compliance requirement"""
        return ComplianceRequirement(
            requirement_id=self.rule_id,
            title=self.name,
            description=self.description,
            category='policy',
            subcategory=self.rule_type,
            severity=self.severity,
            applicable_document_types=self.document_types,
            keywords=self.conditions.get('keywords', []),
            must_contain_keywords=self.conditions.get('must_contain', []),
            must_not_contain_keywords=self.conditions.get('must_not_contain', []),
            required_sections=self.conditions.get('required_sections', []),
            prohibited_content=self.conditions.get('prohibited_content', [])
        )


class ComplianceRequirementsBuilder:
    """Builder for creating default compliance requirements"""

    @staticmethod
    def build_default_requirements() -> List[ComplianceRequirement]:
        """Build default compliance requirements for SCM documents"""

        requirements = []

        # 1. Document Version Control Requirements
        req1 = ComplianceRequirement(
            requirement_id="REQ_VERSION_CONTROL",
            title="Document Version Control",
            description="All documents must have proper version control information",
            category="procedural",
            subcategory="version_management",
            severity="high",
            applicable_document_types=["Procedure", "Policy", "Guide", "Template"],
            keywords=["version", "revision", "updated", "modified"],
            must_contain_keywords=["version"],
            required_sections=["Version History", "Change Log", "Revision History"]
        )
        requirements.append(req1)

        # 2. Approval Requirements
        req2 = ComplianceRequirement(
            requirement_id="REQ_APPROVAL_PROCESS",
            title="Approval Process Documentation",
            description="Documents must document approval processes and authorities",
            category="procedural",
            subcategory="approval_workflow",
            severity="high",
            applicable_document_types=["Procedure", "Policy"],
            keywords=["approval", "authorize", "sign off", "review"],
            must_contain_keywords=["approval", "approved by"],
            required_sections=["Approval Process", "Approval Authority", "Review Process"]
        )
        requirements.append(req2)

        # 3. Document Metadata Requirements
        req3 = ComplianceRequirement(
            requirement_id="REQ_DOCUMENT_METADATA",
            title="Complete Document Metadata",
            description="All documents must have complete metadata including author, date, and purpose",
            category="procedural",
            subcategory="documentation_standards",
            severity="medium",
            applicable_document_types=["Procedure", "Guide", "Policy", "Template"],
            keywords=["author", "created", "purpose", "scope"],
            must_contain_keywords=["author", "date"],
            required_sections=["Document Information", "Metadata", "Document Details"]
        )
        requirements.append(req3)

        # 4. Regulatory Compliance Requirements
        req4 = ComplianceRequirement(
            requirement_id="REQ_REGULATORY_COMPLIANCE",
            title="Regulatory Compliance References",
            description="Documents must reference applicable regulatory requirements where relevant",
            category="regulatory",
            subcategory="compliance_obligations",
            severity="critical",
            applicable_document_types=["Policy", "Procedure"],
            applicable_systems=["Oracle Cloud", "Oracle EBS", "SAP"],
            keywords=["regulation", "compliance", "legal", "requirement"],
            must_contain_keywords=["compliance", "regulatory"]
        )
        requirements.append(req4)

        # 5. Security Requirements
        req5 = ComplianceRequirement(
            requirement_id="REQ_SECURITY_CONTROLS",
            title="Security Control Documentation",
            description="Security-sensitive documents must document access controls and security measures",
            category="security",
            subcategory="access_control",
            severity="critical",
            applicable_document_types=["Policy", "Procedure"],
            keywords=["security", "access", "control", "confidential"],
            must_contain_keywords=["security", "access control"],
            required_sections=["Security Considerations", "Access Requirements"]
        )
        requirements.append(req5)

        # 6. Training Requirements
        req6 = ComplianceRequirement(
            requirement_id="REQ_TRAINING_REQUIREMENTS",
            title="Training and Competency Requirements",
            description="Documents must specify training and competency requirements for procedures",
            category="procedural",
            subcategory="training_competency",
            severity="medium",
            applicable_document_types=["Procedure", "Guide"],
            keywords=["training", "competency", "skill", "qualification"],
            must_contain_keywords=["training", "competent"]
        )
        requirements.append(req6)

        # 7. Exception Handling Requirements
        req7 = ComplianceRequirement(
            requirement_id="REQ_EXCEPTION_HANDLING",
            title="Exception Handling Procedures",
            description="Documents must include procedures for handling exceptions and deviations",
            category="procedural",
            subcategory="exception_management",
            severity="high",
            applicable_document_types=["Procedure", "Policy"],
            keywords=["exception", "deviation", "non-compliance", "waiver"],
            must_contain_keywords=["exception", "deviation"],
            required_sections=["Exception Handling", "Deviation Process"]
        )
        requirements.append(req7)

        # 8. Record Keeping Requirements
        req8 = ComplianceRequirement(
            requirement_id="REQ_RECORD_KEEPING",
            title="Record Keeping and Documentation",
            description="Documents must specify record keeping requirements and retention periods",
            category="procedural",
            subcategory="record_management",
            severity="medium",
            applicable_document_types=["Procedure", "Policy"],
            keywords=["record", "document", "retain", "archive"],
            must_contain_keywords=["record", "documentation"],
            required_sections=["Record Keeping", "Documentation Requirements"]
        )
        requirements.append(req8)

        # 9. Performance Monitoring Requirements
        req9 = ComplianceRequirement(
            requirement_id="REQ_PERFORMANCE_MONITORING",
            title="Performance Monitoring and Metrics",
            description="Documents must include performance monitoring and measurement requirements",
            category="procedural",
            subcategory="performance_management",
            severity="medium",
            applicable_document_types=["Procedure", "Policy"],
            keywords=["monitor", "measure", "metric", "performance"],
            must_contain_keywords=["monitor", "performance"]
        )
        requirements.append(req9)

        # 10. Review and Update Requirements
        req10 = ComplianceRequirement(
            requirement_id="REQ_REVIEW_UPDATE",
            title="Periodic Review and Update Process",
            description="Documents must specify periodic review and update processes",
            category="procedural",
            subcategory="document_lifecycle",
            severity="medium",
            applicable_document_types=["Procedure", "Policy", "Guide"],
            keywords=["review", "update", "periodic", "annual"],
            must_contain_keywords=["review", "update"],
            required_sections=["Review Process", "Update Frequency"]
        )
        requirements.append(req10)

        # 11. System Integration Requirements
        req11 = ComplianceRequirement(
            requirement_id="REQ_SYSTEM_INTEGRATION",
            title="System Integration Documentation",
            description="Documents referencing systems must include integration requirements",
            category="procedural",
            subcategory="system_integration",
            severity="high",
            applicable_systems=["Oracle Cloud", "Oracle EBS", "SAP", "ServiceNow", "Maximo"],
            keywords=["system", "integration", "interface", "connect"],
            must_contain_keywords=["system", "integration"]
        )
        requirements.append(req11)

        # 12. Change Management Requirements
        req12 = ComplianceRequirement(
            requirement_id="REQ_CHANGE_MANAGEMENT",
            title="Change Management Process",
            description="Documents must reference change management processes for modifications",
            category="procedural",
            subcategory="change_control",
            severity="high",
            applicable_document_types=["Procedure", "Policy"],
            keywords=["change", "modify", "amend", "revision"],
            must_contain_keywords=["change management", "change control"]
        )
        requirements.append(req12)

        return requirements

    @staticmethod
    def build_scm_specific_requirements() -> List[ComplianceRequirement]:
        """Build SCM-specific compliance requirements"""

        requirements = []

        # SCM Policy Requirements
        req1 = ComplianceRequirement(
            requirement_id="REQ_SCM_POLICY_LEVELS",
            title="SCM Policy Structure Compliance",
            description="SCM documents must follow the 5-level policy structure",
            category="policy",
            subcategory="policy_structure",
            severity="critical",
            applicable_document_types=["Policy", "Procedure"],
            keywords=["level 1", "level 2", "level 3", "level 4", "level 5"],
            must_contain_keywords=["level", "policy"],
            required_sections=["Policy Level", "Governance Level"]
        )
        requirements.append(req1)

        # Contract Management Requirements
        req2 = ComplianceRequirement(
            requirement_id="REQ_CONTRACT_MANAGEMENT",
            title="Contract Management Compliance",
            description="Contract management documents must include standard processes",
            category="procedural",
            subcategory="contract_management",
            severity="high",
            applicable_document_types=["Procedure"],
            keywords=["contract", "agreement", "supplier", "vendor"],
            must_contain_keywords=["contract management", "supplier agreement"],
            required_sections=["Contract Administration", "Supplier Management"]
        )
        requirements.append(req2)

        # Procurement Requirements
        req3 = ComplianceRequirement(
            requirement_id="REQ_PROCUREMENT_COMPLIANCE",
            title="Procurement Process Compliance",
            description="Procurement documents must comply with P2P processes",
            category="procedural",
            subcategory="procurement",
            severity="high",
            applicable_document_types=["Procedure"],
            keywords=["procurement", "purchase", "acquisition", "p2p"],
            must_contain_keywords=["procure to pay", "procurement"],
            required_sections=["Procurement Process", "P2P Process"]
        )
        requirements.append(req3)

        # Invoice Processing Requirements
        req4 = ComplianceRequirement(
            requirement_id="REQ_INVOICE_PROCESSING",
            title="Invoice Processing Compliance",
            description="Invoice processing documents must include standard procedures",
            category="procedural",
            subcategory="invoice_processing",
            severity="critical",
            applicable_systems=["Oracle Cloud", "Oracle EBS"],
            keywords=["invoice", "payment", "processing", "accounts payable"],
            must_contain_keywords=["invoice processing", "payment processing"],
            required_sections=["Invoice Requirements", "Payment Terms"]
        )
        requirements.append(req4)

        # Vendor Management Requirements
        req5 = ComplianceRequirement(
            requirement_id="REQ_VENDOR_MANAGEMENT",
            title="Vendor Management Compliance",
            description="Vendor management documents must include supplier relationship processes",
            category="procedural",
            subcategory="vendor_management",
            severity="high",
            applicable_document_types=["Procedure"],
            keywords=["vendor", "supplier", "relationship", "management"],
            must_contain_keywords=["supplier relationship", "vendor management"],
            required_sections=["Supplier Management", "Vendor Relationship"]
        )
        requirements.append(req5)

        return requirements


# Global configuration instance
compliance_config = ComplianceConfig()

# Build default requirements
default_requirements = ComplianceRequirementsBuilder.build_default_requirements()
scm_requirements = ComplianceRequirementsBuilder.build_scm_specific_requirements()
all_requirements = default_requirements + scm_requirements