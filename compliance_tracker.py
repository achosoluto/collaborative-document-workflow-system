"""
Regulatory Requirement Mapping and Tracking System
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

try:
    from .compliance_models import (
        ComplianceRequirement, ComplianceAssessment, ComplianceViolation,
        ComplianceDataManager
    )
    from .compliance_config import all_requirements
except ImportError:
    from compliance_models import (
        ComplianceRequirement, ComplianceAssessment, ComplianceViolation,
        ComplianceDataManager
    )
    from compliance_config import all_requirements

logger = logging.getLogger(__name__)


@dataclass
class RegulatoryReference:
    """Represents a regulatory reference or standard"""

    reference_id: str
    title: str
    description: str
    category: str  # 'federal', 'state', 'industry', 'internal'
    jurisdiction: str  # 'US', 'Canada', 'EU', 'Global', etc.
    document_type: str  # 'law', 'regulation', 'standard', 'guideline'

    # Reference details
    reference_number: str
    effective_date: datetime
    version: str

    # Content
    sections: List[str] = None
    keywords: List[str] = None
    requirements: List[str] = None

    # Status
    is_active: bool = True
    last_reviewed: Optional[datetime] = None

    def __post_init__(self):
        if self.sections is None:
            self.sections = []
        if self.keywords is None:
            self.keywords = []
        if self.requirements is None:
            self.requirements = []


@dataclass
class RequirementMapping:
    """Maps regulatory requirements to internal documents"""

    mapping_id: str
    regulatory_reference_id: str
    internal_requirement_id: str

    # Mapping details
    relevance_score: float  # 0.0 to 1.0
    mapping_type: str  # 'direct', 'partial', 'reference'
    confidence_level: str  # 'high', 'medium', 'low'

    # Evidence
    evidence_sections: List[str] = None
    evidence_quotes: List[str] = None
    gap_analysis: Optional[str] = None

    # Status
    is_active: bool = True
    created_at: datetime = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.evidence_sections is None:
            self.evidence_sections = []
        if self.evidence_quotes is None:
            self.evidence_quotes = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()


class RegulatoryRequirementManager:
    """Manages regulatory requirements and their mappings"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()
        self.regulatory_references_file = Path("data/compliance/regulatory_references.json")
        self.mappings_file = Path("data/compliance/requirement_mappings.json")

        # Ensure data directory exists
        self.regulatory_references_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or create default regulatory references
        self.regulatory_references = self._load_regulatory_references()
        self.mappings = self._load_mappings()

    def _load_regulatory_references(self) -> List[RegulatoryReference]:
        """Load regulatory references from file or create defaults"""
        try:
            if self.regulatory_references_file.exists():
                with open(self.regulatory_references_file, 'r') as f:
                    data = json.load(f)

                references = []
                for item in data:
                    ref = RegulatoryReference(
                        reference_id=item['reference_id'],
                        title=item['title'],
                        description=item['description'],
                        category=item['category'],
                        jurisdiction=item['jurisdiction'],
                        document_type=item['document_type'],
                        reference_number=item['reference_number'],
                        effective_date=datetime.fromisoformat(item['effective_date']),
                        version=item['version'],
                        sections=item.get('sections', []),
                        keywords=item.get('keywords', []),
                        requirements=item.get('requirements', []),
                        is_active=item.get('is_active', True),
                        last_reviewed=datetime.fromisoformat(item['last_reviewed']) if item.get('last_reviewed') else None
                    )
                    references.append(ref)

                return references

        except Exception as e:
            logger.error(f"Error loading regulatory references: {e}")

        # Return default references if loading fails
        return self._create_default_regulatory_references()

    def _create_default_regulatory_references(self) -> List[RegulatoryReference]:
        """Create default regulatory references for SCM"""
        references = []

        # SOX Compliance
        sox = RegulatoryReference(
            reference_id="REG_SOX_2002",
            title="Sarbanes-Oxley Act of 2002",
            description="US federal law establishing auditing and financial regulations for public companies",
            category="federal",
            jurisdiction="US",
            document_type="law",
            reference_number="Pub.L. 107-204",
            effective_date=datetime(2002, 7, 30),
            version="1.0",
            sections=["Section 302", "Section 404", "Section 409"],
            keywords=["internal controls", "financial reporting", "audit", "disclosure"],
            requirements=[
                "Maintain adequate internal controls",
                "Ensure accurate financial reporting",
                "Provide timely disclosure of material events"
            ]
        )
        references.append(sox)

        # Canadian Procurement Standards
        cps = RegulatoryReference(
            reference_id="REG_CPS_2023",
            title="Canadian Federal Procurement Standards",
            description="Standards for federal procurement in Canada",
            category="federal",
            jurisdiction="Canada",
            document_type="standard",
            reference_number="TBS-PSPC-2023",
            effective_date=datetime(2023, 1, 1),
            version="1.0",
            sections=["Competitive Bidding", "Contract Management", "Supplier Relations"],
            keywords=["procurement", "competitive bidding", "contract management", "supplier"],
            requirements=[
                "Ensure fair and transparent procurement processes",
                "Maintain proper contract documentation",
                "Monitor supplier performance"
            ]
        )
        references.append(cps)

        # Data Protection Standards
        gdpr = RegulatoryReference(
            reference_id="REG_GDPR_2018",
            title="General Data Protection Regulation",
            description="EU regulation on data protection and privacy",
            category="industry",
            jurisdiction="EU",
            document_type="regulation",
            reference_number="2016/679",
            effective_date=datetime(2018, 5, 25),
            version="1.0",
            sections=["Article 25", "Article 32", "Article 33"],
            keywords=["data protection", "privacy", "security", "breach notification"],
            requirements=[
                "Implement data protection by design",
                "Ensure data security",
                "Report data breaches within 72 hours"
            ]
        )
        references.append(gdpr)

        # Internal SCM Standards
        scm_standards = RegulatoryReference(
            reference_id="REG_SCM_STANDARDS_2024",
            title="Internal SCM Governance Standards",
            description="Internal standards for Supply Chain Management governance",
            category="internal",
            jurisdiction="Global",
            document_type="standard",
            reference_number="SCM-GOV-2024",
            effective_date=datetime(2024, 1, 1),
            version="2.0",
            sections=["Level 1-5 Structure", "Document Control", "Approval Processes"],
            keywords=["governance", "policy", "procedure", "level", "approval"],
            requirements=[
                "Follow 5-level policy structure",
                "Maintain document version control",
                "Document approval processes"
            ]
        )
        references.append(scm_standards)

        return references

    def _load_mappings(self) -> List[RequirementMapping]:
        """Load requirement mappings from file"""
        try:
            if self.mappings_file.exists():
                with open(self.mappings_file, 'r') as f:
                    data = json.load(f)

                mappings = []
                for item in data:
                    mapping = RequirementMapping(
                        mapping_id=item['mapping_id'],
                        regulatory_reference_id=item['regulatory_reference_id'],
                        internal_requirement_id=item['internal_requirement_id'],
                        relevance_score=item['relevance_score'],
                        mapping_type=item['mapping_type'],
                        confidence_level=item['confidence_level'],
                        evidence_sections=item.get('evidence_sections', []),
                        evidence_quotes=item.get('evidence_quotes', []),
                        gap_analysis=item.get('gap_analysis'),
                        is_active=item.get('is_active', True),
                        created_at=datetime.fromisoformat(item['created_at']),
                        last_updated=datetime.fromisoformat(item['last_updated'])
                    )
                    mappings.append(mapping)

                return mappings

        except Exception as e:
            logger.error(f"Error loading mappings: {e}")

        return []

    def save_regulatory_references(self) -> bool:
        """Save regulatory references to file"""
        try:
            data = []
            for ref in self.regulatory_references:
                ref_dict = {
                    'reference_id': ref.reference_id,
                    'title': ref.title,
                    'description': ref.description,
                    'category': ref.category,
                    'jurisdiction': ref.jurisdiction,
                    'document_type': ref.document_type,
                    'reference_number': ref.reference_number,
                    'effective_date': ref.effective_date.isoformat(),
                    'version': ref.version,
                    'sections': ref.sections,
                    'keywords': ref.keywords,
                    'requirements': ref.requirements,
                    'is_active': ref.is_active,
                    'last_reviewed': ref.last_reviewed.isoformat() if ref.last_reviewed else None
                }
                data.append(ref_dict)

            with open(self.regulatory_references_file, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving regulatory references: {e}")
            return False

    def save_mappings(self) -> bool:
        """Save requirement mappings to file"""
        try:
            data = []
            for mapping in self.mappings:
                mapping_dict = {
                    'mapping_id': mapping.mapping_id,
                    'regulatory_reference_id': mapping.regulatory_reference_id,
                    'internal_requirement_id': mapping.internal_requirement_id,
                    'relevance_score': mapping.relevance_score,
                    'mapping_type': mapping.mapping_type,
                    'confidence_level': mapping.confidence_level,
                    'evidence_sections': mapping.evidence_sections,
                    'evidence_quotes': mapping.evidence_quotes,
                    'gap_analysis': mapping.gap_analysis,
                    'is_active': mapping.is_active,
                    'created_at': mapping.created_at.isoformat(),
                    'last_updated': mapping.last_updated.isoformat()
                }
                data.append(mapping_dict)

            with open(self.mappings_file, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving mappings: {e}")
            return False

    def add_regulatory_reference(self, reference: RegulatoryReference) -> bool:
        """Add a new regulatory reference"""
        try:
            self.regulatory_references.append(reference)
            return self.save_regulatory_references()
        except Exception as e:
            logger.error(f"Error adding regulatory reference: {e}")
            return False

    def create_mapping(
        self,
        regulatory_ref_id: str,
        internal_req_id: str,
        mapping_type: str = "partial",
        confidence_level: str = "medium"
    ) -> Optional[RequirementMapping]:
        """Create a new requirement mapping"""
        try:
            mapping = RequirementMapping(
                mapping_id=f"map_{len(self.mappings) + 1}",
                regulatory_reference_id=regulatory_ref_id,
                internal_requirement_id=internal_req_id,
                relevance_score=0.8 if mapping_type == "direct" else 0.5,
                mapping_type=mapping_type,
                confidence_level=confidence_level
            )

            self.mappings.append(mapping)
            self.save_mappings()
            return mapping

        except Exception as e:
            logger.error(f"Error creating mapping: {e}")
            return None

    def get_compliance_status(self, requirement_id: str) -> Dict[str, Any]:
        """Get compliance status for a specific requirement"""
        try:
            # Load assessments for this requirement
            assessments = self.data_manager.load_assessments()

            # Filter assessments for this requirement
            req_assessments = [a for a in assessments if a.requirement_id == requirement_id]

            if not req_assessments:
                return {
                    'requirement_id': requirement_id,
                    'status': 'not_assessed',
                    'message': 'No assessments found for this requirement'
                }

            # Calculate compliance statistics
            total_assessments = len(req_assessments)
            compliant = len([a for a in req_assessments if a.overall_status == 'compliant'])
            non_compliant = len([a for a in req_assessments if a.overall_status == 'non_compliant'])
            partial = len([a for a in req_assessments if a.overall_status == 'partial'])

            compliance_rate = (compliant / total_assessments) * 100 if total_assessments > 0 else 0

            # Get latest assessment
            latest_assessment = max(req_assessments, key=lambda a: a.assessed_at)

            return {
                'requirement_id': requirement_id,
                'total_assessments': total_assessments,
                'compliance_rate': compliance_rate,
                'compliant_documents': compliant,
                'non_compliant_documents': non_compliant,
                'partial_compliance_documents': partial,
                'latest_assessment': latest_assessment.to_dict(),
                'trend': self._calculate_trend(req_assessments)
            }

        except Exception as e:
            logger.error(f"Error getting compliance status for {requirement_id}: {e}")
            return {
                'requirement_id': requirement_id,
                'status': 'error',
                'message': str(e)
            }

    def _calculate_trend(self, assessments: List[ComplianceAssessment]) -> str:
        """Calculate compliance trend over time"""
        if len(assessments) < 2:
            return 'insufficient_data'

        # Sort by assessment date
        sorted_assessments = sorted(assessments, key=lambda a: a.assessed_at)

        # Compare recent vs older assessments
        midpoint = len(sorted_assessments) // 2
        recent_assessments = sorted_assessments[midpoint:]
        older_assessments = sorted_assessments[:midpoint]

        recent_avg = sum(a.compliance_score for a in recent_assessments) / len(recent_assessments)
        older_avg = sum(a.compliance_score for a in older_assessments) / len(older_assessments)

        if recent_avg > older_avg + 5:
            return 'improving'
        elif recent_avg < older_avg - 5:
            return 'declining'
        else:
            return 'stable'

    def get_regulatory_coverage(self) -> Dict[str, Any]:
        """Get overall regulatory coverage statistics"""
        try:
            # Load all assessments
            assessments = self.data_manager.load_assessments()

            if not assessments:
                return {'status': 'no_data'}

            # Group by requirement
            requirement_stats = {}
            for assessment in assessments:
                req_id = assessment.requirement_id
                if req_id not in requirement_stats:
                    requirement_stats[req_id] = {
                        'total': 0,
                        'compliant': 0,
                        'non_compliant': 0,
                        'partial': 0,
                        'avg_score': 0.0
                    }

                requirement_stats[req_id]['total'] += 1
                if assessment.overall_status == 'compliant':
                    requirement_stats[req_id]['compliant'] += 1
                elif assessment.overall_status == 'non_compliant':
                    requirement_stats[req_id]['non_compliant'] += 1
                elif assessment.overall_status == 'partial':
                    requirement_stats[req_id]['partial'] += 1

                # Update average score
                current_avg = requirement_stats[req_id]['avg_score']
                current_total = requirement_stats[req_id]['total']
                requirement_stats[req_id]['avg_score'] = (
                    (current_avg * (current_total - 1)) + assessment.compliance_score
                ) / current_total

            # Calculate overall statistics
            total_requirements = len(requirement_stats)
            fully_compliant = len([r for r in requirement_stats.values() if r['compliant'] > 0])
            overall_compliance_rate = (fully_compliant / total_requirements * 100) if total_requirements > 0 else 0

            return {
                'total_requirements': total_requirements,
                'assessed_requirements': len(assessments),
                'fully_compliant_requirements': fully_compliant,
                'overall_compliance_rate': overall_compliance_rate,
                'requirement_details': requirement_stats,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating regulatory coverage: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_upcoming_deadlines(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """Get upcoming compliance deadlines"""
        deadlines = []

        try:
            # Check for requirements with expiry dates
            cutoff_date = datetime.now() + timedelta(days=days_ahead)

            for requirement in self.data_manager.load_requirements():
                if requirement.expiry_date and requirement.expiry_date <= cutoff_date:
                    # Get compliance status for this requirement
                    status = self.get_compliance_status(requirement.requirement_id)

                    deadline_info = {
                        'requirement_id': requirement.requirement_id,
                        'requirement_title': requirement.title,
                        'expiry_date': requirement.expiry_date.isoformat(),
                        'days_until_expiry': (requirement.expiry_date - datetime.now()).days,
                        'severity': requirement.severity,
                        'compliance_status': status.get('compliance_rate', 0),
                        'category': requirement.category
                    }
                    deadlines.append(deadline_info)

            # Sort by days until expiry
            deadlines.sort(key=lambda x: x['days_until_expiry'])

            return deadlines

        except Exception as e:
            logger.error(f"Error getting upcoming deadlines: {e}")
            return []

    def generate_gap_analysis(self, requirement_id: str) -> Dict[str, Any]:
        """Generate gap analysis for a specific requirement"""
        try:
            # Get requirement details
            requirement = None
            for req in self.data_manager.load_requirements():
                if req.requirement_id == requirement_id:
                    requirement = req
                    break

            if not requirement:
                return {'error': f'Requirement {requirement_id} not found'}

            # Get compliance status
            compliance_status = self.get_compliance_status(requirement_id)

            # Find related regulatory references
            related_mappings = [
                m for m in self.mappings
                if m.internal_requirement_id == requirement_id and m.is_active
            ]

            # Generate gap analysis
            gaps = []
            if compliance_status.get('compliance_rate', 0) < 80:
                gaps.append({
                    'type': 'compliance_gap',
                    'severity': 'high',
                    'description': f"Overall compliance rate is {compliance_status.get('compliance_rate', 0):.1f}%, below acceptable threshold",
                    'recommendation': 'Review and improve compliance across all assessed documents'
                })

            # Check for missing mappings
            if not related_mappings:
                gaps.append({
                    'type': 'mapping_gap',
                    'severity': 'medium',
                    'description': 'No regulatory references mapped to this requirement',
                    'recommendation': 'Map this requirement to relevant regulatory standards'
                })

            return {
                'requirement_id': requirement_id,
                'requirement_title': requirement.title,
                'current_compliance_rate': compliance_status.get('compliance_rate', 0),
                'gaps_identified': len(gaps),
                'gaps': gaps,
                'related_regulatory_references': [
                    mapping.regulatory_reference_id for mapping in related_mappings
                ],
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating gap analysis for {requirement_id}: {e}")
            return {'error': str(e)}


class ComplianceTracker:
    """Main interface for compliance tracking"""

    def __init__(self):
        self.requirement_manager = RegulatoryRequirementManager()

    def track_requirement(self, requirement_id: str) -> Dict[str, Any]:
        """Track compliance for a specific requirement"""
        return self.requirement_manager.get_compliance_status(requirement_id)

    def track_all_requirements(self) -> Dict[str, Any]:
        """Track compliance for all requirements"""
        return self.requirement_manager.get_regulatory_coverage()

    def get_deadlines(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """Get upcoming compliance deadlines"""
        return self.requirement_manager.get_upcoming_deadlines(days_ahead)

    def analyze_gaps(self, requirement_id: str) -> Dict[str, Any]:
        """Analyze gaps for a specific requirement"""
        return self.requirement_manager.generate_gap_analysis(requirement_id)

    def add_regulatory_reference(self, reference: RegulatoryReference) -> bool:
        """Add a new regulatory reference"""
        return self.requirement_manager.add_regulatory_reference(reference)

    def create_mapping(
        self,
        regulatory_ref_id: str,
        internal_req_id: str,
        mapping_type: str = "partial"
    ) -> bool:
        """Create a mapping between regulatory and internal requirements"""
        mapping = self.requirement_manager.create_mapping(
            regulatory_ref_id, internal_req_id, mapping_type
        )
        return mapping is not None