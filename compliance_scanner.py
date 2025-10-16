"""
Automated Compliance Scanning and Assessment Engine
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    from .compliance_models import (
        ComplianceRequirement, ComplianceAssessment, ComplianceViolation,
        ComplianceDataManager
    )
    from .compliance_config import compliance_config, all_requirements
except ImportError:
    from compliance_models import (
        ComplianceRequirement, ComplianceAssessment, ComplianceViolation,
        ComplianceDataManager
    )
    from compliance_config import compliance_config, all_requirements
try:
    from .document_processor import DocumentProcessor
    from .config import DocumentMetadata
except ImportError:
    from document_processor import DocumentProcessor
    from config import DocumentMetadata

logger = logging.getLogger(__name__)


class ComplianceScanner:
    """Automated compliance scanning engine"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()
        self.document_processor = DocumentProcessor()
        self.requirements = self._load_requirements()

    def _load_requirements(self) -> List[ComplianceRequirement]:
        """Load compliance requirements"""
        try:
            # Try to load from data manager first
            requirements = self.data_manager.load_requirements()
            if requirements:
                return requirements

            # Fall back to default requirements
            return all_requirements

        except Exception as e:
            logger.error(f"Error loading requirements: {e}")
            return all_requirements

    def scan_document(self, doc_id: str, file_path: str) -> List[ComplianceAssessment]:
        """
        Scan a single document for compliance against all applicable requirements

        Args:
            doc_id: Document identifier
            file_path: Path to the document file

        Returns:
            List of compliance assessments
        """
        assessments = []

        try:
            # Extract document content and metadata
            content, content_metadata = self.document_processor.extract_content(file_path)
            doc_metadata = self.document_processor.extract_document_metadata(file_path, doc_id)

            if not content.strip():
                logger.warning(f"No content extracted from document {doc_id}")
                return assessments

            # Get applicable requirements for this document
            applicable_requirements = self._get_applicable_requirements(doc_metadata, content)

            logger.info(f"Scanning document {doc_id} against {len(applicable_requirements)} requirements")

            # Assess compliance against each requirement
            for requirement in applicable_requirements:
                assessment = self._assess_compliance(
                    doc_id, requirement, content, doc_metadata, file_path
                )
                if assessment:
                    # Save assessment
                    self.data_manager.save_assessment(assessment)
                    assessments.append(assessment)

        except Exception as e:
            logger.error(f"Error scanning document {doc_id}: {e}")

        return assessments

    def _get_applicable_requirements(
        self,
        doc_metadata: DocumentMetadata,
        content: str
    ) -> List[ComplianceRequirement]:
        """Get requirements applicable to the document"""
        applicable = []

        for requirement in self.requirements:
            if not requirement.is_active:
                continue

            # Check if requirement is still effective
            if requirement.effective_date and requirement.effective_date > datetime.now():
                continue
            if requirement.expiry_date and requirement.expiry_date < datetime.now():
                continue

            # Check document type applicability
            if (requirement.applicable_document_types and
                doc_metadata.document_type not in requirement.applicable_document_types):
                continue

            # Check system applicability
            if (requirement.applicable_systems and
                not self._document_uses_systems(content, doc_metadata, requirement.applicable_systems)):
                continue

            # Check keyword relevance
            if requirement.keywords and not self._document_matches_keywords(content, requirement.keywords):
                continue

            applicable.append(requirement)

        return applicable

    def _document_uses_systems(
        self,
        content: str,
        doc_metadata: DocumentMetadata,
        systems: List[str]
    ) -> bool:
        """Check if document references any of the specified systems"""
        content_lower = (content + str(doc_metadata.__dict__)).lower()

        for system in systems:
            if system.lower() in content_lower:
                return True

        return False

    def _document_matches_keywords(self, content: str, keywords: List[str]) -> bool:
        """Check if document content matches any of the keywords"""
        content_lower = content.lower()

        for keyword in keywords:
            if keyword.lower() in content_lower:
                return True

        return False

    def _assess_compliance(
        self,
        doc_id: str,
        requirement: ComplianceRequirement,
        content: str,
        doc_metadata: DocumentMetadata,
        file_path: str
    ) -> Optional[ComplianceAssessment]:
        """Assess document compliance against a specific requirement"""
        try:
            assessment_id = self.data_manager.generate_id("assessment")

            # Initialize scoring
            total_checks = 0
            passed_checks = 0
            violations = []
            warnings = []
            recommendations = []

            # 1. Check must-contain keywords
            must_contain_violations = self._check_must_contain_keywords(
                content, requirement.must_contain_keywords, file_path
            )
            violations.extend(must_contain_violations)
            total_checks += len(requirement.must_contain_keywords)
            passed_checks += len(requirement.must_contain_keywords) - len(must_contain_violations)

            # 2. Check must-not-contain keywords
            must_not_contain_violations = self._check_must_not_contain_keywords(
                content, requirement.must_not_contain_keywords, file_path
            )
            violations.extend(must_not_contain_violations)
            total_checks += len(requirement.must_not_contain_keywords)
            passed_checks += len(requirement.must_not_contain_keywords) - len(must_not_contain_violations)

            # 3. Check required sections
            section_violations = self._check_required_sections(
                content, requirement.required_sections, file_path
            )
            violations.extend(section_violations)
            total_checks += len(requirement.required_sections)
            passed_checks += len(requirement.required_sections) - len(section_violations)

            # 4. Check prohibited content
            prohibited_violations = self._check_prohibited_content(
                content, requirement.prohibited_content, file_path
            )
            violations.extend(prohibited_violations)
            total_checks += len(requirement.prohibited_content)
            passed_checks += len(requirement.prohibited_content) - len(prohibited_violations)

            # 5. Check structural requirements
            structural_violations = self._check_structural_requirements(
                content, doc_metadata, requirement, file_path
            )
            violations.extend(structural_violations)
            total_checks += 1  # Structural check counts as one
            passed_checks += 0 if structural_violations else 1

            # Calculate compliance score
            compliance_score = (passed_checks / total_checks * 100) if total_checks > 0 else 100.0

            # Determine overall status and risk level
            overall_status, risk_level = self._determine_compliance_status(
                compliance_score, violations, requirement.severity
            )

            # Generate recommendations
            if violations:
                recommendations = self._generate_recommendations(violations, requirement)

            # Create assessment
            assessment = ComplianceAssessment(
                assessment_id=assessment_id,
                doc_id=doc_id,
                requirement_id=requirement.requirement_id,
                overall_status=overall_status,
                compliance_score=compliance_score,
                risk_level=risk_level,
                violations=[v.to_dict() for v in violations],
                warnings=[w.to_dict() for w in warnings],
                recommendations=recommendations,
                document_version=doc_metadata.version,
                document_sections_analyzed=self._extract_document_sections(content)
            )

            # Save violations
            for violation in violations:
                violation.assessment_id = assessment_id
                self.data_manager.save_violation(violation)

            return assessment

        except Exception as e:
            logger.error(f"Error assessing compliance for {doc_id} against {requirement.requirement_id}: {e}")
            return None

    def _check_must_contain_keywords(
        self,
        content: str,
        keywords: List[str],
        file_path: str
    ) -> List[ComplianceViolation]:
        """Check for required keywords in content"""
        violations = []
        content_lower = content.lower()

        for keyword in keywords:
            if keyword.lower() not in content_lower:
                violation = ComplianceViolation(
                    violation_id=self.data_manager.generate_id("violation"),
                    assessment_id="",  # Will be set later
                    doc_id="",  # Will be set later
                    requirement_id="",  # Will be set later
                    violation_type="missing_content",
                    severity="medium",
                    title=f"Missing Required Content: {keyword}",
                    description=f"The document must contain the keyword '{keyword}' but it was not found.",
                    content_snippet=f"Required keyword '{keyword}' is missing from the document content."
                )
                violations.append(violation)

        return violations

    def _check_must_not_contain_keywords(
        self,
        content: str,
        keywords: List[str],
        file_path: str
    ) -> List[ComplianceViolation]:
        """Check for prohibited keywords in content"""
        violations = []
        content_lower = content.lower()

        for keyword in keywords:
            if keyword.lower() in content_lower:
                violation = ComplianceViolation(
                    violation_id=self.data_manager.generate_id("violation"),
                    assessment_id="",  # Will be set later
                    doc_id="",  # Will be set later
                    requirement_id="",  # Will be set later
                    violation_type="prohibited_content",
                    severity="high",
                    title=f"Prohibited Content Found: {keyword}",
                    description=f"The document contains prohibited keyword '{keyword}'.",
                    content_snippet=f"Document contains prohibited keyword '{keyword}'."
                )
                violations.append(violation)

        return violations

    def _check_required_sections(
        self,
        content: str,
        sections: List[str],
        file_path: str
    ) -> List[ComplianceViolation]:
        """Check for required sections in content"""
        violations = []

        for section in sections:
            # Look for section headers (case-insensitive)
            section_pattern = rf'(?:^|\n)\s*{re.escape(section)}\s*(?:\n|:)'
            if not re.search(section_pattern, content, re.IGNORECASE | re.MULTILINE):
                violation = ComplianceViolation(
                    violation_id=self.data_manager.generate_id("violation"),
                    assessment_id="",  # Will be set later
                    doc_id="",  # Will be set later
                    requirement_id="",  # Will be set later
                    violation_type="structural_issue",
                    severity="medium",
                    title=f"Missing Required Section: {section}",
                    description=f"The document must contain a section titled '{section}'.",
                    content_snippet=f"Required section '{section}' is missing from the document structure."
                )
                violations.append(violation)

        return violations

    def _check_prohibited_content(
        self,
        content: str,
        prohibited: List[str],
        file_path: str
    ) -> List[ComplianceViolation]:
        """Check for prohibited content patterns"""
        violations = []

        for prohibited_item in prohibited:
            if prohibited_item.lower() in content.lower():
                violation = ComplianceViolation(
                    violation_id=self.data_manager.generate_id("violation"),
                    assessment_id="",  # Will be set later
                    doc_id="",  # Will be set later
                    requirement_id="",  # Will be set later
                    violation_type="prohibited_content",
                    severity="high",
                    title=f"Prohibited Content Found",
                    description=f"The document contains prohibited content: '{prohibited_item}'.",
                    content_snippet=f"Document contains prohibited content: '{prohibited_item}'."
                )
                violations.append(violation)

        return violations

    def _check_structural_requirements(
        self,
        content: str,
        doc_metadata: DocumentMetadata,
        requirement: ComplianceRequirement,
        file_path: str
    ) -> List[ComplianceViolation]:
        """Check structural requirements for the document"""
        violations = []

        # Check document has minimum length
        if len(content.strip()) < 100:
            violation = ComplianceViolation(
                violation_id=self.data_manager.generate_id("violation"),
                assessment_id="",  # Will be set later
                doc_id="",  # Will be set later
                requirement_id="",  # Will be set later
                violation_type="structural_issue",
                severity="medium",
                title="Document Too Short",
                description="The document appears to be too short to contain sufficient detail.",
                content_snippet="Document length is insufficient for compliance requirements."
            )
            violations.append(violation)

        # Check for basic document structure
        has_title = bool(doc_metadata.title and len(doc_metadata.title.strip()) > 5)
        has_sections = len(self._extract_document_sections(content)) > 0

        if not has_title:
            violation = ComplianceViolation(
                violation_id=self.data_manager.generate_id("violation"),
                assessment_id="",  # Will be set later
                doc_id="",  # Will be set later
                requirement_id="",  # Will be set later
                violation_type="structural_issue",
                severity="low",
                title="Missing Document Title",
                description="The document should have a clear, descriptive title.",
                content_snippet="Document title is missing or insufficient."
            )
            violations.append(violation)

        if not has_sections:
            violation = ComplianceViolation(
                violation_id=self.data_manager.generate_id("violation"),
                assessment_id="",  # Will be set later
                doc_id="",  # Will be set later
                requirement_id="",  # Will be set later
                violation_type="structural_issue",
                severity="low",
                title="Missing Document Structure",
                description="The document should be organized into clear sections.",
                content_snippet="Document lacks clear section structure."
            )
            violations.append(violation)

        return violations

    def _extract_document_sections(self, content: str) -> List[str]:
        """Extract section headers from document content"""
        sections = []

        # Look for common section patterns
        section_patterns = [
            r'^\s*(\d+\.|\-|\*)\s+(.+)$',  # Numbered or bulleted sections
            r'^\s*(SECTION|CHAPTER|PART)\s+\d+[:.]?\s+(.+)$',  # Formal sections
            r'^\s*(INTRODUCTION|OVERVIEW|SCOPE|PURPOSE|PROCEDURE|PROCESS|CONCLUSION)[:\s]',  # Common section names
        ]

        for pattern in section_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                section_title = match[1] if len(match) > 1 else match[0]
                sections.append(section_title.strip())

        return list(set(sections))  # Remove duplicates

    def _determine_compliance_status(
        self,
        score: float,
        violations: List[ComplianceViolation],
        requirement_severity: str
    ) -> Tuple[str, str]:
        """Determine overall compliance status and risk level"""
        # Determine status based on score
        if score >= 90.0:
            status = "compliant"
        elif score >= 70.0:
            status = "partial"
        else:
            status = "non_compliant"

        # Determine risk level based on violations and requirement severity
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]

        if critical_violations or (requirement_severity == "critical" and violations):
            risk_level = "critical"
        elif high_violations or (requirement_severity == "high" and violations):
            risk_level = "high"
        elif score < 50.0:
            risk_level = "high"
        elif score < 70.0:
            risk_level = "medium"
        else:
            risk_level = "low"

        return status, risk_level

    def _generate_recommendations(
        self,
        violations: List[ComplianceViolation],
        requirement: ComplianceRequirement
    ) -> List[str]:
        """Generate recommendations for addressing violations"""
        recommendations = []

        for violation in violations:
            if violation.violation_type == "missing_content":
                recommendations.append(
                    f"Add content related to '{violation.title}' to meet compliance requirements."
                )
            elif violation.violation_type == "prohibited_content":
                recommendations.append(
                    f"Remove or replace prohibited content: {violation.description}"
                )
            elif violation.violation_type == "structural_issue":
                recommendations.append(
                    f"Improve document structure: {violation.description}"
                )
            else:
                recommendations.append(
                    f"Address compliance issue: {violation.title}"
                )

        # Add general recommendations
        if requirement.severity in ["critical", "high"]:
            recommendations.append(
                f"This is a {requirement.severity} severity requirement. Expedited remediation is recommended."
            )

        return list(set(recommendations))  # Remove duplicates


class ComplianceScannerManager:
    """Manages compliance scanning operations"""

    def __init__(self):
        self.scanner = ComplianceScanner()
        self.is_scanning = False

    def scan_all_documents(self, base_path: str = None) -> Dict[str, Any]:
        """Scan all documents in the catalog"""
        results = {
            'total_scanned': 0,
            'total_assessments': 0,
            'errors': [],
            'start_time': datetime.now(),
            'end_time': None
        }

        try:
            # Load document catalog
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if not catalog_path.exists():
                results['errors'].append("Document catalog not found")
                return results

            import json
            with open(catalog_path, 'r') as f:
                catalog = json.load(f)

            # Scan each document
            for doc_id, doc_data in catalog.items():
                try:
                    file_path = doc_data.get('file_path')
                    if not file_path or not Path(file_path).exists():
                        continue

                    assessments = self.scanner.scan_document(doc_id, file_path)
                    results['total_scanned'] += 1
                    results['total_assessments'] += len(assessments)

                except Exception as e:
                    results['errors'].append(f"Error scanning {doc_id}: {str(e)}")

            results['end_time'] = datetime.now()
            return results

        except Exception as e:
            results['errors'].append(f"Error in scan_all_documents: {str(e)}")
            results['end_time'] = datetime.now()
            return results

    def scan_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Scan a specific document by ID"""
        try:
            # Find document in catalog
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if not catalog_path.exists():
                return {'error': 'Document catalog not found'}

            import json
            with open(catalog_path, 'r') as f:
                catalog = json.load(f)

            if doc_id not in catalog:
                return {'error': f'Document {doc_id} not found in catalog'}

            doc_data = catalog[doc_id]
            file_path = doc_data.get('file_path')

            if not file_path or not Path(file_path).exists():
                return {'error': f'Document file not found: {file_path}'}

            # Perform scan
            assessments = self.scanner.scan_document(doc_id, file_path)

            return {
                'success': True,
                'doc_id': doc_id,
                'assessments_count': len(assessments),
                'assessments': [a.to_dict() for a in assessments]
            }

        except Exception as e:
            return {'error': f'Error scanning document {doc_id}: {str(e)}'}

    def get_scan_status(self) -> Dict[str, Any]:
        """Get current scanning status"""
        return {
            'is_scanning': self.is_scanning,
            'scanner_ready': True
        }