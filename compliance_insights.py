"""
Compliance-Aware Insights System
Integrates with compliance systems to generate regulatory and compliance-focused insights
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict

# Custom imports
from .compliance_config import compliance_config, all_requirements, ComplianceRequirement
from .insight_extractor import insight_extractor, ExtractedInsight, InsightType, ConfidenceLevel
from .content_categorizer import content_categorizer
from .summarization_engine import summarization_engine, SummarizationConfig, SummarizationMethod, DetailLevel

logger = logging.getLogger(__name__)


@dataclass
class ComplianceInsight:
    """Represents a compliance-related insight"""
    insight_type: str  # 'requirement', 'violation', 'gap', 'recommendation'
    requirement_id: str
    requirement_title: str
    severity: str
    content: str
    evidence: List[str] = field(default_factory=list)
    location: Dict[str, Any] = field(default_factory=dict)
    compliance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'insight_type': self.insight_type,
            'requirement_id': self.requirement_id,
            'requirement_title': self.requirement_title,
            'severity': self.severity,
            'content': self.content,
            'evidence': self.evidence,
            'location': self.location,
            'compliance_score': self.compliance_score,
            'metadata': self.metadata
        }


@dataclass
class ComplianceAnalysis:
    """Complete compliance analysis for a document"""
    document_id: str
    file_path: str

    # Compliance results
    compliance_score: float = 0.0
    compliance_level: str = "unknown"  # compliant, non_compliant, partially_compliant
    compliance_insights: List[ComplianceInsight] = field(default_factory=list)

    # Requirement analysis
    requirements_met: List[str] = field(default_factory=list)
    requirements_violated: List[str] = field(default_factory=list)
    requirements_missing: List[str] = field(default_factory=list)

    # Risk assessment
    risk_level: str = "low"
    risk_factors: List[str] = field(default_factory=list)

    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    analysis_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'document_id': self.document_id,
            'file_path': self.file_path,
            'compliance_score': self.compliance_score,
            'compliance_level': self.compliance_level,
            'compliance_insights': [insight.to_dict() for insight in self.compliance_insights],
            'requirements_met': self.requirements_met,
            'requirements_violated': self.requirements_violated,
            'requirements_missing': self.requirements_missing,
            'risk_level': self.risk_level,
            'risk_factors': self.risk_factors,
            'analyzed_at': self.analyzed_at.isoformat(),
            'analysis_version': self.analysis_version
        }


class ComplianceInsightExtractor:
    """Extracts compliance-related insights from documents"""

    def __init__(self):
        self.requirements = all_requirements
        self.compliance_patterns = self._build_compliance_patterns()

    def _build_compliance_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for compliance requirement detection"""
        patterns = {}

        for req in self.requirements:
            req_id = req.requirement_id

            # Build patterns from requirement keywords
            must_contain = req.must_contain_keywords or []
            must_not_contain = req.must_not_contain_keywords or []

            patterns[req_id] = {
                'positive_patterns': [re.compile(f'\\b{keyword}\\b', re.IGNORECASE) for keyword in must_contain],
                'negative_patterns': [re.compile(f'\\b{keyword}\\b', re.IGNORECASE) for keyword in must_not_contain],
                'context_patterns': [re.compile(f'\\b{keyword}\\b', re.IGNORECASE) for keyword in req.keywords or []]
            }

        return patterns

    def extract_compliance_insights(self, file_path: str) -> Dict[str, Any]:
        """
        Extract compliance insights from a document

        Args:
            file_path: Path to document

        Returns:
            Compliance analysis results
        """
        try:
            # Get basic document information
            doc_id = self._generate_document_id(file_path)

            # Analyze document against all requirements
            analysis = ComplianceAnalysis(
                document_id=doc_id,
                file_path=file_path
            )

            # Read document content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return {
                    'success': False,
                    'error': 'Document is empty or unreadable'
                }

            # Analyze each requirement
            for requirement in self.requirements:
                req_insights = self._analyze_requirement_compliance(content, requirement)
                analysis.compliance_insights.extend(req_insights)

                # Update requirement status
                if req_insights:
                    insight = req_insights[0]  # Take first insight for status
                    if insight.insight_type == 'requirement':
                        analysis.requirements_met.append(requirement.requirement_id)
                    elif insight.insight_type == 'violation':
                        analysis.requirements_violated.append(requirement.requirement_id)
                    elif insight.insight_type == 'gap':
                        analysis.requirements_missing.append(requirement.requirement_id)

            # Calculate overall compliance score
            analysis.compliance_score = self._calculate_compliance_score(analysis.compliance_insights)

            # Determine compliance level
            analysis.compliance_level = self._determine_compliance_level(analysis.compliance_score)

            # Assess risk level
            analysis.risk_level = self._assess_risk_level(analysis)
            analysis.risk_factors = self._identify_risk_factors(analysis)

            return {
                'success': True,
                'compliance_analysis': analysis.to_dict()
            }

        except Exception as e:
            logger.error(f"Compliance insight extraction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _analyze_requirement_compliance(self, content: str,
                                      requirement: ComplianceRequirement) -> List[ComplianceInsight]:
        """Analyze compliance with a specific requirement"""
        insights = []

        try:
            # Check if requirement applies to this document
            if not self._requirement_applies(requirement, content):
                return insights

            # Check for positive indicators (must contain)
            positive_score = self._check_positive_indicators(content, requirement)

            # Check for negative indicators (must not contain)
            negative_score = self._check_negative_indicators(content, requirement)

            # Check for context indicators
            context_score = self._check_context_indicators(content, requirement)

            # Determine compliance status
            overall_score = (positive_score * 0.5) + (context_score * 0.3) - (negative_score * 0.2)
            overall_score = max(0.0, min(1.0, overall_score))  # Clamp to 0-1

            if overall_score >= 0.7:
                insight_type = 'requirement'
                severity = 'info'
            elif overall_score >= 0.4:
                insight_type = 'gap'
                severity = 'warning'
            else:
                insight_type = 'violation'
                severity = 'error'

            # Create insight
            insight = ComplianceInsight(
                insight_type=insight_type,
                requirement_id=requirement.requirement_id,
                requirement_title=requirement.title,
                severity=severity,
                content=self._generate_insight_content(requirement, overall_score, positive_score, negative_score),
                evidence=self._generate_evidence(content, requirement),
                compliance_score=overall_score,
                metadata={
                    'positive_score': positive_score,
                    'negative_score': negative_score,
                    'context_score': context_score,
                    'requirement_category': requirement.category,
                    'requirement_subcategory': requirement.subcategory
                }
            )

            insights.append(insight)

        except Exception as e:
            logger.error(f"Error analyzing requirement {requirement.requirement_id}: {e}")

        return insights

    def _requirement_applies(self, requirement: ComplianceRequirement, content: str) -> bool:
        """Check if a requirement applies to the document"""
        # Check document types
        if requirement.applicable_document_types:
            # This would need document type detection - for now, assume it applies
            pass

        # Check systems
        if requirement.applicable_systems:
            systems_mentioned = any(system.lower() in content.lower() for system in requirement.applicable_systems)
            if not systems_mentioned:
                return False

        return True

    def _check_positive_indicators(self, content: str, requirement: ComplianceRequirement) -> float:
        """Check for positive compliance indicators"""
        if not requirement.must_contain_keywords:
            return 0.5  # Neutral if no specific requirements

        found_count = 0
        for keyword in requirement.must_contain_keywords:
            if re.search(f'\\b{re.escape(keyword)}\\b', content, re.IGNORECASE):
                found_count += 1

        return found_count / len(requirement.must_contain_keywords) if requirement.must_contain_keywords else 0.5

    def _check_negative_indicators(self, content: str, requirement: ComplianceRequirement) -> float:
        """Check for negative compliance indicators"""
        if not requirement.must_not_contain_keywords:
            return 0.0  # No penalty if no prohibitions

        found_count = 0
        for keyword in requirement.must_not_contain_keywords:
            if re.search(f'\\b{re.escape(keyword)}\\b', content, re.IGNORECASE):
                found_count += 1

        return found_count / len(requirement.must_not_contain_keywords) if requirement.must_not_contain_keywords else 0.0

    def _check_context_indicators(self, content: str, requirement: ComplianceRequirement) -> float:
        """Check for context indicators"""
        if not requirement.keywords:
            return 0.5

        found_count = 0
        for keyword in requirement.keywords:
            if re.search(f'\\b{re.escape(keyword)}\\b', content, re.IGNORECASE):
                found_count += 1

        return found_count / len(requirement.keywords) if requirement.keywords else 0.5

    def _generate_insight_content(self, requirement: ComplianceRequirement, overall_score: float,
                                positive_score: float, negative_score: float) -> str:
        """Generate human-readable insight content"""
        if overall_score >= 0.7:
            return f"Document complies with requirement: {requirement.title}"
        elif overall_score >= 0.4:
            return f"Document partially meets requirement: {requirement.title}. Consider adding missing elements."
        else:
            return f"Document may violate requirement: {requirement.title}. Review and address compliance gaps."

    def _generate_evidence(self, content: str, requirement: ComplianceRequirement) -> List[str]:
        """Generate evidence for compliance assessment"""
        evidence = []

        # Find evidence for positive indicators
        for keyword in requirement.must_contain_keywords or []:
            if re.search(f'\\b{re.escape(keyword)}\\b', content, re.IGNORECASE):
                evidence.append(f"Found required keyword: '{keyword}'")

        # Find evidence for negative indicators
        for keyword in requirement.must_not_contain_keywords or []:
            if re.search(f'\\b{re.escape(keyword)}\\b', content, re.IGNORECASE):
                evidence.append(f"Found prohibited keyword: '{keyword}'")

        # Find evidence for context keywords
        for keyword in requirement.keywords or []:
            if re.search(f'\\b{re.escape(keyword)}\\b', content, re.IGNORECASE):
                evidence.append(f"Found context keyword: '{keyword}'")

        return evidence

    def _calculate_compliance_score(self, insights: List[ComplianceInsight]) -> float:
        """Calculate overall compliance score"""
        if not insights:
            return 0.0

        # Weight by severity
        severity_weights = {
            'info': 1.0,
            'warning': 0.7,
            'error': 0.3
        }

        total_weighted_score = 0.0
        total_weight = 0.0

        for insight in insights:
            weight = severity_weights.get(insight.severity, 0.5)
            total_weighted_score += insight.compliance_score * weight
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _determine_compliance_level(self, score: float) -> str:
        """Determine compliance level based on score"""
        if score >= 0.8:
            return 'compliant'
        elif score >= 0.6:
            return 'partially_compliant'
        else:
            return 'non_compliant'

    def _assess_risk_level(self, analysis: ComplianceAnalysis) -> str:
        """Assess risk level based on compliance analysis"""
        # Count violations and gaps
        violations = len([i for i in analysis.compliance_insights if i.insight_type == 'violation'])
        gaps = len([i for i in analysis.compliance_insights if i.insight_type == 'gap'])

        if violations > 0:
            return 'high'
        elif gaps > 2:
            return 'medium'
        else:
            return 'low'

    def _identify_risk_factors(self, analysis: ComplianceAnalysis) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []

        # Check for critical requirement violations
        critical_violations = [
            i for i in analysis.compliance_insights
            if i.insight_type == 'violation' and 'critical' in i.requirement_title.lower()
        ]

        if critical_violations:
            risk_factors.append(f"{len(critical_violations)} critical requirement violations")

        # Check for high-severity gaps
        high_severity_gaps = [
            i for i in analysis.compliance_insights
            if i.insight_type == 'gap' and i.severity == 'error'
        ]

        if high_severity_gaps:
            risk_factors.append(f"{len(high_severity_gaps)} high-severity compliance gaps")

        # Check for missing required sections
        if len(analysis.requirements_missing) > 3:
            risk_factors.append("Multiple required compliance elements missing")

        return risk_factors

    def _generate_document_id(self, file_path: str) -> str:
        """Generate document ID"""
        import hashlib
        path_str = os.path.abspath(file_path)
        return hashlib.md5(path_str.encode()).hexdigest()[:12]


class ComplianceAwareInsightEngine:
    """Main engine for compliance-aware insights"""

    def __init__(self):
        self.compliance_extractor = ComplianceInsightExtractor()
        self.insight_cache = {}

    def generate_compliance_insights(self, file_path: str,
                                   include_general_insights: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive compliance-aware insights

        Args:
            file_path: Path to document
            include_general_insights: Whether to include general insights

        Returns:
            Combined compliance and general insights
        """
        try:
            # Generate compliance insights
            compliance_result = self.compliance_extractor.extract_compliance_insights(file_path)

            if not compliance_result['success']:
                return compliance_result

            # Generate general insights if requested
            general_insights = []
            if include_general_insights:
                general_result = insight_extractor.extract_insights(file_path)
                if general_result['success']:
                    general_insights = general_result['insights']

            # Combine results
            combined_result = {
                'success': True,
                'file_path': file_path,
                'compliance_analysis': compliance_result['compliance_analysis'],
                'general_insights': [insight.to_dict() for insight in general_insights],
                'summary': self._generate_combined_summary(
                    compliance_result['compliance_analysis'],
                    general_insights
                ),
                'generated_at': datetime.now().isoformat()
            }

            return combined_result

        except Exception as e:
            logger.error(f"Compliance-aware insight generation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_combined_summary(self, compliance_analysis: ComplianceAnalysis,
                                 general_insights: List[ExtractedInsight]) -> Dict[str, Any]:
        """Generate combined summary of compliance and general insights"""
        # Count insight types
        insight_types = Counter(insight.insight_type.value for insight in general_insights)

        # Count compliance issues
        compliance_issues = len([i for i in compliance_analysis.compliance_insights
                               if i.insight_type in ['violation', 'gap']])

        return {
            'compliance_score': compliance_analysis.compliance_score,
            'compliance_level': compliance_analysis.compliance_level,
            'risk_level': compliance_analysis.risk_level,
            'total_general_insights': len(general_insights),
            'insight_type_distribution': dict(insight_types),
            'compliance_issues': compliance_issues,
            'requirements_met': len(compliance_analysis.requirements_met),
            'requirements_violated': len(compliance_analysis.requirements_violated),
            'overall_health': self._calculate_overall_health(compliance_analysis, general_insights)
        }

    def _calculate_overall_health(self, compliance_analysis: ComplianceAnalysis,
                                general_insights: List[ExtractedInsight]) -> str:
        """Calculate overall document health"""
        # Simple health calculation
        compliance_factor = compliance_analysis.compliance_score
        insight_factor = min(len(general_insights) / 10.0, 1.0)  # Normalize to 10 insights = full score
        risk_penalty = 1.0 if compliance_analysis.risk_level == 'low' else 0.7

        overall_score = (compliance_factor * 0.6 + insight_factor * 0.4) * risk_penalty

        if overall_score >= 0.8:
            return 'excellent'
        elif overall_score >= 0.6:
            return 'good'
        elif overall_score >= 0.4:
            return 'fair'
        else:
            return 'poor'

    def analyze_compliance_trends(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze compliance trends across multiple documents

        Args:
            document_paths: List of document paths

        Returns:
            Compliance trend analysis
        """
        try:
            compliance_results = []

            for file_path in document_paths:
                result = self.generate_compliance_insights(file_path, include_general_insights=False)
                if result['success']:
                    compliance_results.append(result['compliance_analysis'])

            if not compliance_results:
                return {
                    'success': False,
                    'error': 'No successful compliance analyses'
                }

            # Analyze trends
            trends = {
                'total_documents': len(document_paths),
                'analyzed_documents': len(compliance_results),
                'compliance_distribution': self._analyze_compliance_distribution(compliance_results),
                'risk_distribution': self._analyze_risk_distribution(compliance_results),
                'common_issues': self._identify_common_issues(compliance_results),
                'improvement_opportunities': self._identify_improvement_opportunities(compliance_results)
            }

            return {
                'success': True,
                'compliance_trends': trends
            }

        except Exception as e:
            logger.error(f"Compliance trends analysis error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _analyze_compliance_distribution(self, analyses: List[ComplianceAnalysis]) -> Dict[str, int]:
        """Analyze distribution of compliance levels"""
        distribution = Counter(analysis.compliance_level for analysis in analyses)
        return dict(distribution)

    def _analyze_risk_distribution(self, analyses: List[ComplianceAnalysis]) -> Dict[str, int]:
        """Analyze distribution of risk levels"""
        distribution = Counter(analysis.risk_level for analysis in analyses)
        return dict(distribution)

    def _identify_common_issues(self, analyses: List[ComplianceAnalysis]) -> List[Dict[str, Any]]:
        """Identify most common compliance issues"""
        issue_counts = Counter()

        for analysis in analyses:
            for insight in analysis.compliance_insights:
                if insight.insight_type in ['violation', 'gap']:
                    issue_counts[insight.requirement_id] += 1

        # Return top 5 issues
        common_issues = []
        for requirement_id, count in issue_counts.most_common(5):
            requirement = next((r for r in self.compliance_extractor.requirements
                              if r.requirement_id == requirement_id), None)

            if requirement:
                common_issues.append({
                    'requirement_id': requirement_id,
                    'requirement_title': requirement.title,
                    'occurrence_count': count,
                    'severity': 'high' if count > len(analyses) * 0.3 else 'medium'
                })

        return common_issues

    def _identify_improvement_opportunities(self, analyses: List[ComplianceAnalysis]) -> List[str]:
        """Identify opportunities for compliance improvement"""
        opportunities = []

        # Analyze patterns in missing requirements
        missing_requirements = Counter()
        for analysis in analyses:
            for req_id in analysis.requirements_missing:
                missing_requirements[req_id] += 1

        if missing_requirements:
            most_missing = missing_requirements.most_common(3)
            opportunities.append(f"Most documents missing: {[req for req, _ in most_missing]}")

        # Analyze low-scoring areas
        avg_compliance_by_category = defaultdict(list)
        for analysis in analyses:
            for insight in analysis.compliance_insights:
                category = insight.metadata.get('requirement_category', 'unknown')
                avg_compliance_by_category[category].append(insight.compliance_score)

        for category, scores in avg_compliance_by_category.items():
            if scores and sum(scores) / len(scores) < 0.6:
                opportunities.append(f"Improve compliance in {category} area")

        return opportunities

    def generate_compliance_report(self, file_path: str, output_format: str = 'pdf') -> Dict[str, Any]:
        """
        Generate comprehensive compliance report

        Args:
            file_path: Path to document
            output_format: Report format

        Returns:
            Report generation results
        """
        try:
            # Generate compliance insights
            result = self.generate_compliance_insights(file_path)

            if not result['success']:
                return result

            # Generate summary
            summary_config = SummarizationConfig(
                method=SummarizationMethod.HYBRID,
                detail_level=DetailLevel.DETAILED
            )
            summary_result = summarization_engine.summarize_document(file_path, summary_config)

            # Prepare report data
            report_data = {
                'document_path': file_path,
                'compliance_analysis': result['compliance_analysis'],
                'general_insights': result['general_insights'],
                'summary': summary_result.summary if summary_result.success else None,
                'generated_at': datetime.now().isoformat()
            }

            # Export report
            from .export_manager import export_manager

            export_result = export_manager.export_dashboard(report_data, output_format)

            return {
                'success': True,
                'report_generated': True,
                'export_result': export_result
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class ComplianceInsightAPI:
    """API for compliance insight operations"""

    def __init__(self):
        self.insight_engine = ComplianceAwareInsightEngine()

    def get_compliance_insights(self, file_path: str) -> Dict[str, Any]:
        """Get compliance insights for a document"""
        return self.insight_engine.generate_compliance_insights(file_path)

    def analyze_compliance_trends(self, document_paths: List[str]) -> Dict[str, Any]:
        """Analyze compliance trends across documents"""
        return self.insight_engine.analyze_compliance_trends(document_paths)

    def generate_compliance_report(self, file_path: str, output_format: str = 'pdf') -> Dict[str, Any]:
        """Generate compliance report"""
        return self.insight_engine.generate_compliance_report(file_path, output_format)

    def get_compliance_requirements(self) -> List[Dict[str, Any]]:
        """Get all compliance requirements"""
        return [req.__dict__ for req in self.insight_engine.compliance_extractor.requirements]


# Global compliance insight instances
compliance_insight_engine = ComplianceAwareInsightEngine()
compliance_insight_api = ComplianceInsightAPI()