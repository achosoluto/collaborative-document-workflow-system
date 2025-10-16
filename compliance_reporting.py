"""
Automated Compliance Reporting and Alert System
"""

import json
import logging
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    try:
        # Handle older Python versions
        from email.MIMEText import MIMEText as MimeText
        from email.MIMEMultipart import MIMEMultipart as MimeMultipart
    except ImportError:
        # Disable email functionality if not available
        MimeText = None
        MimeMultipart = None
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

try:
    from .compliance_models import (
        ComplianceReport, ComplianceAlert, ComplianceDataManager,
        ComplianceAssessment, ComplianceViolation
    )
    from .compliance_config import compliance_config
except ImportError:
    from compliance_models import (
        ComplianceReport, ComplianceAlert, ComplianceDataManager,
        ComplianceAssessment, ComplianceViolation
    )
    from compliance_config import compliance_config

logger = logging.getLogger(__name__)


@dataclass
class ReportTemplate:
    """Template for compliance reports"""

    template_id: str
    name: str
    description: str
    report_type: str  # 'daily', 'weekly', 'monthly', 'ad_hoc'

    # Template configuration
    sections: List[str]
    include_charts: bool = True
    include_detailed_findings: bool = True
    include_recommendations: bool = True

    # Recipients
    default_recipients: List[str] = None

    def __post_init__(self):
        if self.default_recipients is None:
            self.default_recipients = []


@dataclass
class AlertRule:
    """Rule for triggering compliance alerts"""

    rule_id: str
    name: str
    description: str

    # Trigger conditions
    trigger_conditions: Dict[str, Any]
    severity: str = 'medium'

    # Alert configuration
    notification_channels: List[str] = None  # 'email', 'dashboard', 'sms'
    escalation_timeout: int = 24  # hours

    # Status
    enabled: bool = True

    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ['email']


class ComplianceReporter:
    """Generates automated compliance reports"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()
        self.templates = self._load_report_templates()

    def _load_report_templates(self) -> Dict[str, ReportTemplate]:
        """Load report templates"""
        templates = {
            'daily_summary': ReportTemplate(
                template_id='daily_summary',
                name='Daily Compliance Summary',
                description='Daily summary of compliance status and violations',
                report_type='daily',
                sections=['overview', 'violations', 'trends', 'recommendations'],
                include_charts=True,
                include_detailed_findings=True,
                include_recommendations=True,
                default_recipients=['compliance-team@company.com']
            ),
            'weekly_detailed': ReportTemplate(
                template_id='weekly_detailed',
                name='Weekly Compliance Report',
                description='Detailed weekly compliance analysis',
                report_type='weekly',
                sections=['overview', 'detailed_assessments', 'violations', 'trends', 'analytics'],
                include_charts=True,
                include_detailed_findings=True,
                include_recommendations=True,
                default_recipients=['management@company.com', 'compliance-team@company.com']
            ),
            'violation_alert': ReportTemplate(
                template_id='violation_alert',
                name='Critical Violation Alert',
                description='Alert for critical compliance violations',
                report_type='ad_hoc',
                sections=['violation_details', 'immediate_actions', 'escalation'],
                include_charts=False,
                include_detailed_findings=True,
                include_recommendations=True,
                default_recipients=['compliance-officer@company.com']
            )
        }
        return templates

    def generate_daily_report(self) -> ComplianceReport:
        """Generate daily compliance report"""
        return self._generate_report('daily_summary', 'Daily Compliance Summary')

    def generate_weekly_report(self) -> ComplianceReport:
        """Generate weekly compliance report"""
        return self._generate_report('weekly_detailed', 'Weekly Compliance Report')

    def generate_violation_alert_report(self, violation_ids: List[str]) -> ComplianceReport:
        """Generate report for specific violations"""
        return self._generate_report('violation_alert', 'Critical Violation Alert', violation_ids)

    def _generate_report(
        self,
        template_id: str,
        title: str,
        violation_ids: List[str] = None
    ) -> ComplianceReport:
        """Generate a compliance report using specified template"""
        try:
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")

            report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get report data based on template
            if template_id == 'violation_alert' and violation_ids:
                report_data = self._get_violation_report_data(violation_ids)
            else:
                report_data = self._get_standard_report_data()

            # Create report
            report = ComplianceReport(
                report_id=report_id,
                report_type=template.report_type,
                title=title,
                description=template.description,
                scope_type='all',
                scope_value='all',
                period_start=report_data['period_start'],
                period_end=report_data['period_end'],
                total_documents_assessed=report_data['total_documents'],
                compliant_documents=report_data['compliant_documents'],
                non_compliant_documents=report_data['non_compliant_documents'],
                partial_compliance_documents=report_data['partial_documents'],
                critical_violations=report_data['critical_violations'],
                high_risk_violations=report_data['high_violations'],
                medium_risk_violations=report_data['medium_violations'],
                low_risk_violations=report_data['low_violations'],
                assessments=report_data['assessment_ids'],
                violations=report_data['violation_ids'],
                recommendations=report_data['recommendations']
            )

            # Save report
            self._save_report(report)

            return report

        except Exception as e:
            logger.error(f"Error generating report {template_id}: {e}")
            return None

    def _get_standard_report_data(self) -> Dict[str, Any]:
        """Get data for standard reports"""
        # Get assessments from last 24 hours for daily, last 7 days for weekly
        cutoff_date = datetime.now() - timedelta(days=1)  # Daily report

        assessments = self.data_manager.load_assessments()
        recent_assessments = [a for a in assessments if a.assessed_at >= cutoff_date]

        violations = self.data_manager.load_violations()
        recent_violations = [v for v in violations if v.created_at >= cutoff_date]

        # Calculate statistics
        total_assessments = len(recent_assessments)
        compliant = len([a for a in recent_assessments if a.overall_status == 'compliant'])
        non_compliant = len([a for a in recent_assessments if a.overall_status == 'non_compliant'])
        partial = len([a for a in recent_assessments if a.overall_status == 'partial'])

        # Count violations by severity
        critical_violations = len([v for v in recent_violations if v.severity == 'critical'])
        high_violations = len([v for v in recent_violations if v.severity == 'high'])
        medium_violations = len([v for v in recent_violations if v.severity == 'medium'])
        low_violations = len([v for v in recent_violations if v.severity == 'low'])

        # Generate recommendations
        recommendations = self._generate_report_recommendations(recent_assessments, recent_violations)

        return {
            'period_start': cutoff_date,
            'period_end': datetime.now(),
            'total_documents': total_assessments,
            'compliant_documents': compliant,
            'non_compliant_documents': non_compliant,
            'partial_documents': partial,
            'critical_violations': critical_violations,
            'high_violations': high_violations,
            'medium_violations': medium_violations,
            'low_violations': low_violations,
            'assessment_ids': [a.assessment_id for a in recent_assessments],
            'violation_ids': [v.violation_id for v in recent_violations],
            'recommendations': recommendations
        }

    def _get_violation_report_data(self, violation_ids: List[str]) -> Dict[str, Any]:
        """Get data for violation-specific report"""
        violations = self.data_manager.load_violations()
        report_violations = [v for v in violations if v.violation_id in violation_ids]

        # Get related assessments
        assessments = self.data_manager.load_assessments()
        related_assessments = [
            a for a in assessments
            if any(v.assessment_id == a.assessment_id for v in report_violations)
        ]

        # Calculate statistics
        critical_count = len([v for v in report_violations if v.severity == 'critical'])
        high_count = len([v for v in report_violations if v.severity == 'high'])

        return {
            'period_start': datetime.now() - timedelta(days=1),
            'period_end': datetime.now(),
            'total_documents': len(set(v.doc_id for v in report_violations)),
            'compliant_documents': 0,
            'non_compliant_documents': len(set(v.doc_id for v in report_violations)),
            'partial_documents': 0,
            'critical_violations': critical_count,
            'high_violations': high_count,
            'medium_violations': len([v for v in report_violations if v.severity == 'medium']),
            'low_violations': len([v for v in report_violations if v.severity == 'low']),
            'assessment_ids': [a.assessment_id for a in related_assessments],
            'violation_ids': violation_ids,
            'recommendations': [f"Immediate attention required for {len(report_violations)} violations"]
        }

    def _generate_report_recommendations(
        self,
        assessments: List[ComplianceAssessment],
        violations: List[ComplianceViolation]
    ) -> List[str]:
        """Generate recommendations for the report"""
        recommendations = []

        # Analyze compliance patterns
        if assessments:
            avg_score = sum(a.compliance_score for a in assessments) / len(assessments)
            if avg_score < 70:
                recommendations.append(
                    "Overall compliance score is below acceptable threshold. Immediate action required."
                )

        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == 'critical']
        if critical_violations:
            recommendations.append(
                f"{len(critical_violations)} critical violations require immediate attention."
            )

        # Check for overdue violations
        overdue_violations = [
            v for v in violations
            if v.remediation_deadline and v.remediation_deadline < datetime.now()
        ]
        if overdue_violations:
            recommendations.append(
                f"{len(overdue_violations)} violations are past their remediation deadline."
            )

        # General recommendations
        if not recommendations:
            recommendations.append("Compliance status is within acceptable parameters. Continue monitoring.")

        return recommendations

    def _save_report(self, report: ComplianceReport) -> bool:
        """Save report to data manager"""
        # This would integrate with the data manager's save functionality
        # For now, save to a reports file
        try:
            reports_file = Path("data/compliance/reports.json")
            reports_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing reports
            existing_reports = []
            if reports_file.exists():
                with open(reports_file, 'r') as f:
                    existing_reports = json.load(f)

            # Add new report
            existing_reports.append(report.to_dict())

            # Save back to file
            with open(reports_file, 'w') as f:
                json.dump(existing_reports, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return False


class AlertManager:
    """Manages compliance alerts and notifications"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()
        self.alert_rules = self._load_alert_rules()
        self.smtp_configured = self._check_smtp_configuration()

    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rules"""
        rules = [
            AlertRule(
                rule_id='critical_violation_detected',
                name='Critical Violation Detected',
                description='Alert when critical compliance violations are detected',
                trigger_conditions={
                    'violation_severity': 'critical',
                    'violation_status': 'open'
                },
                severity='critical',
                notification_channels=['email', 'dashboard'],
                escalation_timeout=4
            ),
            AlertRule(
                rule_id='high_violation_detected',
                name='High Severity Violation Detected',
                description='Alert when high severity compliance violations are detected',
                trigger_conditions={
                    'violation_severity': 'high',
                    'violation_status': 'open'
                },
                severity='high',
                notification_channels=['email', 'dashboard'],
                escalation_timeout=24
            ),
            AlertRule(
                rule_id='compliance_score_drop',
                name='Compliance Score Drop',
                description='Alert when overall compliance score drops significantly',
                trigger_conditions={
                    'score_drop_threshold': 10,
                    'time_window_hours': 24
                },
                severity='medium',
                notification_channels=['email'],
                escalation_timeout=48
            ),
            AlertRule(
                rule_id='deadline_approaching',
                name='Deadline Approaching',
                description='Alert when compliance deadlines are approaching',
                trigger_conditions={
                    'days_ahead': 7,
                    'requirement_severity': ['critical', 'high']
                },
                severity='medium',
                notification_channels=['email'],
                escalation_timeout=72
            )
        ]
        return rules

    def _check_smtp_configuration(self) -> bool:
        """Check if SMTP is configured for email alerts"""
        # Check for SMTP configuration in environment or config file
        return False  # Placeholder - would check actual SMTP config

    def check_and_trigger_alerts(self) -> List[ComplianceAlert]:
        """Check for alert conditions and trigger alerts"""
        triggered_alerts = []

        try:
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue

                # Check each rule's conditions
                new_alerts = self._check_rule_conditions(rule)
                triggered_alerts.extend(new_alerts)

                # Save triggered alerts
                for alert in new_alerts:
                    self._save_alert(alert)

        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")

        return triggered_alerts

    def _check_rule_conditions(self, rule: AlertRule) -> List[ComplianceAlert]:
        """Check conditions for a specific alert rule"""
        alerts = []

        try:
            if rule.rule_id == 'critical_violation_detected':
                alerts.extend(self._check_critical_violations(rule))
            elif rule.rule_id == 'high_violation_detected':
                alerts.extend(self._check_high_violations(rule))
            elif rule.rule_id == 'compliance_score_drop':
                alerts.extend(self._check_compliance_score_drop(rule))
            elif rule.rule_id == 'deadline_approaching':
                alerts.extend(self._check_deadline_approaching(rule))

        except Exception as e:
            logger.error(f"Error checking conditions for rule {rule.rule_id}: {e}")

        return alerts

    def _check_critical_violations(self, rule: AlertRule) -> List[ComplianceAlert]:
        """Check for critical violations"""
        alerts = []

        # Get recent critical violations
        cutoff_time = datetime.now() - timedelta(hours=1)  # Last hour
        violations = self.data_manager.load_violations(status='open')

        critical_violations = [
            v for v in violations
            if v.severity == 'critical' and v.created_at >= cutoff_time
        ]

        for violation in critical_violations:
            # Check if alert already exists for this violation
            if not self._alert_exists_for_violation(violation.violation_id):
                alert = ComplianceAlert(
                    alert_id=f"alert_{violation.violation_id}",
                    alert_type='violation_detected',
                    severity='critical',
                    title='Critical Compliance Violation Detected',
                    message=f'A critical compliance violation has been detected in document {violation.doc_id}',
                    details={
                        'violation_id': violation.violation_id,
                        'violation_type': violation.violation_type,
                        'description': violation.description,
                        'remediation_deadline': violation.remediation_deadline.isoformat() if violation.remediation_deadline else None
                    },
                    related_doc_id=violation.doc_id,
                    related_violation_id=violation.violation_id,
                    status='active'
                )
                alerts.append(alert)

        return alerts

    def _check_high_violations(self, rule: AlertRule) -> List[ComplianceAlert]:
        """Check for high severity violations"""
        alerts = []

        # Get recent high severity violations
        cutoff_time = datetime.now() - timedelta(hours=6)  # Last 6 hours
        violations = self.data_manager.load_violations(status='open')

        high_violations = [
            v for v in violations
            if v.severity == 'high' and v.created_at >= cutoff_time
        ]

        for violation in high_violations:
            if not self._alert_exists_for_violation(violation.violation_id):
                alert = ComplianceAlert(
                    alert_id=f"alert_{violation.violation_id}",
                    alert_type='violation_detected',
                    severity='high',
                    title='High Severity Compliance Violation',
                    message=f'A high severity compliance violation requires attention in document {violation.doc_id}',
                    details={
                        'violation_id': violation.violation_id,
                        'violation_type': violation.violation_type,
                        'description': violation.description
                    },
                    related_doc_id=violation.doc_id,
                    related_violation_id=violation.violation_id,
                    status='active'
                )
                alerts.append(alert)

        return alerts

    def _check_compliance_score_drop(self, rule: AlertRule) -> List[ComplianceAlert]:
        """Check for significant compliance score drops"""
        alerts = []

        # This would compare current average score with historical average
        # For now, return empty list as this requires more complex analysis
        return alerts

    def _check_deadline_approaching(self, rule: AlertRule) -> List[ComplianceAlert]:
        """Check for approaching deadlines"""
        alerts = []

        # Get requirements with approaching deadlines
        requirements = self.data_manager.load_requirements()
        days_ahead = rule.trigger_conditions.get('days_ahead', 7)

        for requirement in requirements:
            if (requirement.expiry_date and
                requirement.severity in rule.trigger_conditions.get('requirement_severity', [])):

                days_until_expiry = (requirement.expiry_date - datetime.now()).days
                if 0 < days_until_expiry <= days_ahead:

                    alert = ComplianceAlert(
                        alert_id=f"alert_deadline_{requirement.requirement_id}",
                        alert_type='deadline_approaching',
                        severity='medium',
                        title='Compliance Deadline Approaching',
                        message=f'Requirement "{requirement.title}" expires in {days_until_expiry} days',
                        details={
                            'requirement_id': requirement.requirement_id,
                            'expiry_date': requirement.expiry_date.isoformat(),
                            'days_until_expiry': days_until_expiry
                        },
                        related_requirement_id=requirement.requirement_id,
                        status='active'
                    )
                    alerts.append(alert)

        return alerts

    def _alert_exists_for_violation(self, violation_id: str) -> bool:
        """Check if alert already exists for a violation"""
        # This would check the alerts data store
        # For now, return False to always create alerts
        return False

    def _save_alert(self, alert: ComplianceAlert) -> bool:
        """Save alert to data store"""
        try:
            alerts_file = Path("data/compliance/alerts.json")
            alerts_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing alerts
            existing_alerts = []
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    existing_alerts = json.load(f)

            # Add new alert
            existing_alerts.append(alert.to_dict())

            # Save back to file
            with open(alerts_file, 'w') as f:
                json.dump(existing_alerts, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving alert: {e}")
            return False

    def send_email_alert(self, alert: ComplianceAlert, recipients: List[str]) -> bool:
        """Send email alert"""
        if not self.smtp_configured:
            logger.warning("SMTP not configured, cannot send email alert")
            return False

        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = 'compliance-system@company.com'
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f'[{alert.severity.upper()}] {alert.title}'

            # Email body
            body = f"""
Compliance Alert

Title: {alert.title}
Severity: {alert.severity}
Type: {alert.alert_type}

Message:
{alert.message}

Details:
{json.dumps(alert.details, indent=2)}

Generated at: {alert.created_at}

Please log in to the compliance dashboard for more information.
"""
            msg.attach(MimeText(body, 'plain'))

            # Send email (placeholder - would use actual SMTP)
            logger.info(f"Would send email alert to {recipients}: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False


class ComplianceReportingManager:
    """Main manager for compliance reporting and alerts"""

    def __init__(self):
        self.reporter = ComplianceReporter()
        self.alert_manager = AlertManager()

    def generate_daily_reports(self) -> List[ComplianceReport]:
        """Generate all daily reports"""
        reports = []

        try:
            # Generate daily summary report
            daily_report = self.reporter.generate_daily_report()
            if daily_report:
                reports.append(daily_report)

            # Check and send alerts
            alerts = self.alert_manager.check_and_trigger_alerts()

            logger.info(f"Generated {len(reports)} reports and {len(alerts)} alerts")

        except Exception as e:
            logger.error(f"Error in daily reporting: {e}")

        return reports

    def generate_custom_report(
        self,
        template_id: str,
        title: str,
        violation_ids: List[str] = None
    ) -> Optional[ComplianceReport]:
        """Generate a custom report"""
        return self.reporter._generate_report(template_id, title, violation_ids)

    def trigger_violation_alert(self, violation_ids: List[str]) -> bool:
        """Trigger alerts for specific violations"""
        try:
            # Generate violation report
            report = self.reporter.generate_violation_alert_report(violation_ids)

            # Get violations for alerting
            violations = self.alert_manager.data_manager.load_violations()
            alert_violations = [v for v in violations if v.violation_id in violation_ids]

            # Send alerts for critical and high severity violations
            critical_violations = [v for v in alert_violations if v.severity in ['critical', 'high']]

            if critical_violations:
                # Create and send alerts
                for violation in critical_violations:
                    alert = ComplianceAlert(
                        alert_id=f"manual_alert_{violation.violation_id}",
                        alert_type='violation_detected',
                        severity=violation.severity,
                        title=f'Manual Alert: {violation.title}',
                        message=f'Manual alert triggered for compliance violation in document {violation.doc_id}',
                        details={'violation_id': violation.violation_id},
                        related_violation_id=violation.violation_id,
                        status='active'
                    )

                    self.alert_manager._save_alert(alert)

                    # Send email if configured
                    if alert.severity == 'critical':
                        self.alert_manager.send_email_alert(
                            alert,
                            compliance_config.critical_alert_recipients
                        )

            return True

        except Exception as e:
            logger.error(f"Error triggering violation alert: {e}")
            return False