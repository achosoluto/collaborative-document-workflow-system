"""
Compliance Analytics and Trend Analysis Module
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import statistics

try:
    from .compliance_models import (
        ComplianceAssessment, ComplianceViolation, ComplianceDataManager,
        ComplianceAnalytics
    )
    from .compliance_config import compliance_config
except ImportError:
    from compliance_models import (
        ComplianceAssessment, ComplianceViolation, ComplianceDataManager,
        ComplianceAnalytics
    )
    from compliance_config import compliance_config

logger = logging.getLogger(__name__)


class ComplianceTrendAnalyzer:
    """Analyzes compliance trends and patterns"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()

    def analyze_compliance_trends(self, days: int = 90) -> Dict[str, Any]:
        """Analyze compliance trends over a specified period"""
        try:
            # Get assessments from the specified period
            cutoff_date = datetime.now() - timedelta(days=days)
            assessments = self.data_manager.load_assessments()
            recent_assessments = [a for a in assessments if a.assessed_at >= cutoff_date]

            if not recent_assessments:
                return {
                    'status': 'no_data',
                    'message': f'No assessment data available for the last {days} days'
                }

            # Group assessments by day
            daily_data = defaultdict(list)
            for assessment in recent_assessments:
                day_key = assessment.assessed_at.date().isoformat()
                daily_data[day_key].append(assessment.compliance_score)

            # Calculate daily averages
            daily_scores = []
            for day, scores in sorted(daily_data.items()):
                avg_score = statistics.mean(scores)
                daily_scores.append((day, avg_score))

            # Calculate trend statistics
            if len(daily_scores) >= 7:  # Need at least a week for meaningful trends
                scores = [score for _, score in daily_scores]

                # Overall trend
                first_week_avg = statistics.mean(scores[:7])
                last_week_scores = scores[-7:] if len(scores) >= 7 else scores
                last_week_avg = statistics.mean(last_week_scores)

                if last_week_avg > first_week_avg + 5:
                    trend_direction = 'improving'
                elif last_week_avg < first_week_avg - 5:
                    trend_direction = 'declining'
                else:
                    trend_direction = 'stable'

                # Volatility (standard deviation)
                volatility = statistics.stdev(scores) if len(scores) > 1 else 0

                # Best and worst performing days
                best_day = max(daily_scores, key=lambda x: x[1])
                worst_day = min(daily_scores, key=lambda x: x[1])

            else:
                trend_direction = 'insufficient_data'
                volatility = 0
                best_day = worst_day = daily_scores[0] if daily_scores else (None, 0)

            return {
                'period_days': days,
                'total_assessments': len(recent_assessments),
                'daily_scores': daily_scores,
                'trend_direction': trend_direction,
                'average_score': statistics.mean([score for _, score in daily_scores]),
                'volatility': volatility,
                'best_day': best_day,
                'worst_day': worst_day,
                'improvement_rate': self._calculate_improvement_rate(daily_scores)
            }

        except Exception as e:
            logger.error(f"Error analyzing compliance trends: {e}")
            return {'status': 'error', 'message': str(e)}

    def _calculate_improvement_rate(self, daily_scores: List[Tuple[str, float]]) -> float:
        """Calculate the rate of compliance improvement"""
        if len(daily_scores) < 7:
            return 0.0

        # Compare first and last weeks
        first_week = daily_scores[:7]
        last_week = daily_scores[-7:]

        first_avg = statistics.mean([score for _, score in first_week])
        last_avg = statistics.mean([score for _, score in last_week])

        # Calculate improvement rate per week
        weeks_diff = (len(daily_scores) - 7) / 7  # Number of weeks between measurements
        if weeks_diff <= 0:
            return 0.0

        improvement = last_avg - first_avg
        return improvement / weeks_diff

    def analyze_violation_patterns(self, days: int = 90) -> Dict[str, Any]:
        """Analyze patterns in compliance violations"""
        try:
            # Get violations from the specified period
            cutoff_date = datetime.now() - timedelta(days=days)
            violations = self.data_manager.load_violations()
            recent_violations = [v for v in violations if v.created_at >= cutoff_date]

            if not recent_violations:
                return {
                    'status': 'no_data',
                    'message': f'No violation data available for the last {days} days'
                }

            # Analyze by severity
            severity_counts = Counter(v.severity for v in recent_violations)

            # Analyze by type
            type_counts = Counter(v.violation_type for v in recent_violations)

            # Analyze by document type (need to get document metadata)
            doc_type_violations = defaultdict(int)
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    catalog = json.load(f)

                for violation in recent_violations:
                    if violation.doc_id in catalog:
                        doc_type = catalog[violation.doc_id].get('document_type', 'Unknown')
                        doc_type_violations[doc_type] += 1

            # Analyze temporal patterns
            daily_violations = defaultdict(int)
            for violation in recent_violations:
                day_key = violation.created_at.date().isoformat()
                daily_violations[day_key] += 1

            # Find peak violation days
            peak_days = sorted(daily_violations.items(), key=lambda x: x[1], reverse=True)[:5]

            # Calculate resolution time for resolved violations
            resolved_violations = [v for v in recent_violations if v.status == 'resolved' and v.resolved_at]
            resolution_times = []
            for violation in resolved_violations:
                if violation.resolved_at and violation.created_at:
                    days_to_resolve = (violation.resolved_at - violation.created_at).days
                    resolution_times.append(days_to_resolve)

            avg_resolution_time = statistics.mean(resolution_times) if resolution_times else 0

            return {
                'period_days': days,
                'total_violations': len(recent_violations),
                'severity_distribution': dict(severity_counts),
                'type_distribution': dict(type_counts),
                'document_type_distribution': dict(doc_type_violations),
                'daily_violations': dict(daily_violations),
                'peak_violation_days': peak_days,
                'average_resolution_time_days': avg_resolution_time,
                'resolution_rate': len(resolved_violations) / len(recent_violations) if recent_violations else 0
            }

        except Exception as e:
            logger.error(f"Error analyzing violation patterns: {e}")
            return {'status': 'error', 'message': str(e)}

    def predict_future_compliance(self, days_ahead: int = 30) -> Dict[str, Any]:
        """Predict future compliance trends based on historical data"""
        try:
            # Get historical trends
            trends_90 = self.analyze_compliance_trends(90)
            if trends_90.get('status') == 'no_data':
                return {'status': 'insufficient_data'}

            current_trend = trends_90['trend_direction']
            improvement_rate = trends_90['improvement_rate']
            current_avg = trends_90['average_score']

            # Simple linear prediction
            predicted_scores = []
            for day in range(1, days_ahead + 1):
                # Project future score based on current trend
                if current_trend == 'improving':
                    predicted_score = current_avg + (improvement_rate * day / 7)  # Weekly improvement rate
                elif current_trend == 'declining':
                    predicted_score = current_avg - (abs(improvement_rate) * day / 7)
                else:
                    predicted_score = current_avg

                # Keep within realistic bounds
                predicted_score = max(0, min(100, predicted_score))
                predicted_scores.append((day, predicted_score))

            # Predict potential risk areas
            risk_areas = []
            if current_trend == 'declining':
                risk_areas.append('Overall compliance score is declining')
            if trends_90['volatility'] > 15:
                risk_areas.append('High volatility in compliance scores indicates instability')

            # Get violation patterns for risk assessment
            violation_patterns = self.analyze_violation_patterns(30)
            if violation_patterns.get('total_violations', 0) > 10:
                risk_areas.append('High number of recent violations detected')

            return {
                'prediction_period_days': days_ahead,
                'current_trend': current_trend,
                'current_average_score': current_avg,
                'predicted_scores': predicted_scores,
                'improvement_rate': improvement_rate,
                'risk_areas': risk_areas,
                'confidence_level': 'medium',  # Could be calculated based on data quality
                'prediction_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error predicting future compliance: {e}")
            return {'status': 'error', 'message': str(e)}


class ComplianceAnalyticsEngine:
    """Main analytics engine for compliance data"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()
        self.trend_analyzer = ComplianceTrendAnalyzer()

    def generate_compliance_analytics(self, period_days: int = 90) -> ComplianceAnalytics:
        """Generate comprehensive compliance analytics"""
        try:
            analytics_id = f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            period_start = datetime.now() - timedelta(days=period_days)
            period_end = datetime.now()

            # Get all data for the period
            assessments = self.data_manager.load_assessments()
            violations = self.data_manager.load_violations()

            period_assessments = [a for a in assessments if period_start <= a.assessed_at <= period_end]
            period_violations = [v for v in violations if period_start <= v.created_at <= period_end]

            # Basic metrics
            total_documents = len(set(a.doc_id for a in period_assessments))
            assessed_documents = len(period_assessments)
            compliance_rate = 0.0
            avg_compliance_score = 0.0

            if period_assessments:
                compliant_assessments = len([a for a in period_assessments if a.overall_status == 'compliant'])
                compliance_rate = (compliant_assessments / len(period_assessments)) * 100
                avg_compliance_score = statistics.mean([a.compliance_score for a in period_assessments])

            # Violation metrics
            total_violations = len(period_violations)
            new_violations = len([v for v in period_violations if v.status == 'open'])
            resolved_violations = len([v for v in period_violations if v.status == 'resolved'])
            overdue_violations = len([
                v for v in period_violations
                if v.remediation_deadline and v.remediation_deadline < datetime.now()
            ])

            # Risk distribution
            risk_distribution = Counter(v.severity for v in period_violations)

            # Top violation types
            violation_types = Counter(v.violation_type for v in period_violations)
            top_violation_types = violation_types.most_common(5)

            # Compliance by category (based on requirements)
            compliance_by_category = {}
            for assessment in period_assessments:
                # Get requirement category (would need to load requirements)
                category = self._get_requirement_category(assessment.requirement_id)
                if category not in compliance_by_category:
                    compliance_by_category[category] = []
                compliance_by_category[category].append(assessment.compliance_score)

            # Calculate averages by category
            for category in compliance_by_category:
                scores = compliance_by_category[category]
                compliance_by_category[category] = statistics.mean(scores)

            # Compliance by system (based on document metadata)
            compliance_by_system = {}
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    catalog = json.load(f)

                for assessment in period_assessments:
                    if assessment.doc_id in catalog:
                        systems = catalog[assessment.doc_id].get('applicable_systems', [])
                        for system in systems:
                            if system not in compliance_by_system:
                                compliance_by_system[system] = []
                            compliance_by_system[system].append(assessment.compliance_score)

                # Calculate averages by system
                for system in compliance_by_system:
                    scores = compliance_by_system[system]
                    compliance_by_system[system] = statistics.mean(scores)

            # Generate trend data
            daily_scores = self._calculate_daily_compliance_scores(period_assessments)
            weekly_violations = self._calculate_weekly_violation_counts(period_violations)

            # Create analytics object
            analytics = ComplianceAnalytics(
                analytics_id=analytics_id,
                period_start=period_start,
                period_end=period_end,
                total_documents=total_documents,
                assessed_documents=assessed_documents,
                compliance_rate=compliance_rate,
                avg_compliance_score=avg_compliance_score,
                total_violations=total_violations,
                new_violations=new_violations,
                resolved_violations=resolved_violations,
                overdue_violations=overdue_violations,
                risk_distribution=dict(risk_distribution),
                top_violation_types=top_violation_types,
                compliance_by_category=compliance_by_category,
                compliance_by_system=compliance_by_system,
                daily_compliance_scores=daily_scores,
                weekly_violation_counts=weekly_violations
            )

            return analytics

        except Exception as e:
            logger.error(f"Error generating compliance analytics: {e}")
            return None

    def _get_requirement_category(self, requirement_id: str) -> str:
        """Get the category of a requirement"""
        try:
            requirements = self.data_manager.load_requirements()
            requirement = next((r for r in requirements if r.requirement_id == requirement_id), None)
            return requirement.category if requirement else 'unknown'
        except:
            return 'unknown'

    def _calculate_daily_compliance_scores(self, assessments: List[ComplianceAssessment]) -> List[Tuple[str, float]]:
        """Calculate daily average compliance scores"""
        daily_scores = defaultdict(list)

        for assessment in assessments:
            day_key = assessment.assessed_at.date().isoformat()
            daily_scores[day_key].append(assessment.compliance_score)

        # Calculate daily averages
        result = []
        for day in sorted(daily_scores.keys()):
            avg_score = statistics.mean(daily_scores[day])
            result.append((day, avg_score))

        return result

    def _calculate_weekly_violation_counts(self, violations: List[ComplianceViolation]) -> List[Tuple[str, int]]:
        """Calculate weekly violation counts"""
        weekly_counts = defaultdict(int)

        for violation in violations:
            # Get week start (Monday) for the violation date
            violation_date = violation.created_at.date()
            week_start = violation_date - timedelta(days=violation_date.weekday())
            week_key = week_start.isoformat()
            weekly_counts[week_key] += 1

        # Return as list of tuples
        result = []
        for week in sorted(weekly_counts.keys()):
            result.append((week, weekly_counts[week]))

        return result

    def identify_risk_areas(self) -> Dict[str, Any]:
        """Identify areas of high compliance risk"""
        try:
            # Get recent analytics
            analytics = self.generate_compliance_analytics(30)  # Last 30 days
            if not analytics:
                return {'status': 'no_data'}

            risk_areas = []

            # Check overall compliance rate
            if analytics.compliance_rate < 70:
                risk_areas.append({
                    'type': 'overall_compliance',
                    'severity': 'high',
                    'description': f'Overall compliance rate is {analytics.compliance_rate:.1f}%, below acceptable threshold',
                    'affected_documents': analytics.assessed_documents
                })

            # Check for high violation counts
            if analytics.total_violations > 50:
                risk_areas.append({
                    'type': 'high_violation_volume',
                    'severity': 'medium',
                    'description': f'High volume of violations detected ({analytics.total_violations} in 30 days)',
                    'affected_documents': len(set(v.doc_id for v in self.data_manager.load_violations()))
                })

            # Check for categories with low compliance
            for category, score in analytics.compliance_by_category.items():
                if score < 60:
                    risk_areas.append({
                        'type': 'category_risk',
                        'severity': 'high',
                        'description': f'Low compliance in {category} category (score: {score:.1f}%)',
                        'affected_category': category
                    })

            # Check for systems with low compliance
            for system, score in analytics.compliance_by_system.items():
                if score < 60:
                    risk_areas.append({
                        'type': 'system_risk',
                        'severity': 'high',
                        'description': f'Low compliance for {system} system (score: {score:.1f}%)',
                        'affected_system': system
                    })

            # Check for overdue violations
            if analytics.overdue_violations > 0:
                risk_areas.append({
                    'type': 'overdue_remediation',
                    'severity': 'medium',
                    'description': f'{analytics.overdue_violations} violations are past their remediation deadline',
                    'affected_violations': analytics.overdue_violations
                })

            # Sort by severity
            severity_order = {'high': 0, 'medium': 1, 'low': 2}
            risk_areas.sort(key=lambda x: severity_order.get(x['severity'], 3))

            return {
                'total_risk_areas': len(risk_areas),
                'risk_areas': risk_areas,
                'overall_risk_level': self._calculate_overall_risk_level(risk_areas),
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error identifying risk areas: {e}")
            return {'status': 'error', 'message': str(e)}

    def _calculate_overall_risk_level(self, risk_areas: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level based on identified risk areas"""
        if not risk_areas:
            return 'low'

        high_risks = len([r for r in risk_areas if r['severity'] == 'high'])
        medium_risks = len([r for r in risk_areas if r['severity'] == 'medium'])

        if high_risks > 0:
            return 'high'
        elif medium_risks > 2:
            return 'medium'
        else:
            return 'low'

    def generate_insights_report(self) -> Dict[str, Any]:
        """Generate an insights report with key findings and recommendations"""
        try:
            # Get comprehensive analytics
            analytics_30 = self.generate_compliance_analytics(30)
            analytics_90 = self.generate_compliance_analytics(90)

            if not analytics_30 or not analytics_90:
                return {'status': 'no_data'}

            # Get trend analysis
            trends = self.trend_analyzer.analyze_compliance_trends(90)

            # Get risk areas
            risk_areas = self.identify_risk_areas()

            # Generate insights
            insights = []

            # Trend insights
            if trends['trend_direction'] == 'improving':
                insights.append({
                    'type': 'positive_trend',
                    'message': 'Compliance scores are improving over time',
                    'significance': 'positive'
                })
            elif trends['trend_direction'] == 'declining':
                insights.append({
                    'type': 'negative_trend',
                    'message': 'Compliance scores are declining - immediate attention required',
                    'significance': 'negative'
                })

            # Volume insights
            if analytics_30.total_violations > analytics_90.total_violations * 0.4:  # 40% of 90-day total in last 30 days
                insights.append({
                    'type': 'high_violation_rate',
                    'message': 'Violation rate is higher than normal - increased monitoring recommended',
                    'significance': 'negative'
                })

            # Risk insights
            if risk_areas['total_risk_areas'] > 5:
                insights.append({
                    'type': 'multiple_risk_areas',
                    'message': f'Multiple risk areas identified ({risk_areas["total_risk_areas"]}) - comprehensive review recommended',
                    'significance': 'negative'
                })

            # Generate recommendations
            recommendations = []

            if trends['trend_direction'] == 'declining':
                recommendations.append('Implement immediate corrective actions to reverse declining compliance trend')
                recommendations.append('Increase frequency of compliance assessments and monitoring')

            if risk_areas['total_risk_areas'] > 0:
                recommendations.append('Focus remediation efforts on identified high-risk areas')
                recommendations.append('Allocate additional resources to address compliance gaps')

            if analytics_30.avg_compliance_score < 80:
                recommendations.append('Enhance compliance training and awareness programs')
                recommendations.append('Review and update compliance requirements and procedures')

            return {
                'generated_at': datetime.now().isoformat(),
                'analysis_period_days': 90,
                'key_insights': insights,
                'recommendations': recommendations,
                'risk_summary': risk_areas,
                'trend_summary': trends,
                'metrics_summary': {
                    'compliance_rate': analytics_30.compliance_rate,
                    'total_violations': analytics_30.total_violations,
                    'avg_resolution_time': self._get_average_resolution_time()
                }
            }

        except Exception as e:
            logger.error(f"Error generating insights report: {e}")
            return {'status': 'error', 'message': str(e)}

    def _get_average_resolution_time(self) -> float:
        """Get average time to resolve violations"""
        try:
            violations = self.data_manager.load_violations(status='resolved')
            if not violations:
                return 0.0

            resolution_times = []
            for violation in violations:
                if violation.resolved_at and violation.created_at:
                    days = (violation.resolved_at - violation.created_at).days
                    resolution_times.append(days)

            return statistics.mean(resolution_times) if resolution_times else 0.0

        except:
            return 0.0


class PredictiveAnalytics:
    """Predictive analytics for compliance management"""

    def __init__(self):
        self.analytics_engine = ComplianceAnalyticsEngine()

    def predict_violation_risk(self, doc_id: str) -> Dict[str, Any]:
        """Predict the risk of future violations for a document"""
        try:
            # Get document compliance history
            assessments = self.analytics_engine.data_manager.load_assessments(doc_id)

            if not assessments:
                return {
                    'doc_id': doc_id,
                    'risk_level': 'unknown',
                    'confidence': 0.0,
                    'reasoning': 'No compliance history available'
                }

            # Analyze compliance pattern
            scores = [a.compliance_score for a in assessments]
            avg_score = statistics.mean(scores)
            score_volatility = statistics.stdev(scores) if len(scores) > 1 else 0

            # Simple risk calculation
            base_risk = (100 - avg_score) / 100  # Higher risk for lower scores
            volatility_risk = min(score_volatility / 50, 0.3)  # Volatility adds up to 30% risk
            recency_risk = 0.1 if (datetime.now() - assessments[-1].assessed_at).days > 30 else 0

            total_risk = base_risk + volatility_risk + recency_risk
            total_risk = min(total_risk, 1.0)  # Cap at 100%

            # Determine risk level
            if total_risk > 0.7:
                risk_level = 'high'
            elif total_risk > 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            # Calculate confidence based on data quality
            confidence = min(len(assessments) / 10, 1.0)  # More assessments = higher confidence

            return {
                'doc_id': doc_id,
                'risk_score': total_risk,
                'risk_level': risk_level,
                'confidence': confidence,
                'factors': {
                    'base_risk': base_risk,
                    'volatility_risk': volatility_risk,
                    'recency_risk': recency_risk
                },
                'recommendations': self._get_risk_recommendations(risk_level, total_risk),
                'predicted_next_assessment': self._predict_next_assessment_date(assessments)
            }

        except Exception as e:
            logger.error(f"Error predicting violation risk for {doc_id}: {e}")
            return {
                'doc_id': doc_id,
                'risk_level': 'error',
                'confidence': 0.0,
                'reasoning': str(e)
            }

    def _get_risk_recommendations(self, risk_level: str, risk_score: float) -> List[str]:
        """Get recommendations based on risk level"""
        recommendations = []

        if risk_level == 'high':
            recommendations.append('Schedule immediate compliance assessment')
            recommendations.append('Implement enhanced monitoring for this document')
            recommendations.append('Review document for potential compliance issues')
        elif risk_level == 'medium':
            recommendations.append('Schedule compliance assessment within next 30 days')
            recommendations.append('Monitor document for compliance changes')
        else:
            recommendations.append('Continue regular compliance monitoring')

        return recommendations

    def _predict_next_assessment_date(self, assessments: List[ComplianceAssessment]) -> str:
        """Predict when the next assessment should occur"""
        if not assessments:
            return (datetime.now() + timedelta(days=30)).isoformat()

        # Base next assessment on document risk and compliance history
        latest_assessment = max(assessments, key=lambda a: a.assessed_at)
        days_since_last = (datetime.now() - latest_assessment.assessed_at).days

        # If document has low compliance scores, assess more frequently
        recent_scores = [a.compliance_score for a in assessments[-3:]]  # Last 3 assessments
        avg_recent_score = statistics.mean(recent_scores) if recent_scores else 100

        if avg_recent_score < 70:
            next_assessment_days = 14  # Every 2 weeks
        elif avg_recent_score < 85:
            next_assessment_days = 30  # Monthly
        else:
            next_assessment_days = 90  # Quarterly

        # Don't assess more frequently than every 7 days
        next_assessment_days = max(next_assessment_days, 7)

        predicted_date = datetime.now() + timedelta(days=next_assessment_days)
        return predicted_date.isoformat()