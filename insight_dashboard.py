"""
Insight Dashboard with Trends and Analytics Visualization
Creates interactive dashboards for document insights and analytics
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Custom imports
from .insight_extractor import insight_extractor
from .content_categorizer import content_categorizer
from .relationship_analyzer import relationship_analyzer

logger = logging.getLogger(__name__)


class InsightAnalytics:
    """Analytics engine for insights and trends"""

    def __init__(self):
        self.insight_history = []
        self.category_trends = defaultdict(list)
        self.relationship_trends = defaultdict(list)

    def record_analysis(self, analysis_results: Dict[str, Any]):
        """Record analysis results for trend tracking"""
        timestamp = datetime.now()

        # Extract key metrics
        metrics = {
            'timestamp': timestamp,
            'total_insights': len(analysis_results.get('insights', [])),
            'total_relationships': len(analysis_results.get('relationships', [])),
            'total_gaps': len(analysis_results.get('knowledge_gaps', [])),
            'categories_found': len(analysis_results.get('insights_by_type', {})),
            'processing_time': analysis_results.get('metadata', {}).get('processing_time', 0)
        }

        self.insight_history.append(metrics)

        # Keep only last 100 records
        if len(self.insight_history) > 100:
            self.insight_history = self.insight_history[-100:]

    def get_trends_data(self, days: int = 30) -> Dict[str, Any]:
        """Get trend data for the specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_data = [m for m in self.insight_history if m['timestamp'] >= cutoff_date]

        if not recent_data:
            return {'error': 'No data available for the specified period'}

        # Convert to DataFrame for analysis
        df = pd.DataFrame(recent_data)

        trends = {
            'insight_volume': self._analyze_insight_volume_trend(df),
            'category_evolution': self._analyze_category_evolution(df),
            'processing_efficiency': self._analyze_processing_efficiency(df),
            'gap_analysis': self._analyze_gap_trends(df)
        }

        return trends

    def _analyze_insight_volume_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in insight volume"""
        return {
            'total_insights': df['total_insights'].sum(),
            'average_insights': df['total_insights'].mean(),
            'trend_direction': 'increasing' if df['total_insights'].iloc[-1] > df['total_insights'].iloc[0] else 'decreasing',
            'volatility': df['total_insights'].std()
        }

    def _analyze_category_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how categories evolve over time"""
        # This would track category distribution changes
        return {
            'categories_tracked': len(df.columns) - 5,  # Approximate
            'stability_score': 0.8  # Placeholder
        }

    def _analyze_processing_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze processing efficiency trends"""
        avg_time = df['processing_time'].mean()
        return {
            'average_processing_time': avg_time,
            'efficiency_trend': 'improving' if df['processing_time'].iloc[-1] < avg_time else 'degrading'
        }

    def _analyze_gap_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze knowledge gap trends"""
        return {
            'total_gaps': df['total_gaps'].sum(),
            'gap_reduction_rate': df['total_gaps'].diff().mean()
        }


class DashboardGenerator:
    """Generates interactive dashboard visualizations"""

    def __init__(self):
        self.analytics = InsightAnalytics()
        self.color_scheme = px.colors.qualitative.Set3

    def generate_insight_dashboard(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive insight dashboard

        Args:
            analysis_results: Results from insight extraction and analysis

        Returns:
            Dictionary containing dashboard data and visualizations
        """
        try:
            # Record for analytics
            self.analytics.record_analysis(analysis_results)

            # Generate different dashboard sections
            overview_section = self._generate_overview_section(analysis_results)
            insights_section = self._generate_insights_section(analysis_results)
            relationships_section = self._generate_relationships_section(analysis_results)
            gaps_section = self._generate_gaps_section(analysis_results)
            trends_section = self._generate_trends_section()

            # Combine all sections
            dashboard = {
                'title': 'Document Insights Dashboard',
                'generated_at': datetime.now().isoformat(),
                'sections': {
                    'overview': overview_section,
                    'insights': insights_section,
                    'relationships': relationships_section,
                    'knowledge_gaps': gaps_section,
                    'trends': trends_section
                },
                'summary': self._generate_dashboard_summary(analysis_results)
            }

            return dashboard

        except Exception as e:
            logger.error(f"Dashboard generation error: {e}")
            return {'error': str(e)}

    def _generate_overview_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overview section with key metrics"""
        insights = results.get('insights', [])
        relationships = results.get('relationships', [])
        gaps = results.get('knowledge_gaps', [])

        # Key metrics
        metrics = [
            {
                'title': 'Total Insights',
                'value': len(insights),
                'change': '+12%',  # Placeholder
                'trend': 'up',
                'icon': 'lightbulb'
            },
            {
                'title': 'Relationships',
                'value': len(relationships),
                'change': '+8%',  # Placeholder
                'trend': 'up',
                'icon': 'link'
            },
            {
                'title': 'Knowledge Gaps',
                'value': len(gaps),
                'change': '-5%',  # Placeholder
                'trend': 'down',
                'icon': 'alert-triangle'
            },
            {
                'title': 'Categories',
                'value': len(results.get('insights_by_type', {})),
                'change': '+3',  # Placeholder
                'trend': 'up',
                'icon': 'folder'
            }
        ]

        # Generate insights distribution chart
        insights_dist = self._create_insights_distribution_chart(results)

        # Generate gap severity chart
        gap_severity = self._create_gap_severity_chart(gaps)

        return {
            'metrics': metrics,
            'charts': {
                'insights_distribution': insights_dist,
                'gap_severity': gap_severity
            }
        }

    def _generate_insights_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights analysis section"""
        insights = results.get('insights', [])
        insights_by_type = results.get('insights_by_type', {})

        # Insights by type chart
        type_chart = self._create_insights_by_type_chart(insights_by_type)

        # Top insights list
        top_insights = self._get_top_insights(insights, limit=10)

        # Confidence distribution
        confidence_chart = self._create_confidence_distribution_chart(insights)

        return {
            'charts': {
                'insights_by_type': type_chart,
                'confidence_distribution': confidence_chart
            },
            'top_insights': top_insights,
            'insights_by_type': dict(insights_by_type)
        }

    def _generate_relationships_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate relationships analysis section"""
        relationships = results.get('relationships', [])

        # Relationship types chart
        rel_types_chart = self._create_relationship_types_chart(relationships)

        # Relationship strength distribution
        strength_chart = self._create_relationship_strength_chart(relationships)

        # Top relationships
        top_relationships = self._get_top_relationships(relationships, limit=10)

        return {
            'charts': {
                'relationship_types': rel_types_chart,
                'relationship_strength': strength_chart
            },
            'top_relationships': top_relationships,
            'total_relationships': len(relationships)
        }

    def _generate_gaps_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate knowledge gaps section"""
        gaps = results.get('knowledge_gaps', [])

        # Gaps by type chart
        gaps_chart = self._create_gaps_by_type_chart(gaps)

        # Gaps by severity
        severity_chart = self._create_gaps_by_severity_chart(gaps)

        # Priority gaps (high and critical severity)
        priority_gaps = [gap for gap in gaps if gap.severity in ['high', 'critical']]

        return {
            'charts': {
                'gaps_by_type': gaps_chart,
                'gaps_by_severity': severity_chart
            },
            'priority_gaps': priority_gaps[:10],  # Top 10 priority gaps
            'total_gaps': len(gaps)
        }

    def _generate_trends_section(self) -> Dict[str, Any]:
        """Generate trends and analytics section"""
        trends_data = self.analytics.get_trends_data()

        if 'error' in trends_data:
            return {'error': trends_data['error']}

        # Trends charts
        volume_chart = self._create_volume_trends_chart(trends_data)
        efficiency_chart = self._create_efficiency_trends_chart(trends_data)

        return {
            'charts': {
                'insight_volume_trends': volume_chart,
                'processing_efficiency': efficiency_chart
            },
            'trends_summary': trends_data
        }

    def _create_insights_distribution_chart(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create insights distribution visualization"""
        insights_by_type = results.get('insights_by_type', {})

        if not insights_by_type:
            return {'error': 'No insights data available'}

        # Prepare data
        types = list(insights_by_type.keys())
        counts = [len(insights) for insights in insights_by_type.values()]

        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=types,
            values=counts,
            marker_colors=self.color_scheme[:len(types)]
        )])

        fig.update_layout(
            title='Insights Distribution by Type',
            showlegend=True,
            height=400
        )

        return {
            'plotly_figure': fig.to_dict(),
            'chart_type': 'pie',
            'title': 'Insights Distribution'
        }

    def _create_gap_severity_chart(self, gaps: List) -> Dict[str, Any]:
        """Create gap severity visualization"""
        if not gaps:
            return {'error': 'No gaps data available'}

        # Count by severity
        severity_counts = Counter(gap.severity for gap in gaps)

        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())

        # Create bar chart
        fig = go.Figure(data=[go.Bar(
            x=severities,
            y=counts,
            marker_color=self.color_scheme[0]
        )])

        fig.update_layout(
            title='Knowledge Gaps by Severity',
            xaxis_title='Severity',
            yaxis_title='Count',
            height=400
        )

        return {
            'plotly_figure': fig.to_dict(),
            'chart_type': 'bar',
            'title': 'Gap Severity Distribution'
        }

    def _create_insights_by_type_chart(self, insights_by_type: Dict) -> Dict[str, Any]:
        """Create insights by type visualization"""
        if not insights_by_type:
            return {'error': 'No insights data available'}

        types = list(insights_by_type.keys())
        counts = [len(insights) for insights in insights_by_type.values()]

        # Create horizontal bar chart
        fig = go.Figure(data=[go.Bar(
            y=types,
            x=counts,
            orientation='h',
            marker_color=self.color_scheme[:len(types)]
        )])

        fig.update_layout(
            title='Insights by Type',
            xaxis_title='Count',
            yaxis_title='Insight Type',
            height=400
        )

        return {
            'plotly_figure': fig.to_dict(),
            'chart_type': 'bar_horizontal',
            'title': 'Insights by Type'
        }

    def _create_confidence_distribution_chart(self, insights: List) -> Dict[str, Any]:
        """Create confidence distribution visualization"""
        if not insights:
            return {'error': 'No insights data available'}

        confidence_counts = Counter(insight.confidence.value for insight in insights)

        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=list(confidence_counts.keys()),
            values=list(confidence_counts.values()),
            hole=0.4,
            marker_colors=self.color_scheme[:3]
        )])

        fig.update_layout(
            title='Insight Confidence Distribution',
            height=400
        )

        return {
            'plotly_figure': fig.to_dict(),
            'chart_type': 'donut',
            'title': 'Confidence Distribution'
        }

    def _create_relationship_types_chart(self, relationships: List) -> Dict[str, Any]:
        """Create relationship types visualization"""
        if not relationships:
            return {'error': 'No relationships data available'}

        rel_types = Counter(rel.relationship_type for rel in relationships)

        # Create treemap
        fig = go.Figure(data=[go.Treemap(
            labels=list(rel_types.keys()),
            values=list(rel_types.values()),
            textinfo="label+value+percent root"
        )])

        fig.update_layout(
            title='Relationship Types Distribution',
            height=400
        )

        return {
            'plotly_figure': fig.to_dict(),
            'chart_type': 'treemap',
            'title': 'Relationship Types'
        }

    def _create_relationship_strength_chart(self, relationships: List) -> Dict[str, Any]:
        """Create relationship strength distribution"""
        if not relationships:
            return {'error': 'No relationships data available'}

        # Create histogram of relationship strengths
        strengths = [rel.strength for rel in relationships]

        fig = go.Figure(data=[go.Histogram(
            x=strengths,
            nbinsx=20,
            marker_color=self.color_scheme[1]
        )])

        fig.update_layout(
            title='Relationship Strength Distribution',
            xaxis_title='Strength',
            yaxis_title='Frequency',
            height=400
        )

        return {
            'plotly_figure': fig.to_dict(),
            'chart_type': 'histogram',
            'title': 'Relationship Strength'
        }

    def _create_gaps_by_type_chart(self, gaps: List) -> Dict[str, Any]:
        """Create gaps by type visualization"""
        if not gaps:
            return {'error': 'No gaps data available'}

        gap_types = Counter(gap.gap_type for gap in gaps)

        # Create funnel chart
        fig = go.Figure(data=[go.Funnel(
            y=list(gap_types.keys()),
            x=list(gap_types.values()),
            marker_color=self.color_scheme[2]
        )])

        fig.update_layout(
            title='Knowledge Gaps by Type',
            height=400
        )

        return {
            'plotly_figure': fig.to_dict(),
            'chart_type': 'funnel',
            'title': 'Gaps by Type'
        }

    def _create_gaps_by_severity_chart(self, gaps: List) -> Dict[str, Any]:
        """Create gaps by severity visualization"""
        if not gaps:
            return {'error': 'No gaps data available'}

        severity_order = ['low', 'medium', 'high', 'critical']
        severity_counts = Counter(gap.severity for gap in gaps)

        # Sort by severity order
        sorted_data = [(sev, severity_counts.get(sev, 0)) for sev in severity_order]

        fig = go.Figure(data=[go.Bar(
            x=[item[0] for item in sorted_data],
            y=[item[1] for item in sorted_data],
            marker_color=[self.color_scheme[3] if sev in ['high', 'critical'] else self.color_scheme[0]
                         for sev in severity_order]
        )])

        fig.update_layout(
            title='Knowledge Gaps by Severity',
            xaxis_title='Severity',
            yaxis_title='Count',
            height=400
        )

        return {
            'plotly_figure': fig.to_dict(),
            'chart_type': 'bar',
            'title': 'Gaps by Severity'
        }

    def _create_volume_trends_chart(self, trends_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create insight volume trends visualization"""
        # This would create a time series chart
        # For now, return placeholder
        return {
            'chart_type': 'line',
            'title': 'Insight Volume Trends',
            'data_available': bool(trends_data)
        }

    def _create_efficiency_trends_chart(self, trends_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create processing efficiency trends visualization"""
        return {
            'chart_type': 'line',
            'title': 'Processing Efficiency Trends',
            'data_available': bool(trends_data)
        }

    def _get_top_insights(self, insights: List, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top insights by confidence and relevance"""
        # Sort by confidence and return top insights
        sorted_insights = sorted(
            insights,
            key=lambda x: (x.confidence.value, len(x.content)),
            reverse=True
        )

        return [
            {
                'type': insight.insight_type.value,
                'content': insight.content[:200] + '...' if len(insight.content) > 200 else insight.content,
                'confidence': insight.confidence.value,
                'context': insight.context[:100] + '...' if len(insight.context) > 100 else insight.context
            }
            for insight in sorted_insights[:limit]
        ]

    def _get_top_relationships(self, relationships: List, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top relationships by strength"""
        sorted_relationships = sorted(
            relationships,
            key=lambda x: x.strength,
            reverse=True
        )

        return [
            {
                'source_doc': rel.source_doc_id,
                'target_doc': rel.target_doc_id,
                'type': rel.relationship_type,
                'strength': rel.strength,
                'evidence': rel.evidence[:2]  # First 2 evidence items
            }
            for rel in sorted_relationships[:limit]
        ]

    def _generate_dashboard_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall dashboard summary"""
        insights = results.get('insights', [])
        relationships = results.get('relationships', [])
        gaps = results.get('knowledge_gaps', [])

        # Calculate summary statistics
        total_insights = len(insights)
        total_relationships = len(relationships)
        total_gaps = len(gaps)

        # Determine overall health score
        high_confidence_insights = len([i for i in insights if i.confidence.value == 'high'])
        critical_gaps = len([g for g in gaps if g.severity in ['critical', 'high']])

        # Simple health score calculation
        if total_insights > 0:
            quality_score = high_confidence_insights / total_insights
        else:
            quality_score = 0

        if total_gaps > 0:
            gap_penalty = critical_gaps / total_gaps
        else:
            gap_penalty = 0

        health_score = max(0, (quality_score * 100) - (gap_penalty * 50))

        # Generate insights
        summary_insights = []

        if total_insights > total_relationships:
            summary_insights.append("Document collection has rich insight content")
        elif total_relationships > total_insights:
            summary_insights.append("Strong interconnections between documents")

        if critical_gaps == 0:
            summary_insights.append("No critical knowledge gaps identified")
        elif critical_gaps > 0:
            summary_insights.append(f"{critical_gaps} critical knowledge gaps require attention")

        return {
            'total_insights': total_insights,
            'total_relationships': total_relationships,
            'total_gaps': total_gaps,
            'health_score': round(health_score, 1),
            'key_insights': summary_insights,
            'recommendations': self._generate_recommendations(results)
        }

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        gaps = results.get('knowledge_gaps', [])
        critical_gaps = [g for g in gaps if g.severity == 'critical']

        if critical_gaps:
            recommendations.append(f"Address {len(critical_gaps)} critical knowledge gaps immediately")

        # Check for missing document types
        insights_by_type = results.get('insights_by_type', {})
        if not insights_by_type.get('process', []):
            recommendations.append("Document key business processes")

        if not insights_by_type.get('requirement', []):
            recommendations.append("Create requirement specifications")

        # Check relationship density
        relationships = results.get('relationships', [])
        if len(relationships) < 5:
            recommendations.append("Analyze document relationships to identify integration opportunities")

        return recommendations


class InsightDashboardAPI:
    """API for accessing dashboard functionality"""

    def __init__(self):
        self.dashboard_generator = DashboardGenerator()

    def get_dashboard(self, document_paths: List[str]) -> Dict[str, Any]:
        """Generate dashboard for document collection"""
        # This would integrate with the analysis pipeline
        # For now, return a placeholder structure
        return {
            'dashboard': 'Dashboard generation API ready',
            'document_count': len(document_paths),
            'status': 'ready'
        }

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        return self.dashboard_generator.analytics.get_trends_data()


# Global dashboard instances
dashboard_generator = DashboardGenerator()
insight_dashboard_api = InsightDashboardAPI()