"""
Collaborative Dashboard and Reporting System
Real-time dashboard for team collaboration, workflow monitoring, and analytics
"""

import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid
from dataclasses import dataclass, field
from collections import defaultdict

# Import all collaborative components
from .collaborative_workflow import collaborative_manager
from .collaborative_editor import editor_manager
from .annotation_system import annotation_manager
from .notification_system import notification_manager
from .workflow_automation import automation_engine, task_manager


@dataclass
class DashboardWidget:
    """Represents a dashboard widget"""
    widget_id: str
    title: str
    type: str  # chart, metric, table, activity_feed, etc.
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 300  # seconds
    is_visible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'widget_id': self.widget_id,
            'title': self.title,
            'type': self.type,
            'position': self.position,
            'config': self.config,
            'refresh_interval': self.refresh_interval,
            'is_visible': self.is_visible
        }


@dataclass
class Dashboard:
    """Represents a user dashboard"""
    dashboard_id: str
    user_id: str
    title: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dashboard_id': self.dashboard_id,
            'user_id': self.user_id,
            'title': self.title,
            'widgets': [w.to_dict() for w in self.widgets],
            'layout': self.layout,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class CollaborativeDashboardManager:
    """Manager for collaborative dashboards and reporting"""

    def __init__(self):
        self.dashboards: Dict[str, Dashboard] = {}
        self.widget_data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # Threading lock
        self._lock = threading.Lock()

        # Cache settings
        self.cache_ttl = 300  # 5 minutes

        # Start background refresh
        self._start_refresh_task()

    def _start_refresh_task(self):
        """Start background widget refresh task"""
        def refresh_cache():
            while True:
                try:
                    self._refresh_widget_cache()
                    time.sleep(60)  # Refresh every minute
                except Exception as e:
                    print(f"Error refreshing dashboard cache: {e}")
                    time.sleep(300)

        import time
        refresh_thread = threading.Thread(target=refresh_cache, daemon=True)
        refresh_thread.start()

    def _refresh_widget_cache(self):
        """Refresh cached widget data"""
        with self._lock:
            current_time = datetime.now()
            expired_keys = []

            # Find expired cache entries
            for key, timestamp in self.cache_timestamps.items():
                if current_time - timestamp > timedelta(seconds=self.cache_ttl):
                    expired_keys.append(key)

            # Remove expired entries
            for key in expired_keys:
                self.cache_timestamps.pop(key, None)
                self.widget_data_cache.pop(key, None)

    def create_dashboard(self, user_id: str, title: str = "My Dashboard") -> Dashboard:
        """Create a new dashboard for a user"""
        with self._lock:
            dashboard = Dashboard(
                dashboard_id=str(uuid.uuid4()),
                user_id=user_id,
                title=title,
                widgets=self._get_default_widgets()
            )

            self.dashboards[dashboard.dashboard_id] = dashboard
            return dashboard

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get a dashboard by ID"""
        with self._lock:
            return self.dashboards.get(dashboard_id)

    def get_user_dashboard(self, user_id: str) -> Optional[Dashboard]:
        """Get the primary dashboard for a user"""
        with self._lock:
            # Find user's dashboard or create default
            user_dashboards = [
                d for d in self.dashboards.values() if d.user_id == user_id
            ]

            if user_dashboards:
                return user_dashboards[0]

            # Create default dashboard
            return self.create_dashboard(user_id)

    def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any]) -> bool:
        """Update dashboard configuration"""
        with self._lock:
            if dashboard_id not in self.dashboards:
                return False

            dashboard = self.dashboards[dashboard_id]
            dashboard.updated_at = datetime.now()

            # Apply updates
            for key, value in updates.items():
                if key == 'title':
                    dashboard.title = value
                elif key == 'widgets':
                    dashboard.widgets = value
                elif key == 'layout':
                    dashboard.layout = value

            return True

    def get_widget_data(self, widget: DashboardWidget, user_id: str) -> Dict[str, Any]:
        """Get data for a specific widget"""
        cache_key = f"{widget.widget_id}_{user_id}"

        with self._lock:
            # Check cache first
            if (cache_key in self.widget_data_cache and
                cache_key in self.cache_timestamps):

                cache_time = self.cache_timestamps[cache_key]
                if datetime.now() - cache_time < timedelta(seconds=widget.refresh_interval):
                    return self.widget_data_cache[cache_key]

        # Generate fresh data
        data = self._generate_widget_data(widget, user_id)

        # Cache the data
        with self._lock:
            self.widget_data_cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now()

        return data

    def _generate_widget_data(self, widget: DashboardWidget, user_id: str) -> Dict[str, Any]:
        """Generate data for a widget"""
        widget_type = widget.type

        if widget_type == 'collaboration_overview':
            return self._get_collaboration_overview(user_id)
        elif widget_type == 'active_workflows':
            return self._get_active_workflows(user_id)
        elif widget_type == 'document_activity':
            return self._get_document_activity(user_id)
        elif widget_type == 'team_status':
            return self._get_team_status(user_id)
        elif widget_type == 'notification_summary':
            return self._get_notification_summary(user_id)
        elif widget_type == 'task_summary':
            return self._get_task_summary(user_id)
        elif widget_type == 'compliance_status':
            return self._get_compliance_status(user_id)
        elif widget_type == 'activity_feed':
            return self._get_activity_feed(user_id, limit=widget.config.get('limit', 20))
        elif widget_type == 'chart':
            return self._get_chart_data(widget.config, user_id)
        elif widget_type == 'metric':
            return self._get_metric_data(widget.config, user_id)

        return {'error': 'Unknown widget type'}

    def _get_default_widgets(self) -> List[DashboardWidget]:
        """Get default widgets for new dashboards"""
        return [
            DashboardWidget(
                widget_id=str(uuid.uuid4()),
                title='Collaboration Overview',
                type='collaboration_overview',
                position={'x': 0, 'y': 0, 'width': 6, 'height': 3},
                refresh_interval=60
            ),
            DashboardWidget(
                widget_id=str(uuid.uuid4()),
                title='Active Workflows',
                type='active_workflows',
                position={'x': 6, 'y': 0, 'width': 6, 'height': 3},
                refresh_interval=120
            ),
            DashboardWidget(
                widget_id=str(uuid.uuid4()),
                title='Document Activity',
                type='document_activity',
                position={'x': 0, 'y': 3, 'width': 8, 'height': 4},
                refresh_interval=300
            ),
            DashboardWidget(
                widget_id=str(uuid.uuid4()),
                title='Team Status',
                type='team_status',
                position={'x': 8, 'y': 3, 'width': 4, 'height': 4},
                refresh_interval=60
            ),
            DashboardWidget(
                widget_id=str(uuid.uuid4()),
                title='Activity Feed',
                type='activity_feed',
                position={'x': 0, 'y': 7, 'width': 12, 'height': 4},
                config={'limit': 15},
                refresh_interval=30
            )
        ]

    def _get_collaboration_overview(self, user_id: str) -> Dict[str, Any]:
        """Get collaboration overview data"""
        # Get user's active documents
        user_documents = []
        for doc_id, collaborators in collaborative_manager.document_collaborators.items():
            if user_id in collaborators:
                user_documents.append(doc_id)

        # Get collaboration statistics
        stats = collaborative_manager.get_collaboration_stats()

        # Get recent activity for user's documents
        recent_activity = []
        for event in collaborative_manager.events[-50:]:  # Last 50 events
            if event.document_id in user_documents:
                recent_activity.append(event.to_dict())

        return {
            'user_documents': len(user_documents),
            'total_collaborators': stats.get('total_collaborators', 0),
            'recent_activity': len(recent_activity),
            'online_users': stats.get('online_users', 0),
            'recent_events': recent_activity[:10]
        }

    def _get_active_workflows(self, user_id: str) -> Dict[str, Any]:
        """Get active workflows data"""
        # Get workflows where user is involved
        user_workflows = []

        for workflow in collaborative_manager.workflows.values():
            if (workflow.created_by == user_id or
                user_id in workflow.reviewers or
                user_id in workflow.approvers):

                workflow_data = workflow.to_dict()
                workflow_data['is_overdue'] = (
                    workflow.due_date and
                    workflow.due_date < datetime.now() and
                    workflow.status.value != 'completed'
                )

                user_workflows.append(workflow_data)

        # Sort by due date and status
        user_workflows.sort(key=lambda x: (
            0 if x['is_overdue'] else 1,  # Overdue first
            x['due_date'] or '',
            x['status']
        ))

        # Get workflow statistics
        total_workflows = len(user_workflows)
        overdue_workflows = len([w for w in user_workflows if w['is_overdue']])
        in_review_workflows = len([w for w in user_workflows if w['status'] == 'in_review'])

        return {
            'workflows': user_workflows[:20],  # Show recent 20
            'total_workflows': total_workflows,
            'overdue_workflows': overdue_workflows,
            'in_review_workflows': in_review_workflows,
            'completed_this_week': len([
                w for w in collaborative_manager.workflows.values()
                if (w.created_by == user_id and
                    w.completed_at and
                    w.completed_at > datetime.now() - timedelta(days=7))
            ])
        }

    def _get_document_activity(self, user_id: str) -> Dict[str, Any]:
        """Get document activity data"""
        # Get documents with recent activity
        document_activity = []

        for doc_id in collaborative_manager.document_collaborators:
            if user_id in collaborative_manager.document_collaborators[doc_id]:
                # Get recent events for this document
                recent_events = [
                    e for e in collaborative_manager.events
                    if e.document_id == doc_id and e.timestamp > datetime.now() - timedelta(days=7)
                ]

                if recent_events:
                    collaborators = collaborative_manager.get_document_collaborators(doc_id)
                    latest_event = max(recent_events, key=lambda x: x.timestamp)

                    document_activity.append({
                        'document_id': doc_id,
                        'collaborators': len(collaborators),
                        'recent_events': len(recent_events),
                        'latest_activity': latest_event.timestamp.isoformat(),
                        'activity_type': latest_event.event_type.value
                    })

        # Sort by recent activity
        document_activity.sort(key=lambda x: x['latest_activity'], reverse=True)

        # Get activity trends (last 7 days)
        activity_trends = self._get_activity_trends(user_id)

        return {
            'document_activity': document_activity[:15],
            'total_active_documents': len(document_activity),
            'activity_trends': activity_trends
        }

    def _get_team_status(self, user_id: str) -> Dict[str, Any]:
        """Get team status data"""
        # Get all collaborators user has worked with recently
        recent_collaborators = set()

        for event in collaborative_manager.events[-100:]:  # Last 100 events
            if (event.document_id in collaborative_manager.document_collaborators and
                user_id in collaborative_manager.document_collaborators[event.document_id]):

                recent_collaborators.add(event.user_id)

        # Get team member status
        team_members = []
        for collaborator_id in recent_collaborators:
            if collaborator_id in collaborative_manager.users:
                user = collaborative_manager.users[collaborator_id]

                # Count recent interactions
                interaction_count = len([
                    e for e in collaborative_manager.events[-200:]
                    if (e.user_id == collaborator_id and
                        e.document_id in collaborative_manager.document_collaborators.get(user_id, set()))
                ])

                team_members.append({
                    'user_id': user.user_id,
                    'username': user.username,
                    'role': user.role.value,
                    'is_online': user.is_online,
                    'last_seen': user.last_seen.isoformat(),
                    'interaction_count': interaction_count,
                    'current_document': user.current_document
                })

        # Sort by recent interaction and online status
        team_members.sort(key=lambda x: (
            0 if x['is_online'] else 1,  # Online first
            -x['interaction_count']  # Most interactions first
        ))

        return {
            'team_members': team_members[:10],
            'online_members': len([m for m in team_members if m['is_online']]),
            'total_collaborators': len(recent_collaborators),
            'recent_interactions': sum(m['interaction_count'] for m in team_members)
        }

    def _get_notification_summary(self, user_id: str) -> Dict[str, Any]:
        """Get notification summary data"""
        notifications = notification_manager.get_user_notifications(user_id, limit=50)
        unread_count = notification_manager.get_unread_count(user_id)

        # Group by type
        notifications_by_type = {}
        for notification in notifications:
            type_key = notification.type.value
            notifications_by_type[type_key] = notifications_by_type.get(type_key, 0) + 1

        # Recent notifications (last 7 days)
        recent_notifications = [
            n for n in notifications
            if n.created_at > datetime.now() - timedelta(days=7)
        ]

        return {
            'unread_count': unread_count,
            'total_notifications': len(notifications),
            'notifications_by_type': notifications_by_type,
            'recent_notifications': len(recent_notifications),
            'latest_notifications': [
                {
                    'type': n.type.value,
                    'title': n.title,
                    'created_at': n.created_at.isoformat(),
                    'is_read': n.read_at is not None
                }
                for n in notifications[:5]
            ]
        }

    def _get_task_summary(self, user_id: str) -> Dict[str, Any]:
        """Get task summary data"""
        # Get tasks assigned to user
        user_tasks = [
            t for t in collaborative_manager.tasks.values()
            if user_id in t.assigned_to
        ]

        # Task statistics
        total_tasks = len(user_tasks)
        pending_tasks = len([t for t in user_tasks if t.status == 'pending'])
        in_progress_tasks = len([t for t in user_tasks if t.status == 'in_progress'])
        completed_tasks = len([t for t in user_tasks if t.status == 'completed'])

        # Overdue tasks
        overdue_tasks = [
            t for t in user_tasks
            if (t.due_date and t.due_date < datetime.now() and t.status != 'completed')
        ]

        # Priority breakdown
        tasks_by_priority = {}
        for task in user_tasks:
            priority = task.priority
            tasks_by_priority[priority] = tasks_by_priority.get(priority, 0) + 1

        return {
            'total_tasks': total_tasks,
            'pending_tasks': pending_tasks,
            'in_progress_tasks': in_progress_tasks,
            'completed_tasks': completed_tasks,
            'overdue_tasks': len(overdue_tasks),
            'tasks_by_priority': tasks_by_priority,
            'recent_tasks': [
                {
                    'task_id': t.task_id,
                    'title': t.title,
                    'status': t.status,
                    'priority': t.priority,
                    'due_date': t.due_date.isoformat() if t.due_date else None,
                    'is_overdue': t.due_date and t.due_date < datetime.now() and t.status != 'completed'
                }
                for t in sorted(user_tasks, key=lambda x: x.created_at, reverse=True)[:10]
            ]
        }

    def _get_compliance_status(self, user_id: str) -> Dict[str, Any]:
        """Get compliance status data"""
        # This would integrate with the compliance system
        # For now, provide mock data structure

        return {
            'overall_score': 85,
            'total_documents': 1250,
            'compliant_documents': 1060,
            'non_compliant_documents': 190,
            'critical_issues': 12,
            'pending_reviews': 45,
            'recent_assessments': [
                {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 'score': 80 + i*2}
                for i in range(7)
            ]
        }

    def _get_activity_feed(self, user_id: str, limit: int = 20) -> Dict[str, Any]:
        """Get activity feed data"""
        # Get notifications and events for user
        notifications = notification_manager.get_user_notifications(user_id, limit=limit//2)
        events = notification_manager.get_activity_feed(user_id, limit=limit//2)

        # Combine and sort
        combined_activity = []

        for notification in notifications:
            combined_activity.append({
                'type': 'notification',
                'id': notification.notification_id,
                'title': notification.title,
                'message': notification.message,
                'timestamp': notification.created_at.isoformat(),
                'is_read': notification.read_at is not None
            })

        for event in events:
            combined_activity.append({
                'type': 'event',
                'id': event.get('notification_id', ''),
                'title': event.get('title', 'Activity'),
                'message': f"{event.get('type', 'unknown').replace('_', ' ').title()}",
                'timestamp': event.get('timestamp', ''),
                'is_read': True  # Events are always "read"
            })

        # Sort by timestamp
        combined_activity.sort(key=lambda x: x['timestamp'], reverse=True)

        return {
            'activities': combined_activity[:limit],
            'total_activities': len(combined_activity)
        }

    def _get_chart_data(self, config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Get chart data"""
        chart_type = config.get('chart_type', 'line')
        data_source = config.get('data_source', 'activity_trends')

        if data_source == 'activity_trends':
            return {
                'type': 'line',
                'labels': [(datetime.now() - timedelta(days=i)).strftime('%m/%d')
                          for i in range(7)][::-1],
                'datasets': [{
                    'label': 'Daily Activity',
                    'data': [10 + i*5 for i in range(7)]  # Mock data
                }]
            }
        elif data_source == 'workflow_status':
            return {
                'type': 'doughnut',
                'labels': ['Draft', 'In Review', 'Approved', 'Rejected'],
                'datasets': [{
                    'data': [15, 25, 45, 5]  # Mock data
                }]
            }

        return {'error': 'Unknown data source'}

    def _get_metric_data(self, config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Get metric data"""
        metric_type = config.get('metric_type', 'collaboration_score')

        if metric_type == 'collaboration_score':
            # Calculate user's collaboration score
            return {
                'value': 87,
                'unit': 'points',
                'trend': '+5',
                'trend_direction': 'up',
                'description': 'Overall collaboration engagement'
            }
        elif metric_type == 'productivity_score':
            return {
                'value': 92,
                'unit': 'points',
                'trend': '+3',
                'trend_direction': 'up',
                'description': 'Task completion and efficiency'
            }

        return {'error': 'Unknown metric type'}

    def _get_activity_trends(self, user_id: str) -> List[Dict[str, Any]]:
        """Get activity trends for the last 7 days"""
        trends = []

        for i in range(7):
            date = datetime.now() - timedelta(days=i)

            # Count events for this day
            day_events = [
                e for e in collaborative_manager.events
                if (e.timestamp.date() == date.date() and
                    e.document_id in collaborative_manager.document_collaborators.get(user_id, set()))
            ]

            trends.append({
                'date': date.strftime('%Y-%m-%d'),
                'activity_count': len(day_events),
                'document_edits': len([e for e in day_events if e.event_type.value == 'document_edit']),
                'comments': len([e for e in day_events if e.event_type.value == 'comment_added'])
            })

        return trends[::-1]  # Reverse to show oldest first

    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard statistics"""
        return {
            'total_dashboards': len(self.dashboards),
            'total_widgets': sum(len(d.widgets) for d in self.dashboards.values()),
            'cached_widgets': len(self.widget_data_cache),
            'cache_entries': len(self.cache_timestamps),
            'last_refresh': datetime.now().isoformat()
        }

    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export dashboard data for backup"""
        with self._lock:
            return {
                'dashboards': {did: d.to_dict() for did, d in self.dashboards.items()},
                'cache_info': {
                    'cached_widgets': len(self.widget_data_cache),
                    'cache_timestamps': list(self.cache_timestamps.keys()),
                    'cache_ttl': self.cache_ttl
                },
                'statistics': self.get_dashboard_statistics(),
                'exported_at': datetime.now().isoformat()
            }


# Global dashboard manager
dashboard_manager = CollaborativeDashboardManager()