"""
Team Collaboration and Notification System
Real-time notifications, team management, and activity feeds
"""

import json
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
import uuid
from dataclasses import dataclass, field
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Import collaborative workflow components
from .collaborative_workflow import collaborative_manager, CollaborationEvent, CollaborationEventType


class NotificationType(Enum):
    """Types of notifications"""
    DOCUMENT_SHARED = "document_shared"
    COMMENT_MENTION = "comment_mention"
    TASK_ASSIGNED = "task_assigned"
    WORKFLOW_STATUS_CHANGED = "workflow_status_changed"
    DOCUMENT_EDITED = "document_edited"
    DEADLINE_APPROACHING = "deadline_approaching"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ANNOTATION_ADDED = "annotation_added"
    SYSTEM_ANNOUNCEMENT = "system_announcement"


class NotificationPriority(Enum):
    """Priority levels for notifications"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(Enum):
    """Notification delivery channels"""
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"
    SLACK = "slack"
    TEAMS = "teams"


@dataclass
class Notification:
    """Represents a notification"""
    notification_id: str
    user_id: str
    type: NotificationType
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.IN_APP])
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    action_url: Optional[str] = None
    action_label: str = "View"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'notification_id': self.notification_id,
            'user_id': self.user_id,
            'type': self.type.value,
            'title': self.title,
            'message': self.message,
            'priority': self.priority.value,
            'channels': [c.value for c in self.channels],
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'read_at': self.read_at.isoformat() if self.read_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'action_url': self.action_url,
            'action_label': self.action_label,
            'is_read': self.read_at is not None,
            'is_expired': self.expires_at and datetime.now() > self.expires_at
        }


@dataclass
class UserNotificationPreferences:
    """User notification preferences"""
    user_id: str
    enabled_types: Set[NotificationType] = field(default_factory=lambda: set(NotificationType))
    disabled_types: Set[NotificationType] = field(default_factory=set)
    channels_by_type: Dict[NotificationType, List[NotificationChannel]] = field(default_factory=dict)
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    timezone: str = "UTC"
    email_frequency: str = "immediate"  # immediate, hourly, daily

    def should_receive_notification(self, notification_type: NotificationType,
                                  current_hour: int) -> bool:
        """Check if user should receive a notification"""
        # Check if type is disabled
        if notification_type in self.disabled_types:
            return False

        # Check if type is explicitly enabled or all are enabled
        if self.enabled_types and notification_type not in self.enabled_types:
            return False

        # Check quiet hours
        if self.quiet_hours_start and self.quiet_hours_end:
            quiet_start = int(self.quiet_hours_start.split(':')[0])
            quiet_end = int(self.quiet_hours_end.split(':')[0])

            if quiet_start > quiet_end:  # Overnight
                if current_hour >= quiet_start or current_hour < quiet_end:
                    return False
            else:  # Same day
                if quiet_start <= current_hour < quiet_end:
                    return False

        return True

    def get_channels_for_type(self, notification_type: NotificationType) -> List[NotificationChannel]:
        """Get notification channels for a specific type"""
        if notification_type in self.channels_by_type:
            return self.channels_by_type[notification_type]

        # Default channels based on priority
        if notification_type in [NotificationType.TASK_ASSIGNED, NotificationType.DEADLINE_APPROACHING]:
            return [NotificationChannel.IN_APP, NotificationChannel.EMAIL]
        elif notification_type in [NotificationType.SYSTEM_ANNOUNCEMENT]:
            return [NotificationChannel.IN_APP, NotificationChannel.EMAIL, NotificationChannel.PUSH]

        return [NotificationChannel.IN_APP]


class NotificationManager:
    """Manager for notifications and team collaboration"""

    def __init__(self):
        self.notifications: Dict[str, List[Notification]] = {}  # user_id -> notifications
        self.user_preferences: Dict[str, UserNotificationPreferences] = {}
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}

        # Activity feed
        self.activity_feed: List[Dict[str, Any]] = []

        # Threading lock
        self._lock = threading.Lock()

        # Email configuration (would be loaded from config)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',  # Would be configured
            'password': '',  # Would be configured
            'enabled': False
        }

        # Initialize default handlers
        self._initialize_handlers()

        # Start background cleanup
        self._start_cleanup_task()

    def _initialize_handlers(self):
        """Initialize notification delivery handlers"""
        self.notification_handlers[NotificationChannel.IN_APP] = self._send_in_app_notification
        self.notification_handlers[NotificationChannel.EMAIL] = self._send_email_notification
        # self.notification_handlers[NotificationChannel.PUSH] = self._send_push_notification
        # self.notification_handlers[NotificationChannel.SLACK] = self._send_slack_notification

    def _start_cleanup_task(self):
        """Start background cleanup task"""
        def cleanup():
            while True:
                try:
                    self._cleanup_expired_notifications()
                    asyncio.sleep(3600)  # Run every hour
                except Exception as e:
                    print(f"Error in notification cleanup: {e}")
                    asyncio.sleep(300)

        import time
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()

    def _cleanup_expired_notifications(self):
        """Remove expired notifications"""
        with self._lock:
            current_time = datetime.now()
            expired_ids = []

            for user_notifications in self.notifications.values():
                for i, notification in enumerate(user_notifications):
                    if notification.expires_at and current_time > notification.expires_at:
                        expired_ids.append((user_notifications, i))

            # Remove expired notifications
            for user_notifications, index in expired_ids:
                if index < len(user_notifications):
                    user_notifications.pop(index)

    def set_user_preferences(self, user_id: str, preferences: UserNotificationPreferences):
        """Set notification preferences for a user"""
        with self._lock:
            self.user_preferences[user_id] = preferences

    def get_user_preferences(self, user_id: str) -> UserNotificationPreferences:
        """Get notification preferences for a user"""
        with self._lock:
            if user_id not in self.user_preferences:
                # Create default preferences
                self.user_preferences[user_id] = UserNotificationPreferences(
                    user_id=user_id,
                    enabled_types=set(NotificationType)
                )
            return self.user_preferences[user_id]

    def send_notification(self, user_id: str, notification_type: NotificationType,
                         title: str, message: str, priority: NotificationPriority = NotificationPriority.NORMAL,
                         channels: List[NotificationChannel] = None,
                         data: Dict[str, Any] = None,
                         expires_in: timedelta = None) -> str:
        """Send a notification to a user"""
        with self._lock:
            # Get user preferences
            preferences = self.get_user_preferences(user_id)

            # Check if user wants this type of notification
            current_hour = datetime.now().hour
            if not preferences.should_receive_notification(notification_type, current_hour):
                return None

            # Create notification
            notification = Notification(
                notification_id=str(uuid.uuid4()),
                user_id=user_id,
                type=notification_type,
                title=title,
                message=message,
                priority=priority,
                channels=channels or preferences.get_channels_for_type(notification_type),
                data=data or {},
                expires_at=datetime.now() + (expires_in or timedelta(days=30))
            )

            # Store notification
            if user_id not in self.notifications:
                self.notifications[user_id] = []
            self.notifications[user_id].append(notification)

            # Keep only recent notifications (max 200 per user)
            if len(self.notifications[user_id]) > 200:
                self.notifications[user_id] = self.notifications[user_id][-200:]

            # Add to activity feed
            self.activity_feed.append({
                'notification_id': notification.notification_id,
                'user_id': user_id,
                'type': notification_type.value,
                'title': title,
                'timestamp': notification.created_at.isoformat()
            })

            # Keep only recent activity (last 1000 items)
            if len(self.activity_feed) > 1000:
                self.activity_feed = self.activity_feed[-1000:]

            # Send through configured channels
            self._deliver_notification(notification)

            return notification.notification_id

    def _deliver_notification(self, notification: Notification):
        """Deliver notification through configured channels"""
        for channel in notification.channels:
            if channel in self.notification_handlers:
                try:
                    handler = self.notification_handlers[channel]
                    asyncio.create_task(handler(notification))
                except Exception as e:
                    print(f"Error delivering notification via {channel.value}: {e}")

    async def _send_in_app_notification(self, notification: Notification):
        """Send in-app notification"""
        # This would integrate with the WebSocket system to send real-time notifications
        # For now, just log it
        print(f"In-app notification for {notification.user_id}: {notification.title}")

    async def _send_email_notification(self, notification: Notification):
        """Send email notification"""
        if not self.email_config['enabled']:
            return

        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = notification.user_id  # Assuming user_id is email
            msg['Subject'] = notification.title

            # Email body
            body = f"""
            {notification.message}

            Priority: {notification.priority.value}
            Time: {notification.created_at.strftime('%Y-%m-%d %H:%M:%S')}

            {notification.action_label}: {notification.action_url or ''}
            """

            msg.attach(MimeText(body, 'plain'))

            # Send email (would need proper async implementation)
            print(f"Email notification for {notification.user_id}: {notification.title}")

        except Exception as e:
            print(f"Error sending email notification: {e}")

    def mark_as_read(self, user_id: str, notification_ids: List[str] = None):
        """Mark notifications as read"""
        with self._lock:
            if user_id not in self.notifications:
                return

            if notification_ids:
                # Mark specific notifications as read
                for notification in self.notifications[user_id]:
                    if notification.notification_id in notification_ids:
                        notification.read_at = datetime.now()
            else:
                # Mark all notifications as read
                for notification in self.notifications[user_id]:
                    notification.read_at = datetime.now()

    def get_user_notifications(self, user_id: str, unread_only: bool = False,
                              limit: int = 50) -> List[Notification]:
        """Get notifications for a user"""
        with self._lock:
            notifications = self.notifications.get(user_id, [])

            if unread_only:
                notifications = [n for n in notifications if not n.read_at]

            # Sort by creation time (newest first)
            notifications.sort(key=lambda x: x.created_at, reverse=True)

            return notifications[:limit]

    def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications for a user"""
        with self._lock:
            notifications = self.notifications.get(user_id, [])
            return len([n for n in notifications if not n.read_at])

    def send_bulk_notification(self, user_ids: List[str], notification_type: NotificationType,
                              title: str, message: str, **kwargs):
        """Send notification to multiple users"""
        notification_ids = []

        for user_id in user_ids:
            notification_id = self.send_notification(
                user_id, notification_type, title, message, **kwargs
            )
            if notification_id:
                notification_ids.append(notification_id)

        return notification_ids

    def create_deadline_notifications(self):
        """Create notifications for approaching deadlines"""
        with self._lock:
            now = datetime.now()
            tomorrow = now + timedelta(days=1)
            next_week = now + timedelta(days=7)

            # Check workflow deadlines
            for workflow in collaborative_manager.workflows.values():
                if (workflow.due_date and
                    workflow.due_date > now and
                    workflow.due_date <= next_week and
                    workflow.status.value != 'completed'):

                    # Notify assigned users
                    for user_id in list(workflow.reviewers.keys()) + list(workflow.approvers.keys()):
                        if workflow.due_date <= tomorrow:
                            priority = NotificationPriority.URGENT
                        elif workflow.due_date <= next_week:
                            priority = NotificationPriority.HIGH
                        else:
                            priority = NotificationPriority.NORMAL

                        self.send_notification(
                            user_id,
                            NotificationType.DEADLINE_APPROACHING,
                            "Deadline Approaching",
                            f"Workflow '{workflow.title}' is due on {workflow.due_date.strftime('%Y-%m-%d')}",
                            priority=priority,
                            data={'workflow_id': workflow.workflow_id, 'due_date': workflow.due_date.isoformat()}
                        )

            # Check task deadlines
            for task in collaborative_manager.tasks.values():
                if (task.due_date and
                    task.due_date > now and
                    task.due_date <= next_week and
                    task.status != 'completed'):

                    # Notify assigned users
                    for user_id in task.assigned_to:
                        if task.due_date <= tomorrow:
                            priority = NotificationPriority.URGENT
                        elif task.due_date <= next_week:
                            priority = NotificationPriority.HIGH
                        else:
                            priority = NotificationPriority.NORMAL

                        self.send_notification(
                            user_id,
                            NotificationType.DEADLINE_APPROACHING,
                            "Task Deadline Approaching",
                            f"Task '{task.title}' is due on {task.due_date.strftime('%Y-%m-%d')}",
                            priority=priority,
                            data={'task_id': task.task_id, 'due_date': task.due_date.isoformat()}
                        )

    def get_activity_feed(self, user_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get activity feed"""
        with self._lock:
            feed = self.activity_feed.copy()

            if user_id:
                # Filter to user's activities
                feed = [item for item in feed if item['user_id'] == user_id]

            # Sort by timestamp (newest first)
            feed.sort(key=lambda x: x['timestamp'], reverse=True)

            return feed[:limit]

    def integrate_with_collaborative_events(self):
        """Integrate with collaborative workflow events"""
        # This would be called when collaborative events occur
        # For now, we'll set up event handlers for key events

        # Listen for workflow status changes
        def on_workflow_status_changed(event: CollaborationEvent):
            if event.event_type == CollaborationEventType.WORKFLOW_STATUS_CHANGED:
                data = event.data
                if 'workflow_id' in data:
                    workflow = collaborative_manager.workflows.get(data['workflow_id'])
                    if workflow:
                        # Notify workflow participants
                        participants = list(workflow.reviewers.keys()) + list(workflow.approvers.keys())
                        for participant_id in participants:
                            self.send_notification(
                                participant_id,
                                NotificationType.WORKFLOW_STATUS_CHANGED,
                                "Workflow Status Updated",
                                f"Workflow '{workflow.title}' status changed to {workflow.status.value}",
                                data={'workflow_id': workflow.workflow_id}
                            )

        # Listen for comments
        def on_comment_added(event: CollaborationEvent):
            if event.event_type == CollaborationEventType.COMMENT_ADDED:
                # This would be handled by the annotation system integration
                pass

        # Listen for document edits
        def on_document_edited(event: CollaborationEvent):
            if event.event_type == CollaborationEventType.DOCUMENT_EDIT:
                # Notify document collaborators about edits
                document_id = event.document_id
                collaborators = collaborative_manager.get_document_collaborators(document_id)

                for collaborator in collaborators:
                    if collaborator.user_id != event.user_id:  # Don't notify the editor
                        self.send_notification(
                            collaborator.user_id,
                            NotificationType.DOCUMENT_EDITED,
                            "Document Edited",
                            f"{event.username} made changes to the document",
                            data={'document_id': document_id, 'editor_id': event.user_id}
                        )

    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics"""
        with self._lock:
            total_notifications = sum(len(notifications) for notifications in self.notifications.values())
            unread_notifications = sum(
                len([n for n in notifications if not n.read_at])
                for notifications in self.notifications.values()
            )

            # Count by type
            type_counts = {}
            for notifications in self.notifications.values():
                for notification in notifications:
                    type_key = notification.type.value
                    type_counts[type_key] = type_counts.get(type_key, 0) + 1

            return {
                'total_notifications': total_notifications,
                'unread_notifications': unread_notifications,
                'total_users': len(self.notifications),
                'notifications_by_type': type_counts,
                'recent_activity': len(self.activity_feed),
                'delivery_channels': list(self.notification_handlers.keys())
            }

    def export_notification_data(self) -> Dict[str, Any]:
        """Export notification data for backup"""
        with self._lock:
            return {
                'notifications': {
                    user_id: [n.to_dict() for n in notifications]
                    for user_id, notifications in self.notifications.items()
                },
                'user_preferences': {
                    user_id: {
                        'enabled_types': [t.value for t in prefs.enabled_types],
                        'disabled_types': [t.value for t in prefs.disabled_types],
                        'channels_by_type': {
                            t: [c.value for c in channels]
                            for t, channels in prefs.channels_by_type.items()
                        },
                        'quiet_hours_start': prefs.quiet_hours_start,
                        'quiet_hours_end': prefs.quiet_hours_end,
                        'timezone': prefs.timezone,
                        'email_frequency': prefs.email_frequency
                    }
                    for user_id, prefs in self.user_preferences.items()
                },
                'activity_feed': self.activity_feed,
                'statistics': self.get_notification_statistics(),
                'exported_at': datetime.now().isoformat()
            }


# Global notification manager
notification_manager = NotificationManager()

# Integration with collaborative workflow manager
def setup_collaborative_notifications():
    """Set up notification integration with collaborative workflow"""
    def on_collaborative_event(event: CollaborationEvent):
        if event.event_type == CollaborationEventType.WORKFLOW_STATUS_CHANGED:
            workflow = collaborative_manager.workflows.get(event.document_id)  # This would need adjustment
            if workflow:
                participants = list(workflow.reviewers.keys()) + list(workflow.approvers.keys())
                for participant_id in participants:
                    notification_manager.send_notification(
                        participant_id,
                        NotificationType.WORKFLOW_STATUS_CHANGED,
                        "Workflow Updated",
                        f"Workflow '{workflow.title}' was updated",
                        data={'workflow_id': workflow.workflow_id}
                    )

        elif event.event_type == CollaborationEventType.COMMENT_ADDED:
            # Handle comment notifications (mentions would be handled by annotation system)
            pass

        elif event.event_type == CollaborationEventType.DOCUMENT_EDIT:
            # Handle document edit notifications
            document_id = event.document_id
            collaborators = collaborative_manager.get_document_collaborators(document_id)
            for collaborator in collaborators:
                if collaborator.user_id != event.user_id:
                    notification_manager.send_notification(
                        collaborator.user_id,
                        NotificationType.DOCUMENT_EDITED,
                        "Document Edited",
                        f"{event.username} made changes to the document",
                        data={'document_id': document_id}
                    )

    # This would be connected to the collaborative manager's event system
    # collaborative_manager.add_event_listener(on_collaborative_event)