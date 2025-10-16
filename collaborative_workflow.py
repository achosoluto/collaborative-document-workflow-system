"""
Collaborative Document Workflow System
Comprehensive system for real-time collaboration, document workflows, and team management
"""

import json
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid

# Import existing system components
from .version_control import version_manager
from .compliance_tracker import compliance_tracker


class UserRole(Enum):
    """User roles in the collaborative system"""
    VIEWER = "viewer"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    APPROVER = "approver"
    ADMIN = "admin"


class WorkflowStatus(Enum):
    """Workflow status types"""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class CollaborationEventType(Enum):
    """Types of collaboration events"""
    DOCUMENT_EDIT = "document_edit"
    COMMENT_ADDED = "comment_added"
    WORKFLOW_STATUS_CHANGED = "workflow_status_changed"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    PERMISSION_CHANGED = "permission_changed"


@dataclass
class CollaborativeUser:
    """Represents a user in the collaborative system"""
    user_id: str
    username: str
    email: str
    role: UserRole
    is_online: bool = False
    last_seen: datetime = field(default_factory=datetime.now)
    current_document: Optional[str] = None
    permissions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'is_online': self.is_online,
            'last_seen': self.last_seen.isoformat(),
            'current_document': self.current_document,
            'permissions': self.permissions
        }


@dataclass
class DocumentComment:
    """Represents a comment on a document"""
    comment_id: str
    document_id: str
    user_id: str
    username: str
    content: str
    position: Dict[str, Any]  # x, y coordinates for annotations
    parent_comment_id: Optional[str] = None  # For threaded comments
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    mentions: List[str] = field(default_factory=list)  # User IDs mentioned
    attachments: List[str] = field(default_factory=list)  # File paths/URLs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'comment_id': self.comment_id,
            'document_id': self.document_id,
            'user_id': self.user_id,
            'username': self.username,
            'content': self.content,
            'position': self.position,
            'parent_comment_id': self.parent_comment_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_resolved': self.is_resolved,
            'resolved_by': self.resolved_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'mentions': self.mentions,
            'attachments': self.attachments
        }


@dataclass
class WorkflowTask:
    """Represents a task in a document workflow"""
    task_id: str
    document_id: str
    title: str
    description: str
    assigned_to: List[str]  # User IDs
    created_by: str
    status: str  # pending, in_progress, completed, cancelled
    priority: str  # low, medium, high, critical
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)  # Other task IDs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'document_id': self.document_id,
            'title': self.title,
            'description': self.description,
            'assigned_to': self.assigned_to,
            'created_by': self.created_by,
            'status': self.status,
            'priority': self.priority,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }


@dataclass
class DocumentWorkflow:
    """Represents a document workflow"""
    workflow_id: str
    document_id: str
    title: str
    description: str
    status: WorkflowStatus
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    current_stage: str = "draft"
    stages: List[Dict[str, Any]] = field(default_factory=list)
    reviewers: Dict[str, str] = field(default_factory=dict)  # user_id -> role
    approvers: Dict[str, str] = field(default_factory=dict)  # user_id -> role
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'document_id': self.document_id,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'current_stage': self.current_stage,
            'stages': self.stages,
            'reviewers': self.reviewers,
            'approvers': self.approvers,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class CollaborationEvent:
    """Represents a collaboration event"""
    event_id: str
    event_type: CollaborationEventType
    document_id: str
    user_id: str
    username: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'document_id': self.document_id,
            'user_id': self.user_id,
            'username': self.username,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'session_id': self.session_id
        }


@dataclass
class OperationalTransform:
    """Represents an operational transform for collaborative editing"""
    operation_id: str
    document_id: str
    user_id: str
    operation_type: str  # insert, delete, retain
    position: int
    content: str = ""
    length: int = 0  # For delete operations
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_id': self.operation_id,
            'document_id': self.document_id,
            'user_id': self.user_id,
            'operation_type': self.operation_type,
            'position': self.position,
            'content': self.content,
            'length': self.length,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version
        }


class CollaborativeWorkflowManager:
    """Main manager for collaborative document workflows"""

    def __init__(self):
        self.users: Dict[str, CollaborativeUser] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.document_collaborators: Dict[str, Set[str]] = defaultdict(set)
        self.comments: Dict[str, List[DocumentComment]] = defaultdict(list)
        self.workflows: Dict[str, DocumentWorkflow] = {}
        self.tasks: Dict[str, WorkflowTask] = {}
        self.events: List[CollaborationEvent] = []
        self.operation_history: Dict[str, List[OperationalTransform]] = defaultdict(list)

        # Notification system
        self.notifications: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Locks for thread safety
        self._lock = threading.Lock()

        # Start background cleanup task
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background task to clean up old data"""
        def cleanup():
            while True:
                try:
                    self._cleanup_old_data()
                    time.sleep(3600)  # Run every hour
                except Exception as e:
                    print(f"Error in cleanup task: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying

        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()

    def _cleanup_old_data(self):
        """Clean up old events, offline users, etc."""
        cutoff_time = datetime.now() - timedelta(hours=24)

        # Clean old events
        with self._lock:
            self.events = [e for e in self.events if e.timestamp > cutoff_time]

            # Mark users as offline if not seen recently
            for user in self.users.values():
                if user.last_seen < cutoff_time and user.is_online:
                    user.is_online = False

    def add_user(self, user_id: str, username: str, email: str, role: UserRole) -> CollaborativeUser:
        """Add a new user to the collaborative system"""
        with self._lock:
            user = CollaborativeUser(
                user_id=user_id,
                username=username,
                email=email,
                role=role
            )
            self.users[user_id] = user
            return user

    def update_user_presence(self, user_id: str, is_online: bool, current_document: str = None):
        """Update user presence status"""
        with self._lock:
            if user_id in self.users:
                user = self.users[user_id]
                user.is_online = is_online
                user.last_seen = datetime.now()
                if current_document:
                    user.current_document = current_document

                # Add event
                event = CollaborationEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=CollaborationEventType.USER_JOINED if is_online else CollaborationEventType.USER_LEFT,
                    document_id=current_document or "",
                    user_id=user_id,
                    username=user.username
                )
                self.events.append(event)

    def get_document_collaborators(self, document_id: str) -> List[CollaborativeUser]:
        """Get all collaborators for a document"""
        with self._lock:
            user_ids = self.document_collaborators[document_id]
            return [self.users[uid] for uid in user_ids if uid in self.users]

    def add_document_collaborator(self, document_id: str, user_id: str, role: UserRole):
        """Add a collaborator to a document"""
        with self._lock:
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")

            self.document_collaborators[document_id].add(user_id)

            # Update user permissions for this document
            self.users[user_id].permissions[document_id] = role.value

            # Add event
            event = CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=CollaborationEventType.PERMISSION_CHANGED,
                document_id=document_id,
                user_id=user_id,
                username=self.users[user_id].username,
                data={'role': role.value}
            )
            self.events.append(event)

    def remove_document_collaborator(self, document_id: str, user_id: str):
        """Remove a collaborator from a document"""
        with self._lock:
            self.document_collaborators[document_id].discard(user_id)

            # Remove user permissions for this document
            if user_id in self.users and document_id in self.users[user_id].permissions:
                del self.users[user_id].permissions[document_id]

    def add_comment(self, document_id: str, user_id: str, content: str,
                   position: Dict[str, Any] = None, parent_comment_id: str = None) -> DocumentComment:
        """Add a comment to a document"""
        with self._lock:
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")

            comment = DocumentComment(
                comment_id=str(uuid.uuid4()),
                document_id=document_id,
                user_id=user_id,
                username=self.users[user_id].username,
                content=content,
                position=position or {},
                parent_comment_id=parent_comment_id
            )

            self.comments[document_id].append(comment)

            # Extract mentions from content
            mentions = self._extract_mentions(content)
            comment.mentions = mentions

            # Send notifications to mentioned users
            for mentioned_user_id in mentions:
                self._add_notification(mentioned_user_id, {
                    'type': 'mention',
                    'document_id': document_id,
                    'comment_id': comment.comment_id,
                    'from_user': user_id,
                    'message': f"You were mentioned in a comment on document {document_id}"
                })

            # Add event
            event = CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=CollaborationEventType.COMMENT_ADDED,
                document_id=document_id,
                user_id=user_id,
                username=self.users[user_id].username,
                data={'comment_id': comment.comment_id}
            )
            self.events.append(event)

            return comment

    def get_document_comments(self, document_id: str) -> List[DocumentComment]:
        """Get all comments for a document"""
        with self._lock:
            return self.comments[document_id].copy()

    def resolve_comment(self, comment_id: str, user_id: str):
        """Resolve a comment"""
        with self._lock:
            for document_comments in self.comments.values():
                for comment in document_comments:
                    if comment.comment_id == comment_id:
                        comment.is_resolved = True
                        comment.resolved_by = user_id
                        comment.resolved_at = datetime.now()

                        # Add event
                        event = CollaborationEvent(
                            event_id=str(uuid.uuid4()),
                            event_type=CollaborationEventType.COMMENT_ADDED,
                            document_id=comment.document_id,
                            user_id=user_id,
                            username=self.users[user_id].username,
                            data={'comment_id': comment_id, 'action': 'resolved'}
                        )
                        self.events.append(event)
                        return True
            return False

    def create_workflow(self, document_id: str, title: str, description: str,
                       created_by: str, reviewers: List[str] = None,
                       approvers: List[str] = None) -> DocumentWorkflow:
        """Create a new document workflow"""
        with self._lock:
            workflow = DocumentWorkflow(
                workflow_id=str(uuid.uuid4()),
                document_id=document_id,
                title=title,
                description=description,
                status=WorkflowStatus.DRAFT,
                created_by=created_by,
                reviewers={uid: 'reviewer' for uid in (reviewers or [])},
                approvers={uid: 'approver' for uid in (approvers or [])}
            )

            self.workflows[workflow.workflow_id] = workflow

            # Add event
            event = CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=CollaborationEventType.WORKFLOW_STATUS_CHANGED,
                document_id=document_id,
                user_id=created_by,
                username=self.users[created_by].username,
                data={'workflow_id': workflow.workflow_id, 'status': workflow.status.value}
            )
            self.events.append(event)

            return workflow

    def create_task(self, document_id: str, title: str, description: str,
                   assigned_to: List[str], created_by: str, priority: str = 'medium',
                   due_date: datetime = None) -> WorkflowTask:
        """Create a new workflow task"""
        with self._lock:
            task = WorkflowTask(
                task_id=str(uuid.uuid4()),
                document_id=document_id,
                title=title,
                description=description,
                assigned_to=assigned_to,
                created_by=created_by,
                status='pending',
                priority=priority,
                due_date=due_date
            )

            self.tasks[task.task_id] = task

            # Send notifications to assigned users
            for assignee_id in assigned_to:
                self._add_notification(assignee_id, {
                    'type': 'task_assigned',
                    'task_id': task.task_id,
                    'document_id': document_id,
                    'from_user': created_by,
                    'message': f"You have been assigned a task: {title}"
                })

            return task

    def update_task_status(self, task_id: str, status: str, user_id: str):
        """Update task status"""
        with self._lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            old_status = task.status
            task.status = status

            if status == 'completed':
                task.completed_at = datetime.now()

            # Add event
            event = CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=CollaborationEventType.WORKFLOW_STATUS_CHANGED,
                document_id=task.document_id,
                user_id=user_id,
                username=self.users[user_id].username,
                data={'task_id': task_id, 'old_status': old_status, 'new_status': status}
            )
            self.events.append(event)

            return True

    def add_collaborative_edit(self, document_id: str, user_id: str,
                              operation_type: str, position: int,
                              content: str = "", length: int = 0) -> OperationalTransform:
        """Add a collaborative editing operation"""
        with self._lock:
            # Get current version
            current_version = len(self.operation_history[document_id])

            operation = OperationalTransform(
                operation_id=str(uuid.uuid4()),
                document_id=document_id,
                user_id=user_id,
                operation_type=operation_type,
                position=position,
                content=content,
                length=length,
                version=current_version + 1
            )

            self.operation_history[document_id].append(operation)

            # Add event
            event = CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=CollaborationEventType.DOCUMENT_EDIT,
                document_id=document_id,
                user_id=user_id,
                username=self.users[user_id].username,
                data={'operation_id': operation.operation_id}
            )
            self.events.append(event)

            return operation

    def get_document_operations(self, document_id: str, since_version: int = 0) -> List[OperationalTransform]:
        """Get document operations since a specific version"""
        with self._lock:
            operations = self.operation_history[document_id]
            return [op for op in operations if op.version > since_version]

    def _extract_mentions(self, content: str) -> List[str]:
        """Extract user mentions from comment content"""
        mentions = []
        words = content.split()

        for word in words:
            if word.startswith('@') and len(word) > 1:
                username = word[1:]
                # Find user ID by username
                for user in self.users.values():
                    if user.username == username:
                        mentions.append(user.user_id)
                        break

        return mentions

    def _add_notification(self, user_id: str, notification: Dict[str, Any]):
        """Add a notification for a user"""
        notification['id'] = str(uuid.uuid4())
        notification['timestamp'] = datetime.now().isoformat()
        notification['read'] = False

        self.notifications[user_id].append(notification)

    def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get notifications for a user"""
        with self._lock:
            notifications = self.notifications[user_id]

            if unread_only:
                notifications = [n for n in notifications if not n.get('read', False)]

            # Sort by timestamp (newest first)
            notifications.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            return notifications[:50]  # Return last 50 notifications

    def mark_notifications_read(self, user_id: str, notification_ids: List[str] = None):
        """Mark notifications as read"""
        with self._lock:
            if notification_ids:
                # Mark specific notifications as read
                for notification in self.notifications[user_id]:
                    if notification['id'] in notification_ids:
                        notification['read'] = True
            else:
                # Mark all notifications as read
                for notification in self.notifications[user_id]:
                    notification['read'] = True

    def get_collaboration_stats(self, document_id: str = None) -> Dict[str, Any]:
        """Get collaboration statistics"""
        with self._lock:
            if document_id:
                # Document-specific stats
                collaborators = len(self.document_collaborators[document_id])
                comments = len(self.comments[document_id])
                operations = len(self.operation_history[document_id])
                workflows = len([w for w in self.workflows.values() if w.document_id == document_id])

                return {
                    'document_id': document_id,
                    'collaborators': collaborators,
                    'comments': comments,
                    'edit_operations': operations,
                    'active_workflows': workflows,
                    'online_collaborators': len([u for u in self.get_document_collaborators(document_id) if u.is_online])
                }
            else:
                # System-wide stats
                total_users = len(self.users)
                online_users = len([u for u in self.users.values() if u.is_online])
                total_collaborators = sum(len(users) for users in self.document_collaborators.values())
                total_comments = sum(len(comments) for comments in self.comments.values())

                return {
                    'total_users': total_users,
                    'online_users': online_users,
                    'total_collaborators': total_collaborators,
                    'total_comments': total_comments,
                    'total_workflows': len(self.workflows),
                    'total_tasks': len(self.tasks),
                    'recent_events': len([e for e in self.events if e.timestamp > datetime.now() - timedelta(hours=1)])
                }

    def get_document_activity_feed(self, document_id: str, limit: int = 50) -> List[CollaborationEvent]:
        """Get activity feed for a document"""
        with self._lock:
            feed = [e for e in self.events if e.document_id == document_id]
            feed.sort(key=lambda x: x.timestamp, reverse=True)
            return feed[:limit]

    def export_collaboration_data(self, document_id: str = None) -> Dict[str, Any]:
        """Export collaboration data for backup/reporting"""
        with self._lock:
            if document_id:
                # Document-specific export
                return {
                    'document_id': document_id,
                    'collaborators': [self.users[uid].to_dict() for uid in self.document_collaborators[document_id] if uid in self.users],
                    'comments': [c.to_dict() for c in self.comments[document_id]],
                    'workflows': [w.to_dict() for w in self.workflows.values() if w.document_id == document_id],
                    'tasks': [t.to_dict() for t in self.tasks.values() if t.document_id == document_id],
                    'operations': [op.to_dict() for op in self.operation_history[document_id]],
                    'events': [e.to_dict() for e in self.events if e.document_id == document_id],
                    'exported_at': datetime.now().isoformat()
                }
            else:
                # Full system export
                return {
                    'users': {uid: u.to_dict() for uid, u in self.users.items()},
                    'document_collaborators': {did: list(uids) for did, uids in self.document_collaborators.items()},
                    'comments': {did: [c.to_dict() for c in comments] for did, comments in self.comments.items()},
                    'workflows': {wid: w.to_dict() for wid, w in self.workflows.items()},
                    'tasks': {tid: t.to_dict() for tid, t in self.tasks.items()},
                    'operation_history': {did: [op.to_dict() for op in ops] for did, ops in self.operation_history.items()},
                    'events': [e.to_dict() for e in self.events],
                    'exported_at': datetime.now().isoformat()
                }


# Global collaborative workflow manager instance
collaborative_manager = CollaborativeWorkflowManager()