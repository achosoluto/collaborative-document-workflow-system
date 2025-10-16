"""
Commenting and Annotation System
Threaded discussions and document annotations with rich text support
"""

import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import uuid
from dataclasses import dataclass, field
from enum import Enum

# Import collaborative workflow components
from .collaborative_workflow import collaborative_manager, CollaborationEvent, CollaborationEventType


class AnnotationType(Enum):
    """Types of annotations"""
    COMMENT = "comment"
    HIGHLIGHT = "highlight"
    NOTE = "note"
    SUGGESTION = "suggestion"
    QUESTION = "question"
    TODO = "todo"
    APPROVAL = "approval"
    REJECTION = "rejection"


class CommentStatus(Enum):
    """Status of comments"""
    OPEN = "open"
    RESOLVED = "resolved"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"


@dataclass
class DocumentAnnotation:
    """Represents an annotation on a document"""
    annotation_id: str
    document_id: str
    type: AnnotationType
    content: str
    position: Dict[str, Any]  # x, y, width, height, page, etc.
    author_id: str
    author_name: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # For threaded comments
    parent_id: Optional[str] = None
    thread_id: Optional[str] = None

    # Status and metadata
    status: CommentStatus = CommentStatus.OPEN
    priority: str = "normal"  # low, normal, high, urgent
    tags: List[str] = field(default_factory=list)
    assigned_to: List[str] = field(default_factory=list)  # User IDs
    due_date: Optional[datetime] = None

    # Rich content
    mentions: List[str] = field(default_factory=list)  # User IDs
    attachments: List[Dict[str, Any]] = field(default_factory=list)  # File info
    reactions: Dict[str, List[str]] = field(default_factory=dict)  # emoji -> user_ids

    # Position in document (for different document types)
    page_number: Optional[int] = None
    text_range: Optional[Dict[str, Any]] = None  # start_offset, end_offset
    element_selector: Optional[str] = None  # CSS selector for HTML documents

    def to_dict(self) -> Dict[str, Any]:
        return {
            'annotation_id': self.annotation_id,
            'document_id': self.document_id,
            'type': self.type.value,
            'content': self.content,
            'position': self.position,
            'author_id': self.author_id,
            'author_name': self.author_name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'parent_id': self.parent_id,
            'thread_id': self.thread_id,
            'status': self.status.value,
            'priority': self.priority,
            'tags': self.tags,
            'assigned_to': self.assigned_to,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'mentions': self.mentions,
            'attachments': self.attachments,
            'reactions': self.reactions,
            'page_number': self.page_number,
            'text_range': self.text_range,
            'element_selector': self.element_selector
        }


@dataclass
class AnnotationThread:
    """Represents a thread of related annotations"""
    thread_id: str
    document_id: str
    title: str
    annotations: List[str] = field(default_factory=list)  # annotation_ids
    participants: Set[str] = field(default_factory=set)  # user_ids
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'thread_id': self.thread_id,
            'document_id': self.document_id,
            'title': self.title,
            'annotations': self.annotations,
            'participants': list(self.participants),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_resolved': self.is_resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by
        }


class AnnotationManager:
    """Manager for document annotations and comments"""

    def __init__(self):
        self.annotations: Dict[str, DocumentAnnotation] = {}
        self.threads: Dict[str, AnnotationThread] = {}
        self.document_annotations: Dict[str, List[str]] = {}  # doc_id -> annotation_ids
        self.user_annotations: Dict[str, List[str]] = {}  # user_id -> annotation_ids

        # Threading lock for data consistency
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_annotations': 0,
            'total_threads': 0,
            'annotations_by_type': {},
            'recent_activity': []
        }

    def create_annotation(self, document_id: str, annotation_type: AnnotationType,
                         content: str, position: Dict[str, Any], author_id: str,
                         author_name: str, **kwargs) -> DocumentAnnotation:
        """Create a new annotation"""
        with self._lock:
            annotation = DocumentAnnotation(
                annotation_id=str(uuid.uuid4()),
                document_id=document_id,
                type=annotation_type,
                content=content,
                position=position,
                author_id=author_id,
                author_name=author_name,
                **kwargs
            )

            # Store annotation
            self.annotations[annotation.annotation_id] = annotation

            # Update document index
            if document_id not in self.document_annotations:
                self.document_annotations[document_id] = []
            self.document_annotations[document_id].append(annotation.annotation_id)

            # Update user index
            if author_id not in self.user_annotations:
                self.user_annotations[author_id] = []
            self.user_annotations[author_id].append(annotation.annotation_id)

            # Extract mentions and create thread if needed
            annotation.mentions = self._extract_mentions(content)

            thread_id = self._create_or_update_thread(annotation)

            # Update statistics
            self._update_stats(annotation)

            # Add to collaborative workflow manager
            collaborative_manager.add_comment(
                document_id, author_id, content, position,
                parent_comment_id=annotation.parent_id
            )

            return annotation

    def reply_to_annotation(self, parent_annotation_id: str, content: str,
                          author_id: str, author_name: str) -> DocumentAnnotation:
        """Create a reply to an existing annotation"""
        with self._lock:
            if parent_annotation_id not in self.annotations:
                raise ValueError(f"Parent annotation {parent_annotation_id} not found")

            parent_annotation = self.annotations[parent_annotation_id]

            # Create reply annotation
            reply = DocumentAnnotation(
                annotation_id=str(uuid.uuid4()),
                document_id=parent_annotation.document_id,
                type=AnnotationType.COMMENT,
                content=content,
                position=parent_annotation.position,  # Same position as parent
                author_id=author_id,
                author_name=author_name,
                parent_id=parent_annotation_id,
                thread_id=parent_annotation.thread_id
            )

            # Store reply
            self.annotations[reply.annotation_id] = reply

            # Update document index
            self.document_annotations[parent_annotation.document_id].append(reply.annotation_id)

            # Update user index
            self.user_annotations[author_id].append(reply.annotation_id)

            # Update thread
            if reply.thread_id:
                self.threads[reply.thread_id].annotations.append(reply.annotation_id)
                self.threads[reply.thread_id].participants.add(author_id)
                self.threads[reply.thread_id].updated_at = datetime.now()

            # Extract mentions
            reply.mentions = self._extract_mentions(content)

            # Update statistics
            self._update_stats(reply)

            # Add to collaborative workflow manager
            collaborative_manager.add_comment(
                parent_annotation.document_id, author_id, content,
                parent_annotation.position, parent_annotation_id
            )

            return reply

    def update_annotation(self, annotation_id: str, updates: Dict[str, Any],
                         user_id: str) -> bool:
        """Update an annotation"""
        with self._lock:
            if annotation_id not in self.annotations:
                return False

            annotation = self.annotations[annotation_id]

            # Check if user can update (author or assigned)
            if annotation.author_id != user_id and user_id not in annotation.assigned_to:
                return False

            # Apply updates
            for key, value in updates.items():
                if hasattr(annotation, key):
                    setattr(annotation, key, value)

            annotation.updated_at = datetime.now()

            # Update thread timestamp if part of thread
            if annotation.thread_id and annotation.thread_id in self.threads:
                self.threads[annotation.thread_id].updated_at = datetime.now()

            return True

    def delete_annotation(self, annotation_id: str, user_id: str) -> bool:
        """Delete an annotation"""
        with self._lock:
            if annotation_id not in self.annotations:
                return False

            annotation = self.annotations[annotation_id]

            # Only author can delete
            if annotation.author_id != user_id:
                return False

            # Remove from all indexes
            if annotation.document_id in self.document_annotations:
                self.document_annotations[annotation.document_id].remove(annotation_id)

            if annotation.author_id in self.user_annotations:
                self.user_annotations[annotation.author_id].remove(annotation_id)

            # Remove from thread if part of one
            if annotation.thread_id and annotation.thread_id in self.threads:
                thread = self.threads[annotation.thread_id]
                if annotation_id in thread.annotations:
                    thread.annotations.remove(annotation_id)

                # Remove thread if no annotations left
                if not thread.annotations:
                    del self.threads[annotation.thread_id]

            # Delete the annotation
            del self.annotations[annotation_id]

            # Update statistics
            self._decrement_stats(annotation)

            return True

    def resolve_annotation(self, annotation_id: str, user_id: str) -> bool:
        """Resolve an annotation"""
        with self._lock:
            if annotation_id not in self.annotations:
                return False

            annotation = self.annotations[annotation_id]
            annotation.status = CommentStatus.RESOLVED

            # Update thread if part of one
            if annotation.thread_id and annotation.thread_id in self.threads:
                self.threads[annotation.thread_id].is_resolved = True
                self.threads[annotation.thread_id].resolved_at = datetime.now()
                self.threads[annotation.thread_id].resolved_by = user_id

            return True

    def add_reaction(self, annotation_id: str, emoji: str, user_id: str) -> bool:
        """Add a reaction to an annotation"""
        with self._lock:
            if annotation_id not in self.annotations:
                return False

            annotation = self.annotations[annotation_id]

            if emoji not in annotation.reactions:
                annotation.reactions[emoji] = []

            if user_id not in annotation.reactions[emoji]:
                annotation.reactions[emoji].append(user_id)

            return True

    def remove_reaction(self, annotation_id: str, emoji: str, user_id: str) -> bool:
        """Remove a reaction from an annotation"""
        with self._lock:
            if annotation_id not in self.annotations:
                return False

            annotation = self.annotations[annotation_id]

            if emoji in annotation.reactions and user_id in annotation.reactions[emoji]:
                annotation.reactions[emoji].remove(user_id)

                # Remove emoji key if no reactions left
                if not annotation.reactions[emoji]:
                    del annotation.reactions[emoji]

                return True

            return False

    def get_document_annotations(self, document_id: str,
                               annotation_type: AnnotationType = None,
                               status: CommentStatus = None) -> List[DocumentAnnotation]:
        """Get all annotations for a document with optional filtering"""
        with self._lock:
            annotation_ids = self.document_annotations.get(document_id, [])

            annotations = []
            for annotation_id in annotation_ids:
                if annotation_id in self.annotations:
                    annotation = self.annotations[annotation_id]

                    # Apply filters
                    if annotation_type and annotation.type != annotation_type:
                        continue
                    if status and annotation.status != status:
                        continue

                    annotations.append(annotation)

            # Sort by creation time (newest first)
            annotations.sort(key=lambda x: x.created_at, reverse=True)

            return annotations

    def get_annotation_thread(self, thread_id: str) -> Optional[AnnotationThread]:
        """Get a complete annotation thread"""
        with self._lock:
            if thread_id not in self.threads:
                return None

            thread = self.threads[thread_id]

            # Get all annotations in the thread
            thread_annotations = []
            for annotation_id in thread.annotations:
                if annotation_id in self.annotations:
                    thread_annotations.append(self.annotations[annotation_id])

            # Sort annotations chronologically
            thread_annotations.sort(key=lambda x: x.created_at)

            return thread

    def get_user_annotations(self, user_id: str, limit: int = 100) -> List[DocumentAnnotation]:
        """Get annotations created by a user"""
        with self._lock:
            annotation_ids = self.user_annotations.get(user_id, [])

            annotations = []
            for annotation_id in annotation_ids:
                if annotation_id in self.annotations:
                    annotations.append(self.annotations[annotation_id])

            # Sort by creation time (newest first)
            annotations.sort(key=lambda x: x.created_at, reverse=True)

            return annotations[:limit]

    def search_annotations(self, query: str, document_id: str = None,
                          author_id: str = None) -> List[DocumentAnnotation]:
        """Search annotations by content"""
        with self._lock:
            results = []

            # Get candidate annotations
            candidate_ids = []
            if document_id:
                candidate_ids = self.document_annotations.get(document_id, [])
            else:
                candidate_ids = list(self.annotations.keys())

            query_lower = query.lower()

            for annotation_id in candidate_ids:
                if annotation_id in self.annotations:
                    annotation = self.annotations[annotation_id]

                    # Apply author filter
                    if author_id and annotation.author_id != author_id:
                        continue

                    # Search in content
                    if (query_lower in annotation.content.lower() or
                        any(query_lower in tag.lower() for tag in annotation.tags)):
                        results.append(annotation)

            # Sort by relevance and recency
            results.sort(key=lambda x: (x.created_at, x.content.lower().count(query_lower)), reverse=True)

            return results

    def get_annotation_statistics(self, document_id: str = None) -> Dict[str, Any]:
        """Get annotation statistics"""
        with self._lock:
            if document_id:
                # Document-specific statistics
                annotations = self.get_document_annotations(document_id)

                stats = {
                    'total_annotations': len(annotations),
                    'annotations_by_type': {},
                    'annotations_by_status': {},
                    'annotations_by_priority': {},
                    'recent_activity': []
                }

                for annotation in annotations:
                    # Count by type
                    type_key = annotation.type.value
                    stats['annotations_by_type'][type_key] = stats['annotations_by_type'].get(type_key, 0) + 1

                    # Count by status
                    status_key = annotation.status.value
                    stats['annotations_by_status'][status_key] = stats['annotations_by_status'].get(status_key, 0) + 1

                    # Count by priority
                    priority_key = annotation.priority
                    stats['annotations_by_priority'][priority_key] = stats['annotations_by_priority'].get(priority_key, 0) + 1

                    # Recent activity (last 7 days)
                    if annotation.created_at > datetime.now() - timedelta(days=7):
                        stats['recent_activity'].append({
                            'annotation_id': annotation.annotation_id,
                            'type': annotation.type.value,
                            'created_at': annotation.created_at.isoformat()
                        })

                return stats

            else:
                # System-wide statistics
                return {
                    'total_annotations': len(self.annotations),
                    'total_threads': len(self.threads),
                    'annotations_by_type': self.stats['annotations_by_type'].copy(),
                    'annotations_by_status': self._count_by_status(),
                    'most_active_documents': self._get_most_active_documents(),
                    'recent_activity': self.stats['recent_activity'][-50:]  # Last 50 activities
                }

    def _create_or_update_thread(self, annotation: DocumentAnnotation) -> str:
        """Create a new thread or update existing thread"""
        # If no parent, create new thread
        if not annotation.parent_id:
            thread = AnnotationThread(
                thread_id=str(uuid.uuid4()),
                document_id=annotation.document_id,
                title=f"Discussion on {annotation.type.value}"
            )
            thread.annotations.append(annotation.annotation_id)
            thread.participants.add(annotation.author_id)

            self.threads[thread.thread_id] = thread
            annotation.thread_id = thread.thread_id

            return thread.thread_id

        # If has parent, use parent's thread
        parent_annotation = self.annotations.get(annotation.parent_id)
        if parent_annotation and parent_annotation.thread_id:
            annotation.thread_id = parent_annotation.thread_id
            return parent_annotation.thread_id

        # Fallback: create new thread
        return self._create_or_update_thread(DocumentAnnotation(
            annotation_id=annotation.annotation_id,
            document_id=annotation.document_id,
            type=AnnotationType.COMMENT,
            content=annotation.content,
            position=annotation.position,
            author_id=annotation.author_id,
            author_name=annotation.author_name
        ))

    def _extract_mentions(self, content: str) -> List[str]:
        """Extract user mentions from content"""
        mentions = []
        words = content.split()

        for word in words:
            if word.startswith('@') and len(word) > 1:
                username = word[1:]
                # Find user ID by username (this would need to integrate with user system)
                # For now, return the username as-is
                mentions.append(username)

        return mentions

    def _update_stats(self, annotation: DocumentAnnotation):
        """Update internal statistics"""
        self.stats['total_annotations'] += 1

        # Update type counts
        type_key = annotation.type.value
        self.stats['annotations_by_type'][type_key] = self.stats['annotations_by_type'].get(type_key, 0) + 1

        # Track recent activity
        self.stats['recent_activity'].append({
            'annotation_id': annotation.annotation_id,
            'type': annotation.type.value,
            'document_id': annotation.document_id,
            'created_at': annotation.created_at.isoformat()
        })

        # Keep only recent activity (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        self.stats['recent_activity'] = [
            activity for activity in self.stats['recent_activity']
            if datetime.fromisoformat(activity['created_at']) > cutoff
        ]

    def _decrement_stats(self, annotation: DocumentAnnotation):
        """Decrement statistics when annotation is deleted"""
        self.stats['total_annotations'] -= 1

        # Decrement type counts
        type_key = annotation.type.value
        if type_key in self.stats['annotations_by_type']:
            self.stats['annotations_by_type'][type_key] -= 1
            if self.stats['annotations_by_type'][type_key] <= 0:
                del self.stats['annotations_by_type'][type_key]

    def _count_by_status(self) -> Dict[str, int]:
        """Count annotations by status"""
        status_counts = {}

        for annotation in self.annotations.values():
            status_key = annotation.status.value
            status_counts[status_key] = status_counts.get(status_key, 0) + 1

        return status_counts

    def _get_most_active_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get documents with most annotation activity"""
        doc_activity = {}

        for annotation in self.annotations.values():
            doc_id = annotation.document_id
            if doc_id not in doc_activity:
                doc_activity[doc_id] = 0
            doc_activity[doc_id] += 1

        # Sort by activity count
        sorted_docs = sorted(doc_activity.items(), key=lambda x: x[1], reverse=True)

        return [
            {'document_id': doc_id, 'annotation_count': count}
            for doc_id, count in sorted_docs[:limit]
        ]

    def export_annotations(self, document_id: str = None) -> Dict[str, Any]:
        """Export annotations for backup or reporting"""
        with self._lock:
            if document_id:
                # Export annotations for specific document
                annotation_ids = self.document_annotations.get(document_id, [])
                annotations = [
                    self.annotations[aid].to_dict()
                    for aid in annotation_ids if aid in self.annotations
                ]

                thread_ids = set()
                for annotation in annotations:
                    if annotation.get('thread_id'):
                        thread_ids.add(annotation['thread_id'])

                threads = [
                    self.threads[tid].to_dict()
                    for tid in thread_ids if tid in self.threads
                ]

                return {
                    'document_id': document_id,
                    'annotations': annotations,
                    'threads': threads,
                    'exported_at': datetime.now().isoformat()
                }

            else:
                # Export all annotations
                return {
                    'annotations': [a.to_dict() for a in self.annotations.values()],
                    'threads': [t.to_dict() for t in self.threads.values()],
                    'statistics': self.get_annotation_statistics(),
                    'exported_at': datetime.now().isoformat()
                }


# Global annotation manager
annotation_manager = AnnotationManager()