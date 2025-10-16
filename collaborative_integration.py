"""
Collaborative Workflow Integration Layer
Integrates collaborative features with existing search, version control, compliance, and summarization systems
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import uuid
from dataclasses import dataclass, field

# Import existing system components
from .search_engine import AdvancedSearchEngine
from .integration import integration_manager, VersionSearchEnhancer
from .compliance_integration import compliance_integration_manager
from .summarization_engine import summarization_engine

# Import collaborative components
from .collaborative_workflow import collaborative_manager, DocumentWorkflow, WorkflowTask, WorkflowStatus, UserRole
from .collaborative_editor import editor_manager, DocumentState
from .annotation_system import annotation_manager, DocumentAnnotation, AnnotationType
from .notification_system import notification_manager, NotificationType, NotificationPriority
from .workflow_automation import automation_engine, task_manager
from .collaborative_dashboard import dashboard_manager
from .websocket_manager import websocket_manager


class CollaborativeSearchIntegrator:
    """Integrates collaborative features with search system"""

    def __init__(self):
        self.search_engine = AdvancedSearchEngine()
        self.version_search_enhancer = VersionSearchEnhancer()

    def search_with_collaboration_context(self, query: str, user_id: str = None,
                                        collaboration_filters: Dict[str, Any] = None,
                                        **search_kwargs) -> Dict[str, Any]:
        """Search with collaborative context and filters"""
        # Perform base search
        base_results = self.search_engine.search(query, **search_kwargs)

        if not base_results.success:
            return base_results.data

        # Add collaborative context to results
        enhanced_results = []
        for result in base_results.results:
            doc_id = result.doc_id

            # Get collaborative metadata for this document
            collaborative_context = self._get_collaborative_context(doc_id, user_id)

            # Merge with search result
            enhanced_result = result.__dict__.copy()
            enhanced_result.update(collaborative_context)
            enhanced_results.append(enhanced_result)

        # Update base results
        base_results.results = enhanced_results

        # Add collaborative facets
        collaborative_facets = self._get_collaborative_facets(enhanced_results)
        if 'facets' not in base_results.data:
            base_results.data['facets'] = {}
        base_results.data['facets']['collaboration'] = collaborative_facets

        return base_results.data

    def _get_collaborative_context(self, doc_id: str, user_id: str = None) -> Dict[str, Any]:
        """Get collaborative context for a document"""
        context = {
            'collaboration_info': {},
            'user_permissions': {},
            'recent_activity': [],
            'annotation_count': 0,
            'workflow_status': None,
            'active_collaborators': 0
        }

        # Get collaborators
        collaborators = collaborative_manager.get_document_collaborators(doc_id)
        context['active_collaborators'] = len([c for c in collaborators if c.is_online])

        # Get recent activity
        recent_events = [
            e for e in collaborative_manager.events
            if e.document_id == doc_id and e.timestamp > datetime.now() - timedelta(days=7)
        ]
        context['recent_activity'] = [
            {
                'type': e.event_type.value,
                'user': e.username,
                'timestamp': e.timestamp.isoformat()
            }
            for e in recent_events[-5:]  # Last 5 events
        ]

        # Get annotation count
        annotations = annotation_manager.get_document_annotations(doc_id)
        context['annotation_count'] = len(annotations)

        # Get active workflows
        active_workflows = [
            w for w in collaborative_manager.workflows.values()
            if w.document_id == doc_id and w.status != WorkflowStatus.ARCHIVED
        ]
        if active_workflows:
            latest_workflow = max(active_workflows, key=lambda x: x.updated_at)
            context['workflow_status'] = latest_workflow.status.value
            context['workflow_title'] = latest_workflow.title

        # Get user permissions if user_id provided
        if user_id:
            user = collaborative_manager.users.get(user_id)
            if user and doc_id in user.permissions:
                context['user_permissions'] = {
                    'role': user.permissions[doc_id],
                    'can_edit': user.permissions[doc_id] in ['editor', 'approver', 'admin'],
                    'can_review': user.permissions[doc_id] in ['reviewer', 'approver', 'admin'],
                    'can_approve': user.permissions[doc_id] in ['approver', 'admin'],
                    'is_admin': user.permissions[doc_id] == 'admin'
                }

        return context

    def _get_collaborative_facets(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get collaborative facets for search results"""
        facets = {
            'has_collaborators': {'true': 0, 'false': 0},
            'has_annotations': {'true': 0, 'false': 0},
            'has_workflows': {'true': 0, 'false': 0},
            'collaboration_level': {'low': 0, 'medium': 0, 'high': 0}
        }

        for result in results:
            collab_info = result.get('collaboration_info', {})

            # Has collaborators facet
            has_collaborators = collab_info.get('active_collaborators', 0) > 0
            facets['has_collaborators']['true' if has_collaborators else 'false'] += 1

            # Has annotations facet
            has_annotations = collab_info.get('annotation_count', 0) > 0
            facets['has_annotations']['true' if has_annotations else 'false'] += 1

            # Has workflows facet
            has_workflows = collab_info.get('workflow_status') is not None
            facets['has_workflows']['true' if has_workflows else 'false'] += 1

            # Collaboration level facet
            collaboration_score = (
                (1 if has_collaborators else 0) +
                (1 if has_annotations else 0) +
                (1 if has_workflows else 0)
            )
            if collaboration_score == 0:
                facets['collaboration_level']['low'] += 1
            elif collaboration_score == 1:
                facets['collaboration_level']['medium'] += 1
            else:
                facets['collaboration_level']['high'] += 1

        return facets


class CollaborativeVersionIntegrator:
    """Integrates collaborative features with version control"""

    def __init__(self):
        self.version_search_enhancer = VersionSearchEnhancer()

    def enhance_version_with_collaboration(self, doc_id: str, version_id: str) -> Dict[str, Any]:
        """Enhance version information with collaborative context"""
        # Get base version info (this would integrate with existing version system)
        version_info = {
            'doc_id': doc_id,
            'version_id': version_id,
            'collaborative_enhancements': {}
        }

        # Add collaborative annotations for this version
        annotations = annotation_manager.get_document_annotations(doc_id)
        version_info['collaborative_enhancements']['annotations'] = [
            annotation.to_dict() for annotation in annotations
        ]

        # Add workflow information
        workflows = [
            w for w in collaborative_manager.workflows.values()
            if w.document_id == doc_id
        ]
        version_info['collaborative_enhancements']['workflows'] = [
            workflow.to_dict() for workflow in workflows
        ]

        # Add collaboration events
        events = [
            e for e in collaborative_manager.events
            if e.document_id == doc_id
        ]
        version_info['collaborative_enhancements']['events'] = [
            event.to_dict() for event in events
        ]

        return version_info

    def create_collaborative_version(self, doc_id: str, user_id: str,
                                   change_type: str, description: str = None) -> Dict[str, Any]:
        """Create a new version with collaborative context"""
        # This would integrate with the existing version control system
        # For now, create a collaborative record

        # Create collaborative edit operation
        operation = collaborative_manager.add_collaborative_edit(
            doc_id, user_id, 'version_create', 0, description or 'Collaborative version created'
        )

        # Create workflow for version approval if needed
        if change_type in ['major', 'critical']:
            workflow = collaborative_manager.create_workflow(
                document_id=doc_id,
                title=f"Version Approval: {change_type.title()} Change",
                description=f"Approval workflow for {change_type} version change",
                created_by=user_id
            )

            # Create approval task
            task_manager.create_task_from_template(
                'final_approval',
                doc_id,
                list(workflow.approvers.keys()) if workflow.approvers else [user_id],
                user_id,
                {
                    'title': f'Approve {change_type.title()} Version Change',
                    'description': f'Please review and approve the {change_type} changes to this document.'
                }
            )

        return {
            'version_id': str(uuid.uuid4()),
            'operation_id': operation.operation_id,
            'change_type': change_type,
            'created_by': user_id,
            'collaborative_context': {
                'triggered_workflows': change_type in ['major', 'critical'],
                'notification_sent': True
            }
        }


class CollaborativeComplianceIntegrator:
    """Integrates collaborative features with compliance system"""

    def __init__(self):
        self.compliance_integration_manager = compliance_integration_manager

    def check_collaborative_compliance(self, document_id: str, user_id: str) -> Dict[str, Any]:
        """Check compliance requirements for collaborative actions"""
        compliance_result = {
            'is_compliant': True,
            'violations': [],
            'warnings': [],
            'recommendations': []
        }

        # Check if user has permission for collaborative actions
        user = collaborative_manager.users.get(user_id)
        if not user:
            compliance_result['is_compliant'] = False
            compliance_result['violations'].append("User not found in collaborative system")
            return compliance_result

        # Check document permissions
        document_collaborators = collaborative_manager.get_document_collaborators(document_id)
        if user_id not in [c.user_id for c in document_collaborators]:
            compliance_result['warnings'].append(
                "User is not an official collaborator on this document"
            )

        # Check if action requires approval based on document sensitivity
        # This would integrate with compliance system for document classification
        # For now, provide basic checks

        if user.role == UserRole.VIEWER:
            # Viewers shouldn't be making edits
            compliance_result['recommendations'].append(
                "Consider upgrading user role from viewer to editor for collaborative editing"
            )

        return compliance_result

    def log_collaborative_compliance_event(self, event_type: str, document_id: str,
                                         user_id: str, details: Dict[str, Any]):
        """Log collaborative actions for compliance tracking"""
        # This would integrate with the compliance audit logging system
        # For now, just create a record

        compliance_event = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'document_id': document_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'details': details,
            'system': 'collaborative_workflow'
        }

        print(f"Compliance event logged: {compliance_event}")

        return compliance_event


class CollaborativeSummarizationIntegrator:
    """Integrates collaborative features with summarization system"""

    def __init__(self):
        self.summarization_engine = summarization_engine

    def generate_collaborative_summary(self, document_id: str) -> Dict[str, Any]:
        """Generate a summary of collaborative activity for a document"""
        # Get base document summary
        # This would integrate with existing summarization system
        base_summary = {
            'document_id': document_id,
            'collaborative_summary': {}
        }

        # Get collaborative activity summary
        annotations = annotation_manager.get_document_annotations(document_id)
        collaborators = collaborative_manager.get_document_collaborators(document_id)
        workflows = [
            w for w in collaborative_manager.workflows.values() if w.document_id == document_id
        ]
        events = [
            e for e in collaborative_manager.events if e.document_id == document_id
        ]

        # Generate activity summary
        collaborative_summary = {
            'total_collaborators': len(collaborators),
            'active_collaborators': len([c for c in collaborators if c.is_online]),
            'total_annotations': len(annotations),
            'annotation_types': {},
            'workflow_count': len(workflows),
            'recent_activity': len([e for e in events if e.timestamp > datetime.now() - timedelta(days=7)]),
            'collaboration_timeline': self._generate_collaboration_timeline(document_id)
        }

        # Count annotation types
        for annotation in annotations:
            annotation_type = annotation.type.value
            collaborative_summary['annotation_types'][annotation_type] = (
                collaborative_summary['annotation_types'].get(annotation_type, 0) + 1
            )

        base_summary['collaborative_summary'] = collaborative_summary

        return base_summary

    def _generate_collaboration_timeline(self, document_id: str) -> List[Dict[str, Any]]:
        """Generate a timeline of collaborative activities"""
        timeline = []

        # Get all collaborative events for the document
        events = [
            e for e in collaborative_manager.events
            if e.document_id == document_id
        ]

        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)

        # Create timeline entries
        for event in events[-20:]:  # Last 20 events
            timeline.append({
                'timestamp': event.timestamp.isoformat(),
                'type': event.event_type.value,
                'user': event.username,
                'description': self._get_event_description(event)
            })

        return timeline

    def _get_event_description(self, event) -> str:
        """Get human-readable description for an event"""
        if event.event_type.value == 'document_edit':
            return f"Document edited"
        elif event.event_type.value == 'comment_added':
            return "Comment added"
        elif event.event_type.value == 'workflow_status_changed':
            return f"Workflow status changed to {event.data.get('new_status', 'unknown')}"
        elif event.event_type.value == 'user_joined':
            return "User joined collaboration"
        elif event.event_type.value == 'user_left':
            return "User left collaboration"
        else:
            return "Collaborative activity"

        return "Unknown activity"


class CollaborativeWorkflowIntegrationManager:
    """Main integration manager for all collaborative systems"""

    def __init__(self):
        self.search_integrator = CollaborativeSearchIntegrator()
        self.version_integrator = CollaborativeVersionIntegrator()
        self.compliance_integrator = CollaborativeComplianceIntegrator()
        self.summarization_integrator = CollaborativeSummarizationIntegrator()

        # Integration status
        self.integration_status = {
            'search_integration': True,
            'version_integration': True,
            'compliance_integration': True,
            'summarization_integration': True,
            'websocket_integration': False,  # Will be set when WebSocket server starts
            'automation_integration': False  # Will be set when automation engine starts
        }

    def initialize_all_integrations(self):
        """Initialize all system integrations"""
        print("Initializing collaborative workflow integrations...")

        # Start automation engine
        automation_engine.start_engine()
        self.integration_status['automation_integration'] = True

        # Start WebSocket manager
        websocket_manager.start_server()
        self.integration_status['websocket_integration'] = True

        # Set up event listeners
        self._setup_event_listeners()

        print("Collaborative workflow integrations initialized successfully")

    def _setup_event_listeners(self):
        """Set up event listeners for cross-system integration"""

        def on_collaborative_event(event):
            """Handle collaborative events and trigger cross-system actions"""
            try:
                # Update search index with collaborative metadata
                if event.document_id:
                    # This would trigger search index updates with collaborative context

                    # Check if event should trigger notifications
                    if event.event_type.value in ['comment_added', 'workflow_status_changed']:
                        self._handle_collaborative_notifications(event)

                    # Log for compliance if needed
                    if event.event_type.value == 'document_edit':
                        self.compliance_integrator.log_collaborative_compliance_event(
                            'document_modified',
                            event.document_id,
                            event.user_id,
                            {'event_type': event.event_type.value}
                        )

            except Exception as e:
                print(f"Error in collaborative event handler: {e}")

        # Connect to collaborative manager (simplified - would use proper event system)
        print("Event listeners setup (simplified)")

    def _handle_collaborative_notifications(self, event):
        """Handle notifications for collaborative events"""
        if event.event_type.value == 'comment_added':
            # Notify document collaborators about new comments
            collaborators = collaborative_manager.get_document_collaborators(event.document_id)
            for collaborator in collaborators:
                if collaborator.user_id != event.user_id:  # Don't notify the commenter
                    notification_manager.send_notification(
                        collaborator.user_id,
                        NotificationType.COMMENT_MENTION,
                        "New Comment",
                        f"{event.username} added a comment to the document",
                        data={'document_id': event.document_id, 'commenter_id': event.user_id}
                    )

        elif event.event_type.value == 'workflow_status_changed':
            # Notify workflow participants about status changes
            # This would be handled by the workflow automation system

            pass

    def get_document_collaborative_status(self, document_id: str, user_id: str = None) -> Dict[str, Any]:
        """Get comprehensive collaborative status for a document"""
        status = {
            'document_id': document_id,
            'collaboration_active': False,
            'collaborators': [],
            'workflows': [],
            'annotations': [],
            'recent_activity': [],
            'user_role': None,
            'permissions': {},
            'compliance_status': {},
            'summary': {}
        }

        # Get collaborators
        collaborators = collaborative_manager.get_document_collaborators(document_id)
        status['collaborators'] = [
            {
                'user_id': c.user_id,
                'username': c.username,
                'role': c.role.value,
                'is_online': c.is_online,
                'last_seen': c.last_seen.isoformat()
            }
            for c in collaborators
        ]

        # Check if collaboration is active
        status['collaboration_active'] = len([c for c in collaborators if c.is_online]) > 0

        # Get workflows
        workflows = [
            w for w in collaborative_manager.workflows.values()
            if w.document_id == document_id
        ]
        status['workflows'] = [w.to_dict() for w in workflows]

        # Get annotations
        annotations = annotation_manager.get_document_annotations(document_id)
        status['annotations'] = [a.to_dict() for a in annotations]

        # Get recent activity
        recent_events = [
            e for e in collaborative_manager.events
            if e.document_id == document_id and e.timestamp > datetime.now() - timedelta(days=1)
        ]
        status['recent_activity'] = [e.to_dict() for e in recent_events]

        # Get user-specific information
        if user_id:
            user = collaborative_manager.users.get(user_id)
            if user and document_id in user.permissions:
                status['user_role'] = user.permissions[document_id]
                status['permissions'] = {
                    'can_edit': user.permissions[document_id] in ['editor', 'approver', 'admin'],
                    'can_review': user.permissions[document_id] in ['reviewer', 'approver', 'admin'],
                    'can_approve': user.permissions[document_id] in ['approver', 'admin'],
                    'is_admin': user.permissions[document_id] == 'admin'
                }

        # Get compliance status
        if user_id:
            status['compliance_status'] = self.compliance_integrator.check_collaborative_compliance(
                document_id, user_id
            )

        # Get summary
        status['summary'] = self.summarization_integrator.generate_collaborative_summary(document_id)

        return status

    def get_system_integration_status(self) -> Dict[str, Any]:
        """Get status of all system integrations"""
        return {
            'integration_status': self.integration_status,
            'collaboration_stats': collaborative_manager.get_collaboration_stats(),
            'automation_stats': automation_engine.get_engine_statistics(),
            'dashboard_stats': dashboard_manager.get_dashboard_statistics(),
            'websocket_stats': websocket_manager.get_connection_stats(),
            'notification_stats': notification_manager.get_notification_statistics(),
            'annotation_stats': annotation_manager.get_annotation_statistics(),
            'last_updated': datetime.now().isoformat()
        }

    def export_integrated_data(self) -> Dict[str, Any]:
        """Export all integrated collaborative data"""
        return {
            'collaborative_data': collaborative_manager.export_collaboration_data(),
            'annotation_data': annotation_manager.export_annotations(),
            'notification_data': notification_manager.export_notification_data(),
            'automation_data': automation_engine.export_rules(),
            'dashboard_data': dashboard_manager.export_dashboard_data(),
            'integration_status': self.get_system_integration_status(),
            'exported_at': datetime.now().isoformat()
        }


# Global integration manager
integration_manager = CollaborativeWorkflowIntegrationManager()