"""
Document Lifecycle Management System
Manages document states, workflows, and automated transitions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import threading
from enum import Enum

from .version_control import version_manager, DocumentVersion, ChangeRequest
from .document_monitor import lifecycle_manager as basic_lifecycle_manager

class DocumentStatus(Enum):
    """Document lifecycle statuses"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    RETIRED = "retired"

class TransitionRule:
    """Defines rules for lifecycle transitions"""

    def __init__(self, from_status: str, to_status: str,
                 conditions: List[Callable] = None,
                 actions: List[Callable] = None,
                 requires_approval: bool = False,
                 auto_approve_roles: List[str] = None):
        self.from_status = from_status
        self.to_status = to_status
        self.conditions = conditions or []
        self.actions = actions or []
        self.requires_approval = requires_approval
        self.auto_approve_roles = auto_approve_roles or []

@dataclass
class WorkflowStep:
    """Represents a step in a document workflow"""
    step_id: str
    step_name: str
    assigned_role: str
    timeout_hours: int = 72
    is_required: bool = True
    can_skip: bool = False

    # Step actions
    actions: List[str] = field(default_factory=list)  # approve, reject, review, etc.

@dataclass
class DocumentWorkflow:
    """Defines a workflow for document processing"""
    workflow_id: str
    workflow_name: str
    document_types: List[str]  # Types of documents this workflow applies to
    steps: List[WorkflowStep]
    is_active: bool = True

    # Workflow settings
    allow_parallel_approval: bool = False
    require_all_approvals: bool = True
    auto_archive_days: int = 365

@dataclass
class WorkflowInstance:
    """Instance of a workflow for a specific document"""
    instance_id: str
    workflow_id: str
    doc_id: str
    version_id: str
    status: str = "active"

    # Progress tracking
    current_step: int = 0
    completed_steps: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Assignment and tracking
    assigned_users: Dict[str, str] = field(default_factory=dict)  # step_id -> user_id
    step_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class AdvancedLifecycleManager:
    """Advanced lifecycle management with workflows and automation"""

    def __init__(self):
        self.workflows: Dict[str, DocumentWorkflow] = {}
        self.workflow_instances: Dict[str, WorkflowInstance] = {}
        self.transition_rules: List[TransitionRule] = []
        self._lock = threading.Lock()

        # Initialize default workflows and rules
        self._initialize_default_workflows()
        self._initialize_transition_rules()

    def _initialize_default_workflows(self):
        """Initialize default document workflows"""

        # Standard document workflow
        standard_workflow = DocumentWorkflow(
            workflow_id="standard_approval",
            workflow_name="Standard Document Approval",
            document_types=["policy", "procedure", "guideline", "default"],
            steps=[
                WorkflowStep("draft_review", "Draft Review", "author", 48, True, False),
                WorkflowStep("technical_review", "Technical Review", "technical_reviewer", 72, True, False),
                WorkflowStep("final_approval", "Final Approval", "manager", 48, True, False),
                WorkflowStep("publication", "Publication", "publisher", 24, False, True)
            ],
            allow_parallel_approval=False,
            require_all_approvals=True,
            auto_archive_days=365
        )

        # Quick approval workflow for minor changes
        quick_workflow = DocumentWorkflow(
            workflow_id="quick_approval",
            workflow_name="Quick Approval",
            document_types=["update", "revision"],
            steps=[
                WorkflowStep("review", "Review", "reviewer", 24, True, False),
                WorkflowStep("approval", "Approval", "manager", 24, True, False)
            ],
            allow_parallel_approval=True,
            require_all_approvals=False,
            auto_archive_days=180
        )

        self.workflows[standard_workflow.workflow_id] = standard_workflow
        self.workflows[quick_workflow.workflow_id] = quick_workflow

    def _initialize_transition_rules(self):
        """Initialize transition rules for lifecycle management"""

        # Define transition rules with conditions and actions
        rules = [
            TransitionRule(
                "draft", "under_review",
                requires_approval=False,
                actions=[self._notify_reviewers]
            ),
            TransitionRule(
                "under_review", "approved",
                requires_approval=True,
                auto_approve_roles=["admin", "manager"],
                actions=[self._archive_previous_versions, self._update_search_index]
            ),
            TransitionRule(
                "approved", "published",
                requires_approval=False,
                actions=[self._publish_document, self._notify_stakeholders]
            ),
            TransitionRule(
                "published", "deprecated",
                requires_approval=True,
                actions=[self._create_deprecation_notice]
            ),
            TransitionRule(
                "deprecated", "archived",
                requires_approval=False,
                actions=[self._move_to_archive]
            ),
            TransitionRule(
                "archived", "retired",
                requires_approval=True,
                actions=[self._schedule_deletion]
            )
        ]

        self.transition_rules = rules

    def transition_document(self, version_id: str, new_status: str,
                          user_id: str, reason: str = None) -> bool:
        """Transition a document to a new lifecycle status"""
        version = version_manager.db.get_version(version_id)
        if not version:
            return False

        current_status = version.lifecycle_status

        # Find applicable transition rule
        transition_rule = self._find_transition_rule(current_status, new_status)
        if not transition_rule:
            print(f"No transition rule found for {current_status} -> {new_status}")
            return False

        # Check conditions
        if not self._check_transition_conditions(transition_rule, version, user_id):
            return False

        # Check if approval is required
        if transition_rule.requires_approval:
            if not self._check_approval_permission(transition_rule, user_id):
                # Create approval request
                return self._create_approval_request(version_id, new_status, user_id, reason)

        # Execute transition
        return self._execute_transition(version, new_status, user_id, reason)

    def _find_transition_rule(self, from_status: str, to_status: str) -> Optional[TransitionRule]:
        """Find the transition rule for a status change"""
        for rule in self.transition_rules:
            if rule.from_status == from_status and rule.to_status == to_status:
                return rule
        return None

    def _check_transition_conditions(self, rule: TransitionRule, version: DocumentVersion,
                                   user_id: str) -> bool:
        """Check if transition conditions are met"""
        for condition in rule.conditions:
            if not condition(version, user_id):
                return False
        return True

    def _check_approval_permission(self, rule: TransitionRule, user_id: str) -> bool:
        """Check if user has permission to auto-approve"""
        # This would integrate with your user management system
        # For now, assume admin/manager roles can auto-approve
        user_roles = self._get_user_roles(user_id)
        return any(role in rule.auto_approve_roles for role in user_roles)

    def _execute_transition(self, version: DocumentVersion, new_status: str,
                          user_id: str, reason: str) -> bool:
        """Execute the actual transition"""
        old_status = version.lifecycle_status

        # Update version status
        version.lifecycle_status = new_status

        # Add transition metadata
        if not hasattr(version, 'transition_history'):
            version.transition_history = []

        version.transition_history.append({
            'from_status': old_status,
            'to_status': new_status,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'reason': reason
        })

        # Save updated version
        success = version_manager.db.save_version(version)

        if success:
            # Execute transition actions
            transition_rule = self._find_transition_rule(old_status, new_status)
            if transition_rule:
                self._execute_transition_actions(transition_rule.actions, version, user_id)

            print(f"Document {version.doc_id} transitioned from {old_status} to {new_status}")

        return success

    def _execute_transition_actions(self, actions: List[Callable],
                                  version: DocumentVersion, user_id: str):
        """Execute actions associated with a transition"""
        for action in actions:
            try:
                action(version, user_id)
            except Exception as e:
                print(f"Error executing transition action: {e}")

    def _create_approval_request(self, version_id: str, target_status: str,
                               requested_by: str, reason: str) -> bool:
        """Create an approval request for a transition"""
        request = ChangeRequest(
            request_id=f"approval_{version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            doc_id=version_manager.db.get_version(version_id).doc_id,
            requested_by=requested_by,
            request_type="status_change",
            target_version_id=version_id,
            title=f"Status Change Approval: {target_status}",
            description=reason or f"Request to change document status to {target_status}",
            priority="medium",
            status="pending",
            created_at=datetime.now()
        )

        return version_manager.db.save_change_request(request)

    def start_workflow(self, doc_id: str, version_id: str,
                     workflow_id: str = None) -> Optional[str]:
        """Start a workflow for a document"""
        version = version_manager.db.get_version(version_id)
        if not version:
            return None

        # Determine appropriate workflow
        if not workflow_id:
            workflow_id = self._select_workflow_for_document(version)

        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None

        # Create workflow instance
        instance = WorkflowInstance(
            instance_id=f"workflow_{doc_id}_{version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            workflow_id=workflow_id,
            doc_id=doc_id,
            version_id=version_id
        )

        with self._lock:
            self.workflow_instances[instance.instance_id] = instance

        # Start first step
        self._execute_workflow_step(instance, 0)

        return instance.instance_id

    def _select_workflow_for_document(self, version: DocumentVersion) -> str:
        """Select appropriate workflow for a document"""
        # This would use document metadata to select workflow
        # For now, use standard workflow as default
        return "standard_approval"

    def _execute_workflow_step(self, instance: WorkflowInstance, step_index: int):
        """Execute a specific workflow step"""
        workflow = self.workflows.get(instance.workflow_id)
        if not workflow or step_index >= len(workflow.steps):
            return

        step = workflow.steps[step_index]
        instance.current_step = step_index

        # Assign step to user(s)
        if step.assigned_role:
            assigned_users = self._get_users_by_role(step.assigned_role)
            for user_id in assigned_users:
                instance.assigned_users[step.step_id] = user_id

        # Set step timeout
        timeout_at = datetime.now() + timedelta(hours=step.timeout_hours)

        print(f"Executing workflow step: {step.step_name} for document {instance.doc_id}")

    def complete_workflow_step(self, instance_id: str, step_id: str,
                             user_id: str, action: str, notes: str = None) -> bool:
        """Complete a workflow step"""
        instance = self.workflow_instances.get(instance_id)
        if not instance:
            return False

        workflow = self.workflows.get(instance.workflow_id)
        if not workflow:
            return False

        # Find the step
        step = next((s for s in workflow.steps if s.step_id == step_id), None)
        if not step:
            return False

        # Record step result
        instance.step_results[step_id] = {
            'completed_at': datetime.now().isoformat(),
            'completed_by': user_id,
            'action': action,
            'notes': notes
        }

        # Mark step as completed
        if step_id not in instance.completed_steps:
            instance.completed_steps.append(step_id)

        # Move to next step or complete workflow
        current_step_index = instance.current_step
        if current_step_index < len(workflow.steps) - 1:
            next_step_index = current_step_index + 1
            self._execute_workflow_step(instance, next_step_index)
        else:
            # Workflow completed
            instance.status = "completed"
            instance.completed_at = datetime.now()

            # Transition document to final status
            final_version = version_manager.db.get_version(instance.version_id)
            if final_version:
                self.transition_document(
                    instance.version_id, "published", user_id,
                    "Workflow completed successfully"
                )

        return True

    def get_workflow_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow instance"""
        instance = self.workflow_instances.get(instance_id)
        if not instance:
            return None

        workflow = self.workflows.get(instance.workflow_id)
        if not workflow:
            return None

        return {
            'instance_id': instance.instance_id,
            'workflow_name': workflow.workflow_name,
            'status': instance.status,
            'current_step': instance.current_step,
            'total_steps': len(workflow.steps),
            'progress_percent': (len(instance.completed_steps) / len(workflow.steps)) * 100,
            'started_at': instance.started_at.isoformat(),
            'completed_at': instance.completed_at.isoformat() if instance.completed_at else None,
            'assigned_users': instance.assigned_users,
            'step_results': instance.step_results
        }

    # Transition action methods
    def _notify_reviewers(self, version: DocumentVersion, user_id: str):
        """Notify reviewers about document under review"""
        print(f"Notifying reviewers about document {version.doc_id} version {version.version_number}")

    def _archive_previous_versions(self, version: DocumentVersion, user_id: str):
        """Archive previous versions of the document"""
        versions = version_manager.db.get_document_versions(version.doc_id)
        for v in versions:
            if v.version_id != version.version_id and v.lifecycle_status == "published":
                v.lifecycle_status = "archived"
                version_manager.db.save_version(v)

    def _update_search_index(self, version: DocumentVersion, user_id: str):
        """Update search index with new document status"""
        print(f"Updating search index for document {version.doc_id}")

    def _publish_document(self, version: DocumentVersion, user_id: str):
        """Publish document to production"""
        print(f"Publishing document {version.doc_id} version {version.version_number}")

    def _notify_stakeholders(self, version: DocumentVersion, user_id: str):
        """Notify stakeholders about document publication"""
        print(f"Notifying stakeholders about published document {version.doc_id}")

    def _create_deprecation_notice(self, version: DocumentVersion, user_id: str):
        """Create deprecation notice for document"""
        print(f"Creating deprecation notice for document {version.doc_id}")

    def _move_to_archive(self, version: DocumentVersion, user_id: str):
        """Move document to archive"""
        print(f"Moving document {version.doc_id} to archive")

    def _schedule_deletion(self, version: DocumentVersion, user_id: str):
        """Schedule document for deletion"""
        print(f"Scheduling document {version.doc_id} for deletion")

    def _get_user_roles(self, user_id: str) -> List[str]:
        """Get roles for a user (integrate with user management system)"""
        # Placeholder - integrate with your user management system
        return ["user"]  # Default role

    def _get_users_by_role(self, role: str) -> List[str]:
        """Get users with a specific role"""
        # Placeholder - integrate with your user management system
        return ["default_user"]

    def get_document_lifecycle_info(self, doc_id: str) -> Dict[str, Any]:
        """Get comprehensive lifecycle information for a document"""
        versions = version_manager.db.get_document_versions(doc_id)

        # Get current active version
        current_version = None
        for version in versions:
            if version.lifecycle_status in ["published", "approved"]:
                current_version = version
                break

        if not current_version:
            current_version = versions[0] if versions else None

        # Get workflow instances
        active_workflows = [
            instance for instance in self.workflow_instances.values()
            if instance.doc_id == doc_id and instance.status == "active"
        ]

        # Get lifecycle history
        lifecycle_events = []
        for version in versions:
            if hasattr(version, 'transition_history') and version.transition_history:
                for transition in version.transition_history:
                    lifecycle_events.append({
                        'version_id': version.version_id,
                        'version_number': version.version_number,
                        'event_type': 'transition',
                        'from_status': transition['from_status'],
                        'to_status': transition['to_status'],
                        'timestamp': transition['timestamp'],
                        'user_id': transition['user_id']
                    })

        # Sort by timestamp
        lifecycle_events.sort(key=lambda x: x['timestamp'], reverse=True)

        return {
            'doc_id': doc_id,
            'current_version': current_version.to_dict() if current_version else None,
            'total_versions': len(versions),
            'active_workflows': len(active_workflows),
            'lifecycle_events': lifecycle_events[:20],  # Last 20 events
            'available_transitions': self._get_available_transitions(current_version) if current_version else []
        }

    def _get_available_transitions(self, version: DocumentVersion) -> List[str]:
        """Get available transitions for a version"""
        available = []
        for rule in self.transition_rules:
            if rule.from_status == version.lifecycle_status:
                available.append(rule.to_status)
        return available

    def schedule_automated_transitions(self):
        """Schedule automated lifecycle transitions"""
        def check_and_transition():
            current_time = datetime.now()

            # Check for documents that should be auto-archived
            for workflow in self.workflows.values():
                if workflow.auto_archive_days > 0:
                    cutoff_date = current_time - timedelta(days=workflow.auto_archive_days)

                    # Find documents that should be archived
                    versions = version_manager.db.get_document_versions("")
                    for version in versions:
                        if (version.lifecycle_status == "published" and
                            version.created_at < cutoff_date):

                            # Check if document is in this workflow
                            instance = next(
                                (inst for inst in self.workflow_instances.values()
                                 if inst.version_id == version.version_id),
                                None
                            )

                            if instance and instance.workflow_id == workflow.workflow_id:
                                self.transition_document(
                                    version.version_id, "deprecated", "system",
                                    f"Auto-archived after {workflow.auto_archive_days} days"
                                )

        # Schedule to run daily
        import schedule
        schedule.every().day.at("02:00").do(check_and_transition)

        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour

        threading.Thread(target=run_schedule, daemon=True).start()

# Global lifecycle manager instance
advanced_lifecycle_manager = AdvancedLifecycleManager()

# Start automated transitions
advanced_lifecycle_manager.schedule_automated_transitions()