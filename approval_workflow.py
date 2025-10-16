"""
Change Notification and Approval Workflow System
Manages document change requests, approval processes, and notifications
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import threading
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import queue
import time

from .version_control import version_manager, DocumentVersion, ChangeRequest
from .lifecycle_manager import advanced_lifecycle_manager

@dataclass
class NotificationTemplate:
    """Template for different types of notifications"""
    template_id: str
    name: str
    subject: str
    body_template: str
    channels: List[str] = field(default_factory=lambda: ["email"])

@dataclass
class NotificationRule:
    """Rules for when to send notifications"""
    rule_id: str
    event_type: str  # version_created, status_changed, workflow_step, etc.
    conditions: Dict[str, Any]
    recipients: List[str] = field(default_factory=list)  # user_ids or roles
    template_id: str = None
    priority: str = "normal"  # low, normal, high, urgent

@dataclass
class ApprovalChain:
    """Defines an approval chain for document changes"""
    chain_id: str
    name: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    requires_all_approvals: bool = True
    allow_delegation: bool = True

@dataclass
class ApprovalRequest:
    """Represents an approval request in the workflow"""
    request_id: str
    change_request_id: str
    approver_id: str
    step_order: int
    status: str = "pending"  # pending, approved, rejected, delegated
    assigned_at: datetime = field(default_factory=datetime.now)
    responded_at: Optional[datetime] = None
    response_notes: Optional[str] = None
    delegated_to: Optional[str] = None

class NotificationManager:
    """Manages all notifications for the version control system"""

    def __init__(self):
        self.templates: Dict[str, NotificationTemplate] = {}
        self.rules: Dict[str, NotificationRule] = {}
        self.notification_queue = queue.Queue()
        self._lock = threading.Lock()

        # Initialize default templates and rules
        self._initialize_default_templates()
        self._initialize_default_rules()

        # Start notification processor
        self._start_notification_processor()

    def _initialize_default_templates(self):
        """Initialize default notification templates"""

        templates = [
            NotificationTemplate(
                template_id="version_created",
                name="Version Created",
                subject="New Document Version Created: {document_id}",
                body_template="""
A new version has been created for document {document_id}.

Version Details:
- Version Number: {version_number}
- Created By: {created_by}
- Created At: {created_at}
- Change Type: {change_type}
- Description: {change_description}

View Document: {document_url}
"""
            ),
            NotificationTemplate(
                template_id="status_changed",
                name="Status Changed",
                subject="Document Status Changed: {document_id}",
                body_template="""
Document {document_id} status has changed.

Status Change Details:
- Document: {document_id}
- Previous Status: {old_status}
- New Status: {new_status}
- Changed By: {changed_by}
- Changed At: {changed_at}
- Reason: {reason}

Current Version: {version_number}
View Document: {document_url}
"""
            ),
            NotificationTemplate(
                template_id="approval_required",
                name="Approval Required",
                subject="Approval Required: {document_id} - {change_type}",
                body_template="""
Your approval is required for a document change.

Approval Request Details:
- Document: {document_id}
- Change Type: {change_type}
- Requested By: {requested_by}
- Description: {description}
- Priority: {priority}

Please review and approve/reject this request.
Approval URL: {approval_url}
Due Date: {due_date}
"""
            ),
            NotificationTemplate(
                template_id="workflow_step_completed",
                name="Workflow Step Completed",
                subject="Workflow Step Completed: {document_id}",
                body_template="""
A workflow step has been completed for document {document_id}.

Step Details:
- Step: {step_name}
- Completed By: {completed_by}
- Completed At: {completed_at}
- Status: {step_status}
- Notes: {notes}

Next Step: {next_step}
Overall Progress: {progress_percent}%

View Workflow: {workflow_url}
"""
            )
        ]

        for template in templates:
            self.templates[template.template_id] = template

    def _initialize_default_rules(self):
        """Initialize default notification rules"""

        rules = [
            NotificationRule(
                rule_id="notify_on_version_create",
                event_type="version_created",
                conditions={"change_type": ["major", "minor"]},
                recipients=["document_owner", "stakeholders"],
                template_id="version_created",
                priority="normal"
            ),
            NotificationRule(
                rule_id="notify_on_status_change",
                event_type="status_changed",
                conditions={},
                recipients=["document_owner", "followers"],
                template_id="status_changed",
                priority="normal"
            ),
            NotificationRule(
                rule_id="notify_approval_required",
                event_type="approval_required",
                conditions={},
                recipients=["approvers"],
                template_id="approval_required",
                priority="high"
            ),
            NotificationRule(
                rule_id="notify_workflow_progress",
                event_type="workflow_step_completed",
                conditions={},
                recipients=["workflow_participants"],
                template_id="workflow_step_completed",
                priority="normal"
            )
        ]

        for rule in rules:
            self.rules[rule.rule_id] = rule

    def send_notification(self, event_type: str, event_data: Dict[str, Any],
                         recipients: List[str] = None):
        """Send a notification for an event"""
        # Find applicable rules
        applicable_rules = []
        for rule in self.rules.values():
            if rule.event_type == event_type:
                if self._check_rule_conditions(rule, event_data):
                    applicable_rules.append(rule)

        # Queue notifications for each applicable rule
        for rule in applicable_rules:
            notification_data = {
                'rule_id': rule.rule_id,
                'event_type': event_type,
                'event_data': event_data,
                'template_id': rule.template_id,
                'recipients': recipients or rule.recipients,
                'priority': rule.priority,
                'timestamp': datetime.now()
            }

            self.notification_queue.put(notification_data)

    def _check_rule_conditions(self, rule: NotificationRule, event_data: Dict[str, Any]) -> bool:
        """Check if rule conditions are met"""
        for key, expected_values in rule.conditions.items():
            if key in event_data:
                actual_value = event_data.get(key)
                if isinstance(expected_values, list):
                    if actual_value not in expected_values:
                        return False
                elif actual_value != expected_values:
                    return False
        return True

    def _start_notification_processor(self):
        """Start the notification processing thread"""
        def process_notifications():
            while True:
                try:
                    notification_data = self.notification_queue.get(timeout=1)

                    # Process the notification
                    self._process_notification(notification_data)

                    self.notification_queue.task_done()
                    time.sleep(0.1)  # Small delay between notifications

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error processing notification: {e}")

        threading.Thread(target=process_notifications, daemon=True).start()

    def _process_notification(self, notification_data: Dict[str, Any]):
        """Process a single notification"""
        try:
            template_id = notification_data['template_id']
            template = self.templates.get(template_id)

            if not template:
                print(f"Template not found: {template_id}")
                return

            event_data = notification_data['event_data']
            recipients = notification_data['recipients']

            # Render template
            subject = self._render_template(template.subject, event_data)
            body = self._render_template(template.body_template, event_data)

            # Send via configured channels
            for channel in template.channels:
                if channel == "email":
                    self._send_email_notification(recipients, subject, body, notification_data['priority'])
                elif channel == "webhook":
                    self._send_webhook_notification(recipients, subject, body, event_data)

        except Exception as e:
            print(f"Error processing notification: {e}")

    def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """Render template with data"""
        result = template
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        return result

    def _send_email_notification(self, recipients: List[str], subject: str,
                               body: str, priority: str):
        """Send email notification"""
        try:
            # This would integrate with your email system
            # For now, just log the notification
            print(f"EMAIL NOTIFICATION ({priority}): {subject}")
            print(f"To: {', '.join(recipients)}")
            print(f"Body: {body}")
            print("-" * 50)

        except Exception as e:
            print(f"Error sending email notification: {e}")

    def _send_webhook_notification(self, webhooks: List[str], subject: str,
                                 body: str, event_data: Dict[str, Any]):
        """Send webhook notification"""
        try:
            # This would integrate with webhook endpoints
            # For now, just log the notification
            for webhook in webhooks:
                print(f"WEBHOOK NOTIFICATION: {webhook}")
                print(f"Event: {subject}")
                print(f"Data: {json.dumps(event_data)}")

        except Exception as e:
            print(f"Error sending webhook notification: {e}")

class ApprovalWorkflowManager:
    """Manages approval workflows for document changes"""

    def __init__(self):
        self.approval_chains: Dict[str, ApprovalChain] = {}
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.notification_manager = NotificationManager()
        self._lock = threading.Lock()

        # Initialize default approval chains
        self._initialize_default_chains()

    def _initialize_default_chains(self):
        """Initialize default approval chains"""

        # Standard approval chain
        standard_chain = ApprovalChain(
            chain_id="standard_approval",
            name="Standard Approval Chain",
            steps=[
                {
                    'step_order': 1,
                    'role': 'technical_reviewer',
                    'description': 'Technical review and validation',
                    'timeout_hours': 48
                },
                {
                    'step_order': 2,
                    'role': 'manager',
                    'description': 'Manager approval',
                    'timeout_hours': 24
                },
                {
                    'step_order': 3,
                    'role': 'compliance_officer',
                    'description': 'Compliance review',
                    'timeout_hours': 72,
                    'required': False
                }
            ],
            requires_all_approvals=False,
            allow_delegation=True
        )

        # Critical change approval chain
        critical_chain = ApprovalChain(
            chain_id="critical_approval",
            name="Critical Change Approval Chain",
            steps=[
                {
                    'step_order': 1,
                    'role': 'technical_reviewer',
                    'description': 'Technical review',
                    'timeout_hours': 24
                },
                {
                    'step_order': 2,
                    'role': 'security_officer',
                    'description': 'Security review',
                    'timeout_hours': 48
                },
                {
                    'step_order': 3,
                    'role': 'manager',
                    'description': 'Management approval',
                    'timeout_hours': 24
                },
                {
                    'step_order': 4,
                    'role': 'executive',
                    'description': 'Executive approval',
                    'timeout_hours': 72
                }
            ],
            requires_all_approvals=True,
            allow_delegation=False
        )

        self.approval_chains[standard_chain.chain_id] = standard_chain
        self.approval_chains[critical_chain.chain_id] = critical_chain

    def create_approval_request(self, change_request_id: str,
                              approver_id: str, step_order: int) -> str:
        """Create an approval request"""
        request_id = f"approval_{change_request_id}_{step_order}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        request = ApprovalRequest(
            request_id=request_id,
            change_request_id=change_request_id,
            approver_id=approver_id,
            step_order=step_order
        )

        with self._lock:
            self.approval_requests[request_id] = request

        # Send notification to approver
        change_request = version_manager.db.get_change_requests(status="pending")
        matching_request = next(
            (cr for cr in change_request if cr.request_id == change_request_id),
            None
        )

        if matching_request:
            self.notification_manager.send_notification(
                "approval_required",
                {
                    'change_request_id': change_request_id,
                    'document_id': matching_request.doc_id,
                    'change_type': matching_request.request_type,
                    'requested_by': matching_request.requested_by,
                    'description': matching_request.description,
                    'priority': matching_request.priority,
                    'due_date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
                },
                recipients=[approver_id]
            )

        return request_id

    def respond_to_approval(self, request_id: str, approver_id: str,
                          decision: str, notes: str = None) -> bool:
        """Respond to an approval request"""
        request = self.approval_requests.get(request_id)
        if not request:
            return False

        # Verify the approver
        if request.approver_id != approver_id:
            return False

        # Update request
        request.status = decision
        request.responded_at = datetime.now()
        request.response_notes = notes

        # Check if this completes the change request
        change_request = version_manager.db.get_change_requests()
        matching_request = next(
            (cr for cr in change_request if cr.request_id == request.change_request_id),
            None
        )

        if matching_request:
            if decision == "approved":
                # Check if all required approvals are complete
                if self._check_all_approvals_complete(request.change_request_id):
                    matching_request.status = "approved"
                    matching_request.reviewed_at = datetime.now()
                    matching_request.review_notes = notes

                    # Implement the change
                    self._implement_approved_change(matching_request)
            elif decision == "rejected":
                matching_request.status = "rejected"
                matching_request.reviewed_at = datetime.now()
                matching_request.review_notes = notes

            version_manager.db.save_change_request(matching_request)

        return True

    def _check_all_approvals_complete(self, change_request_id: str) -> bool:
        """Check if all required approvals are complete"""
        # Get all approval requests for this change request
        relevant_requests = [
            req for req in self.approval_requests.values()
            if req.change_request_id == change_request_id and req.status != "pending"
        ]

        # For now, assume at least one approval is required
        return len(relevant_requests) > 0

    def _implement_approved_change(self, change_request: ChangeRequest):
        """Implement an approved change request"""
        try:
            if change_request.request_type == "status_change":
                # Transition document status
                success = advanced_lifecycle_manager.transition_document(
                    change_request.target_version_id,
                    "approved",  # This would be determined by the request
                    change_request.requested_by,
                    change_request.description
                )

                if success:
                    change_request.status = "implemented"
                    change_request.implemented_at = datetime.now()
                    version_manager.db.save_change_request(change_request)

            elif change_request.request_type == "rollback":
                # Perform rollback
                success = version_manager.rollback_to_version(
                    change_request.doc_id,
                    change_request.target_version_id,
                    change_request.requested_by
                )

                if success:
                    change_request.status = "implemented"
                    change_request.implemented_at = datetime.now()
                    version_manager.db.save_change_request(change_request)

        except Exception as e:
            print(f"Error implementing approved change: {e}")

    def get_pending_approvals(self, user_id: str) -> List[Dict[str, Any]]:
        """Get pending approvals for a user"""
        pending_approvals = []

        for request in self.approval_requests.values():
            if request.approver_id == user_id and request.status == "pending":
                # Get change request details
                change_requests = version_manager.db.get_change_requests()
                change_request = next(
                    (cr for cr in change_requests if cr.request_id == request.change_request_id),
                    None
                )

                if change_request:
                    pending_approvals.append({
                        'approval_request_id': request.request_id,
                        'change_request_id': request.change_request_id,
                        'document_id': change_request.doc_id,
                        'request_type': change_request.request_type,
                        'title': change_request.title,
                        'description': change_request.description,
                        'priority': change_request.priority,
                        'requested_at': change_request.created_at,
                        'step_order': request.step_order
                    })

        return pending_approvals

    def delegate_approval(self, request_id: str, delegator_id: str,
                         delegate_to_id: str, reason: str = None) -> bool:
        """Delegate an approval to another user"""
        request = self.approval_requests.get(request_id)
        if not request or request.approver_id != delegator_id:
            return False

        # Check if delegation is allowed
        change_requests = version_manager.db.get_change_requests()
        change_request = next(
            (cr for cr in change_requests if cr.request_id == request.change_request_id),
            None
        )

        if not change_request:
            return False

        # Find the approval chain to check delegation rules
        # For now, assume delegation is allowed
        request.delegated_to = delegate_to_id
        request.response_notes = f"Delegated to {delegate_to_id} by {delegator_id}. Reason: {reason}"

        # Create new approval request for delegate
        new_request = ApprovalRequest(
            request_id=f"delegated_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            change_request_id=request.change_request_id,
            approver_id=delegate_to_id,
            step_order=request.step_order,
            status="pending"
        )

        with self._lock:
            self.approval_requests[new_request.request_id] = new_request

        # Send notification to delegate
        self.notification_manager.send_notification(
            "approval_required",
            {
                'change_request_id': request.change_request_id,
                'document_id': change_request.doc_id,
                'delegated_by': delegator_id,
                'reason': reason
            },
            recipients=[delegate_to_id]
        )

        return True

class WorkflowNotificationManager:
    """Manages notifications specifically for workflow events"""

    def __init__(self):
        self.notification_manager = NotificationManager()

    def notify_version_created(self, version: DocumentVersion):
        """Notify about new version creation"""
        self.notification_manager.send_notification(
            "version_created",
            {
                'document_id': version.doc_id,
                'version_id': version.version_id,
                'version_number': version.version_number,
                'created_by': version.created_by,
                'created_at': version.created_at.isoformat(),
                'change_type': version.change_type,
                'change_description': version.change_description,
                'document_url': f"/documents/{version.doc_id}/versions/{version.version_id}"
            }
        )

    def notify_status_changed(self, version_id: str, old_status: str,
                            new_status: str, changed_by: str, reason: str = None):
        """Notify about status change"""
        version = version_manager.db.get_version(version_id)
        if not version:
            return

        self.notification_manager.send_notification(
            "status_changed",
            {
                'document_id': version.doc_id,
                'version_id': version_id,
                'version_number': version.version_number,
                'old_status': old_status,
                'new_status': new_status,
                'changed_by': changed_by,
                'changed_at': datetime.now().isoformat(),
                'reason': reason,
                'document_url': f"/documents/{version.doc_id}/versions/{version.version_id}"
            }
        )

    def notify_workflow_step_completed(self, workflow_instance_id: str,
                                     step_name: str, completed_by: str,
                                     step_status: str, notes: str = None):
        """Notify about workflow step completion"""
        # Get workflow instance details
        status = advanced_lifecycle_manager.get_workflow_status(workflow_instance_id)
        if not status:
            return

        self.notification_manager.send_notification(
            "workflow_step_completed",
            {
                'workflow_instance_id': workflow_instance_id,
                'document_id': status['instance_id'].split('_')[1],  # Extract doc_id
                'step_name': step_name,
                'completed_by': completed_by,
                'completed_at': datetime.now().isoformat(),
                'step_status': step_status,
                'notes': notes,
                'next_step': f"Step {status['current_step'] + 1}" if status['current_step'] < status['total_steps'] else "Completed",
                'progress_percent': status['progress_percent'],
                'workflow_url': f"/workflows/{workflow_instance_id}"
            }
        )

# Global instances
notification_manager = NotificationManager()
approval_workflow_manager = ApprovalWorkflowManager()
workflow_notification_manager = WorkflowNotificationManager()

# Integration with existing systems
def integrate_with_version_control():
    """Integrate notification system with version control"""

    # Hook into version creation
    original_create_version = version_manager.create_version

    def create_version_with_notification(doc_id: str, file_path: str,
                                       change_type: str = "auto",
                                       change_description: str = None,
                                       created_by: str = None):
        version = original_create_version(doc_id, file_path, change_type,
                                        change_description, created_by)

        if version:
            workflow_notification_manager.notify_version_created(version)

        return version

    version_manager.create_version = create_version_with_notification

    # Hook into lifecycle transitions
    original_transition = advanced_lifecycle_manager.transition_document

    def transition_with_notification(version_id: str, new_status: str,
                                   user_id: str, reason: str = None):
        version = version_manager.db.get_version(version_id)
        old_status = version.lifecycle_status if version else "unknown"

        success = original_transition(version_id, new_status, user_id, reason)

        if success:
            workflow_notification_manager.notify_status_changed(
                version_id, old_status, new_status, user_id, reason
            )

        return success

    advanced_lifecycle_manager.transition_document = transition_with_notification

# Initialize integration
integrate_with_version_control()