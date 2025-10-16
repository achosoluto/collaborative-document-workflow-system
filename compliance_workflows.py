"""
Compliance Violation Detection and Remediation Workflows
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

try:
    from .compliance_models import ComplianceViolation, ComplianceDataManager
    from .compliance_reporting import ComplianceReportingManager
except ImportError:
    from compliance_models import ComplianceViolation, ComplianceDataManager
    from compliance_reporting import ComplianceReportingManager

logger = logging.getLogger(__name__)


@dataclass
class RemediationTask:
    """Represents a remediation task for a violation"""

    task_id: str
    violation_id: str
    title: str
    description: str

    # Assignment
    assigned_to: Optional[str] = None
    assigned_by: str = "system"
    assigned_at: datetime = None

    # Status
    status: str = "pending"  # 'pending', 'in_progress', 'completed', 'cancelled'
    priority: str = "medium"  # 'low', 'medium', 'high', 'critical'

    # Timeline
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    due_date: Optional[datetime] = None

    # Progress
    progress_notes: List[str] = None
    completion_evidence: List[str] = None

    def __post_init__(self):
        if self.progress_notes is None:
            self.progress_notes = []
        if self.completion_evidence is None:
            self.completion_evidence = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.assigned_at is None:
            self.assigned_at = self.created_at


@dataclass
class WorkflowStep:
    """Represents a step in a remediation workflow"""

    step_id: str
    name: str
    description: str
    step_type: str  # 'notification', 'approval', 'task', 'verification'

    # Step configuration
    assigned_to: Optional[str] = None
    due_duration_hours: int = 24
    requires_approval: bool = False

    # Conditions
    conditions: Dict[str, Any] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


@dataclass
class RemediationWorkflow:
    """Defines a remediation workflow for violations"""

    workflow_id: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]

    # Workflow steps
    steps: List[WorkflowStep] = None

    # Status
    is_active: bool = True

    def __post_init__(self):
        if self.steps is None:
            self.steps = []


class ViolationWorkflowManager:
    """Manages violation detection and remediation workflows"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()
        self.reporting_manager = ComplianceReportingManager()
        self.workflows = self._load_default_workflows()
        self.remediation_tasks_file = Path("data/compliance/remediation_tasks.json")

    def _load_default_workflows(self) -> List[RemediationWorkflow]:
        """Load default remediation workflows"""
        workflows = []

        # Critical Violation Workflow
        critical_workflow = RemediationWorkflow(
            workflow_id="critical_violation_workflow",
            name="Critical Violation Remediation",
            description="Workflow for handling critical compliance violations",
            trigger_conditions={
                'violation_severity': 'critical'
            },
            steps=[
                WorkflowStep(
                    step_id="immediate_notification",
                    name="Immediate Notification",
                    description="Notify compliance officer immediately",
                    step_type="notification",
                    assigned_to="compliance_officer",
                    due_duration_hours=1
                ),
                WorkflowStep(
                    step_id="initial_assessment",
                    name="Initial Assessment",
                    description="Assess violation impact and urgency",
                    step_type="task",
                    assigned_to="compliance_officer",
                    due_duration_hours=4,
                    requires_approval=True
                ),
                WorkflowStep(
                    step_id="remediation_planning",
                    name="Remediation Planning",
                    description="Create detailed remediation plan",
                    step_type="task",
                    assigned_to="document_owner",
                    due_duration_hours=24
                ),
                WorkflowStep(
                    step_id="remediation_execution",
                    name="Execute Remediation",
                    description="Implement remediation actions",
                    step_type="task",
                    assigned_to="document_owner",
                    due_duration_hours=72
                ),
                WorkflowStep(
                    step_id="verification",
                    name="Verification",
                    description="Verify remediation effectiveness",
                    step_type="verification",
                    assigned_to="compliance_officer",
                    due_duration_hours=24,
                    requires_approval=True
                )
            ]
        )
        workflows.append(critical_workflow)

        # High Severity Violation Workflow
        high_workflow = RemediationWorkflow(
            workflow_id="high_violation_workflow",
            name="High Severity Violation Remediation",
            description="Workflow for handling high severity compliance violations",
            trigger_conditions={
                'violation_severity': 'high'
            },
            steps=[
                WorkflowStep(
                    step_id="notification",
                    name="Notification",
                    description="Notify relevant stakeholders",
                    step_type="notification",
                    assigned_to="compliance_team",
                    due_duration_hours=4
                ),
                WorkflowStep(
                    step_id="assessment",
                    name="Assessment",
                    description="Assess violation and plan remediation",
                    step_type="task",
                    assigned_to="document_owner",
                    due_duration_hours=24
                ),
                WorkflowStep(
                    step_id="remediation",
                    name="Remediation",
                    description="Implement remediation actions",
                    step_type="task",
                    assigned_to="document_owner",
                    due_duration_hours=120
                ),
                WorkflowStep(
                    step_id="review",
                    name="Review",
                    description="Review remediation results",
                    step_type="approval",
                    assigned_to="compliance_officer",
                    due_duration_hours=48
                )
            ]
        )
        workflows.append(high_workflow)

        return workflows

    def process_new_violation(self, violation: ComplianceViolation) -> List[RemediationTask]:
        """Process a new violation and create remediation tasks"""
        tasks = []

        try:
            # Find applicable workflow
            applicable_workflow = self._find_applicable_workflow(violation)
            if not applicable_workflow:
                logger.warning(f"No workflow found for violation {violation.violation_id}")
                return tasks

            # Create remediation tasks for each workflow step
            for step in applicable_workflow.steps:
                task = self._create_task_from_step(violation, step)
                if task:
                    tasks.append(task)

            # Save tasks
            self._save_remediation_tasks(tasks)

            logger.info(f"Created {len(tasks)} remediation tasks for violation {violation.violation_id}")

        except Exception as e:
            logger.error(f"Error processing violation {violation.violation_id}: {e}")

        return tasks

    def _find_applicable_workflow(self, violation: ComplianceViolation) -> Optional[RemediationWorkflow]:
        """Find the appropriate workflow for a violation"""
        for workflow in self.workflows:
            if not workflow.is_active:
                continue

            # Check trigger conditions
            if self._matches_trigger_conditions(violation, workflow.trigger_conditions):
                return workflow

        return None

    def _matches_trigger_conditions(self, violation: ComplianceViolation, conditions: Dict[str, Any]) -> bool:
        """Check if violation matches workflow trigger conditions"""
        for key, value in conditions.items():
            if key == 'violation_severity':
                if violation.severity != value:
                    return False
            elif key == 'violation_type':
                if violation.violation_type != value:
                    return False
            # Add more condition checks as needed

        return True

    def _create_task_from_step(self, violation: ComplianceViolation, step: WorkflowStep) -> Optional[RemediationTask]:
        """Create a remediation task from a workflow step"""
        try:
            # Determine assignment based on step type and violation
            assigned_to = self._determine_task_assignment(violation, step)

            # Calculate due date
            due_date = datetime.now() + timedelta(hours=step.due_duration_hours)

            # Adjust due date based on violation severity
            if violation.severity == 'critical':
                due_date = datetime.now() + timedelta(hours=step.due_duration_hours // 2)
            elif violation.severity == 'high':
                due_date = datetime.now() + timedelta(hours=step.due_duration_hours * 3 // 4)

            task = RemediationTask(
                task_id=f"task_{violation.violation_id}_{step.step_id}",
                violation_id=violation.violation_id,
                title=f"{step.name}: {violation.title}",
                description=f"{step.description}\n\nViolation Details: {violation.description}",
                assigned_to=assigned_to,
                priority=violation.severity,
                due_date=due_date
            )

            return task

        except Exception as e:
            logger.error(f"Error creating task from step {step.step_id}: {e}")
            return None

    def _determine_task_assignment(self, violation: ComplianceViolation, step: WorkflowStep) -> str:
        """Determine who should be assigned the task"""
        # Use step assignment if specified
        if step.assigned_to:
            return step.assigned_to

        # Otherwise, determine based on violation characteristics
        if violation.severity == 'critical':
            return 'compliance_officer'
        elif violation.severity == 'high':
            return 'compliance_team'
        else:
            return 'document_owner'

    def _save_remediation_tasks(self, tasks: List[RemediationTask]) -> bool:
        """Save remediation tasks to file"""
        try:
            # Load existing tasks
            existing_tasks = []
            if self.remediation_tasks_file.exists():
                with open(self.remediation_tasks_file, 'r') as f:
                    existing_tasks = json.load(f)

            # Add new tasks
            for task in tasks:
                task_dict = {
                    'task_id': task.task_id,
                    'violation_id': task.violation_id,
                    'title': task.title,
                    'description': task.description,
                    'assigned_to': task.assigned_to,
                    'assigned_by': task.assigned_by,
                    'assigned_at': task.assigned_at.isoformat(),
                    'status': task.status,
                    'priority': task.priority,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'due_date': task.due_date.isoformat() if task.due_date else None,
                    'progress_notes': task.progress_notes,
                    'completion_evidence': task.completion_evidence
                }
                existing_tasks.append(task_dict)

            # Save back to file
            with open(self.remediation_tasks_file, 'w') as f:
                json.dump(existing_tasks, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving remediation tasks: {e}")
            return False

    def get_remediation_tasks(
        self,
        violation_id: str = None,
        assigned_to: str = None,
        status: str = None
    ) -> List[RemediationTask]:
        """Get remediation tasks with optional filtering"""
        try:
            if not self.remediation_tasks_file.exists():
                return []

            with open(self.remediation_tasks_file, 'r') as f:
                tasks_data = json.load(f)

            tasks = []
            for task_data in tasks_data:
                task = RemediationTask(
                    task_id=task_data['task_id'],
                    violation_id=task_data['violation_id'],
                    title=task_data['title'],
                    description=task_data['description'],
                    assigned_to=task_data.get('assigned_to'),
                    assigned_by=task_data.get('assigned_by', 'system'),
                    assigned_at=datetime.fromisoformat(task_data['assigned_at']),
                    status=task_data.get('status', 'pending'),
                    priority=task_data.get('priority', 'medium'),
                    created_at=datetime.fromisoformat(task_data['created_at']),
                    started_at=datetime.fromisoformat(task_data['started_at']) if task_data.get('started_at') else None,
                    completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data.get('completed_at') else None,
                    due_date=datetime.fromisoformat(task_data['due_date']) if task_data.get('due_date') else None,
                    progress_notes=task_data.get('progress_notes', []),
                    completion_evidence=task_data.get('completion_evidence', [])
                )
                tasks.append(task)

            # Apply filters
            if violation_id:
                tasks = [t for t in tasks if t.violation_id == violation_id]
            if assigned_to:
                tasks = [t for t in tasks if t.assigned_to == assigned_to]
            if status:
                tasks = [t for t in tasks if t.status == status]

            return tasks

        except Exception as e:
            logger.error(f"Error loading remediation tasks: {e}")
            return []

    def update_task_status(
        self,
        task_id: str,
        status: str,
        progress_note: str = None,
        completion_evidence: str = None
    ) -> bool:
        """Update the status of a remediation task"""
        try:
            # Load current tasks
            tasks = self.get_remediation_tasks()

            # Find and update the task
            for task in tasks:
                if task.task_id == task_id:
                    task.status = status
                    task.updated_at = datetime.now()

                    if status == 'in_progress' and not task.started_at:
                        task.started_at = datetime.now()
                    elif status == 'completed':
                        task.completed_at = datetime.now()
                        if completion_evidence:
                            task.completion_evidence.append(completion_evidence)

                    if progress_note:
                        task.progress_notes.append(f"{datetime.now().isoformat()}: {progress_note}")

                    break

            # Save updated tasks
            return self._save_remediation_tasks(tasks)

        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
            return False

    def get_overdue_tasks(self) -> List[RemediationTask]:
        """Get tasks that are past their due date"""
        try:
            all_tasks = self.get_remediation_tasks()
            now = datetime.now()

            overdue_tasks = [
                task for task in all_tasks
                if (task.due_date and task.due_date < now and task.status not in ['completed', 'cancelled'])
            ]

            return overdue_tasks

        except Exception as e:
            logger.error(f"Error getting overdue tasks: {e}")
            return []

    def escalate_overdue_tasks(self) -> int:
        """Escalate overdue tasks and return count of escalated tasks"""
        escalated_count = 0

        try:
            overdue_tasks = self.get_overdue_tasks()

            for task in overdue_tasks:
                # Escalate by updating priority and adding escalation note
                if task.priority in ['low', 'medium']:
                    task.priority = 'high'
                    task.progress_notes.append(
                        f"{datetime.now().isoformat()}: Task escalated due to overdue status"
                    )

                    # Update violation status if needed
                    self._update_violation_escalation(task.violation_id)

                    escalated_count += 1

            # Save updated tasks
            if escalated_count > 0:
                all_tasks = self.get_remediation_tasks()
                self._save_remediation_tasks(all_tasks)

            return escalated_count

        except Exception as e:
            logger.error(f"Error escalating overdue tasks: {e}")
            return 0

    def _update_violation_escalation(self, violation_id: str) -> bool:
        """Update violation status when tasks are escalated"""
        try:
            violations = self.data_manager.load_violations()
            violation = next((v for v in violations if v.violation_id == violation_id), None)

            if violation:
                violation.status = 'escalated'
                violation.updated_at = datetime.now()

                # Save updated violation
                # This would use the data manager's save functionality
                return True

        except Exception as e:
            logger.error(f"Error updating violation escalation: {e}")

        return False

    def get_workflow_status(self, violation_id: str) -> Dict[str, Any]:
        """Get the status of all workflow tasks for a violation"""
        try:
            tasks = self.get_remediation_tasks(violation_id=violation_id)

            if not tasks:
                return {'status': 'no_tasks', 'message': 'No remediation tasks found for this violation'}

            # Calculate overall status
            completed_tasks = len([t for t in tasks if t.status == 'completed'])
            in_progress_tasks = len([t for t in tasks if t.status == 'in_progress'])
            pending_tasks = len([t for t in tasks if t.status == 'pending'])
            overdue_tasks = len([t for t in tasks if t.due_date and t.due_date < datetime.now()])

            overall_status = 'completed' if completed_tasks == len(tasks) else 'in_progress'

            return {
                'violation_id': violation_id,
                'total_tasks': len(tasks),
                'completed_tasks': completed_tasks,
                'in_progress_tasks': in_progress_tasks,
                'pending_tasks': pending_tasks,
                'overdue_tasks': overdue_tasks,
                'overall_status': overall_status,
                'tasks': [
                    {
                        'task_id': t.task_id,
                        'title': t.title,
                        'status': t.status,
                        'assigned_to': t.assigned_to,
                        'due_date': t.due_date.isoformat() if t.due_date else None,
                        'priority': t.priority
                    }
                    for t in tasks
                ]
            }

        except Exception as e:
            logger.error(f"Error getting workflow status for {violation_id}: {e}")
            return {'status': 'error', 'message': str(e)}


class ComplianceViolationManager:
    """Main manager for violation detection and remediation"""

    def __init__(self):
        self.data_manager = ComplianceDataManager()
        self.workflow_manager = ViolationWorkflowManager()

    def detect_violations(self, assessment_id: str) -> List[ComplianceViolation]:
        """Detect violations from a compliance assessment"""
        violations = []

        try:
            # Load the assessment
            assessments = self.data_manager.load_assessments()
            assessment = next((a for a in assessments if a.assessment_id == assessment_id), None)

            if not assessment:
                logger.error(f"Assessment {assessment_id} not found")
                return violations

            # Extract violations from assessment results
            for violation_data in assessment.violations:
                violation = ComplianceViolation(
                    violation_id=f"violation_{assessment_id}_{len(violations) + 1}",
                    assessment_id=assessment_id,
                    doc_id=assessment.doc_id,
                    requirement_id=assessment.requirement_id,
                    violation_type=violation_data.get('type', 'unknown'),
                    severity=violation_data.get('severity', 'medium'),
                    title=violation_data.get('title', 'Compliance Violation'),
                    description=violation_data.get('description', ''),
                    section=violation_data.get('section'),
                    content_snippet=violation_data.get('content_snippet')
                )

                # Save violation
                self.data_manager.save_violation(violation)
                violations.append(violation)

                # Process violation through workflow
                self.workflow_manager.process_new_violation(violation)

            logger.info(f"Detected {len(violations)} violations for assessment {assessment_id}")

        except Exception as e:
            logger.error(f"Error detecting violations for assessment {assessment_id}: {e}")

        return violations

    def get_violations_requiring_attention(self) -> Dict[str, List[ComplianceViolation]]:
        """Get violations that require immediate attention"""
        try:
            violations = self.data_manager.load_violations(status='open')

            # Group by priority
            attention_required = {
                'critical': [],
                'high': [],
                'overdue': []
            }

            now = datetime.now()

            for violation in violations:
                if violation.severity == 'critical':
                    attention_required['critical'].append(violation)
                elif violation.severity == 'high':
                    attention_required['high'].append(violation)

                # Check for overdue remediation
                if (violation.remediation_deadline and
                    violation.remediation_deadline < now and
                    violation.status != 'resolved'):
                    attention_required['overdue'].append(violation)

            return attention_required

        except Exception as e:
            logger.error(f"Error getting violations requiring attention: {e}")
            return {'critical': [], 'high': [], 'overdue': []}

    def resolve_violation(
        self,
        violation_id: str,
        resolved_by: str,
        resolution_notes: str = None
    ) -> bool:
        """Mark a violation as resolved"""
        try:
            violations = self.data_manager.load_violations()
            violation = next((v for v in violations if v.violation_id == violation_id), None)

            if not violation:
                logger.error(f"Violation {violation_id} not found")
                return False

            # Update violation status
            violation.status = 'resolved'
            violation.resolved_at = datetime.now()
            violation.resolved_by = resolved_by
            violation.updated_at = datetime.now()

            if resolution_notes:
                if not hasattr(violation, 'resolution_notes'):
                    violation.resolution_notes = []
                violation.resolution_notes.append(f"{datetime.now().isoformat()}: {resolution_notes}")

            # Update related tasks
            tasks = self.workflow_manager.get_remediation_tasks(violation_id=violation_id)
            for task in tasks:
                if task.status not in ['completed', 'cancelled']:
                    self.workflow_manager.update_task_status(
                        task.task_id,
                        'completed',
                        f'Completed due to violation resolution: {resolution_notes}'
                    )

            # Save updated violation
            # This would use the data manager's save functionality
            logger.info(f"Violation {violation_id} resolved by {resolved_by}")

            return True

        except Exception as e:
            logger.error(f"Error resolving violation {violation_id}: {e}")
            return False

    def get_remediation_dashboard_data(self) -> Dict[str, Any]:
        """Get data for remediation dashboard"""
        try:
            # Get task statistics
            all_tasks = self.workflow_manager.get_remediation_tasks()
            overdue_tasks = self.workflow_manager.get_overdue_tasks()

            # Task status breakdown
            status_counts = {}
            for task in all_tasks:
                status_counts[task.status] = status_counts.get(task.status, 0) + 1

            # Priority breakdown
            priority_counts = {}
            for task in all_tasks:
                priority_counts[task.priority] = priority_counts.get(task.priority, 0) + 1

            # Get violations requiring attention
            attention_violations = self.get_violations_requiring_attention()

            return {
                'total_tasks': len(all_tasks),
                'overdue_tasks': len(overdue_tasks),
                'completed_tasks': status_counts.get('completed', 0),
                'in_progress_tasks': status_counts.get('in_progress', 0),
                'pending_tasks': status_counts.get('pending', 0),
                'task_status_breakdown': status_counts,
                'task_priority_breakdown': priority_counts,
                'attention_required': attention_violations,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting remediation dashboard data: {e}")
            return {'error': str(e)}