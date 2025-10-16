"""
Workflow Automation and Task Management Engine
Intelligent automation for document workflows and task management
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import uuid
from dataclasses import dataclass, field
from enum import Enum
import re

# Import collaborative workflow components
from .collaborative_workflow import (
    collaborative_manager, DocumentWorkflow, WorkflowTask,
    WorkflowStatus, CollaborationEvent, CollaborationEventType, UserRole
)
from .notification_system import notification_manager, NotificationType, NotificationPriority
from .annotation_system import annotation_manager, AnnotationType


class AutomationTrigger(Enum):
    """Types of automation triggers"""
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    MANUAL = "manual"


class AutomationAction(Enum):
    """Types of automation actions"""
    SEND_NOTIFICATION = "send_notification"
    UPDATE_WORKFLOW_STATUS = "update_workflow_status"
    CREATE_TASK = "create_task"
    ASSIGN_USER = "assign_user"
    SEND_EMAIL = "send_email"
    UPDATE_DOCUMENT = "update_document"
    RUN_SCRIPT = "run_script"


@dataclass
class WorkflowRule:
    """Represents an automation rule"""
    rule_id: str
    name: str
    description: str
    trigger: AutomationTrigger
    trigger_config: Dict[str, Any]
    actions: List[Dict[str, Any]]  # List of action configurations
    conditions: List[Dict[str, Any]] = field(default_factory=list)  # Conditions that must be met
    is_active: bool = True
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'trigger': self.trigger.value,
            'trigger_config': self.trigger_config,
            'actions': self.actions,
            'conditions': self.conditions,
            'is_active': self.is_active,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
            'trigger_count': self.trigger_count
        }


@dataclass
class ScheduledTask:
    """Represents a scheduled automation task"""
    task_id: str
    rule_id: str
    scheduled_time: datetime
    status: str  # pending, running, completed, failed
    attempts: int = 0
    max_attempts: int = 3
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'rule_id': self.rule_id,
            'scheduled_time': self.scheduled_time.isoformat(),
            'status': self.status,
            'attempts': self.attempts,
            'max_attempts': self.max_attempts,
            'error_message': self.error_message
        }


class WorkflowAutomationEngine:
    """Engine for workflow automation and task management"""

    def __init__(self):
        self.rules: Dict[str, WorkflowRule] = {}
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.running_tasks: Set[str] = set()

        # Event handlers
        self.event_handlers: Dict[CollaborationEventType, List[Callable]] = {}

        # Threading control
        self._lock = threading.Lock()
        self._running = False
        self._scheduler_thread = None

        # Load default rules
        self._load_default_rules()

        # Start automation engine
        self.start_engine()

    def start_engine(self):
        """Start the automation engine"""
        if self._running:
            return

        self._running = True

        # Start scheduler thread
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        print("Workflow automation engine started")

    def stop_engine(self):
        """Stop the automation engine"""
        self._running = False
        print("Workflow automation engine stopped")

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                self._process_scheduled_tasks()
                self._check_time_based_rules()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Error in automation scheduler: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def _process_scheduled_tasks(self):
        """Process scheduled automation tasks"""
        with self._lock:
            current_time = datetime.now()
            pending_tasks = [
                task for task in self.scheduled_tasks.values()
                if task.status == 'pending' and task.scheduled_time <= current_time
            ]

            for task in pending_tasks:
                self._execute_scheduled_task(task)

    def _check_time_based_rules(self):
        """Check and trigger time-based automation rules"""
        with self._lock:
            current_time = datetime.now()

            for rule in self.rules.values():
                if not rule.is_active:
                    continue

                if rule.trigger == AutomationTrigger.TIME_BASED:
                    self._check_time_based_rule(rule, current_time)

    def _check_time_based_rule(self, rule: WorkflowRule, current_time: datetime):
        """Check if a time-based rule should be triggered"""
        config = rule.trigger_config

        if config.get('type') == 'daily':
            # Check if it's time to run (e.g., every day at 9 AM)
            target_hour = config.get('hour', 9)
            target_minute = config.get('minute', 0)

            if (current_time.hour == target_hour and
                current_time.minute == target_minute and
                not self._was_rule_triggered_today(rule, current_time)):

                self._trigger_rule(rule, {'current_time': current_time.isoformat()})

        elif config.get('type') == 'interval':
            # Check if enough time has passed since last trigger
            interval_minutes = config.get('interval_minutes', 60)
            last_triggered = rule.last_triggered

            if (last_triggered is None or
                current_time - last_triggered >= timedelta(minutes=interval_minutes)):

                self._trigger_rule(rule, {'current_time': current_time.isoformat()})

    def _was_rule_triggered_today(self, rule: WorkflowRule, current_time: datetime) -> bool:
        """Check if rule was already triggered today"""
        if rule.last_triggered is None:
            return False

        today = current_time.date()
        last_triggered_date = rule.last_triggered.date()

        return today == last_triggered_date

    def _execute_scheduled_task(self, task: ScheduledTask):
        """Execute a scheduled task"""
        try:
            task.status = 'running'
            task.attempts += 1

            # Find the rule
            rule = self.rules.get(task.rule_id)
            if not rule or not rule.is_active:
                task.status = 'failed'
                task.error_message = "Rule not found or inactive"
                return

            # Execute the rule
            self._trigger_rule(rule, {'scheduled_task': task.task_id})

            task.status = 'completed'

        except Exception as e:
            task.error_message = str(e)

            if task.attempts >= task.max_attempts:
                task.status = 'failed'
            else:
                # Schedule retry
                retry_time = datetime.now() + timedelta(minutes=5 * task.attempts)
                task.scheduled_time = retry_time
                task.status = 'pending'

    def _trigger_rule(self, rule: WorkflowRule, context: Dict[str, Any]):
        """Trigger a rule and execute its actions"""
        # Check conditions
        if not self._check_conditions(rule.conditions, context):
            return

        # Execute actions
        for action_config in rule.actions:
            try:
                self._execute_action(action_config, context, rule.created_by)
            except Exception as e:
                print(f"Error executing action {action_config.get('type', 'unknown')}: {e}")

        # Update rule statistics
        rule.last_triggered = datetime.now()
        rule.trigger_count += 1

        print(f"Rule '{rule.name}' triggered successfully")

    def _check_conditions(self, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        """Check if rule conditions are met"""
        if not conditions:
            return True  # No conditions means always execute

        for condition in conditions:
            condition_type = condition.get('type')

            if condition_type == 'document_status':
                document_id = condition.get('document_id')
                expected_status = condition.get('status')

                if document_id:
                    # Check if document exists and has expected status
                    workflow = next(
                        (w for w in collaborative_manager.workflows.values() if w.document_id == document_id),
                        None
                    )
                    if not workflow or workflow.status.value != expected_status:
                        return False

            elif condition_type == 'time_range':
                # Check if current time is in specified range
                start_time = condition.get('start_time')  # "09:00"
                end_time = condition.get('end_time')     # "17:00"

                if start_time and end_time:
                    current_hour = datetime.now().hour
                    start_hour = int(start_time.split(':')[0])
                    end_hour = int(end_time.split(':')[0])

                    if start_hour > end_hour:  # Overnight range
                        if not (current_hour >= start_hour or current_hour < end_hour):
                            return False
                    else:  # Same day range
                        if not (start_hour <= current_hour < end_hour):
                            return False

            elif condition_type == 'user_count':
                # Check number of users/collaborators
                min_users = condition.get('min_users', 1)
                max_users = condition.get('max_users', 100)

                document_id = condition.get('document_id')
                if document_id:
                    collaborators = collaborative_manager.get_document_collaborators(document_id)
                    user_count = len(collaborators)

                    if not (min_users <= user_count <= max_users):
                        return False

            elif condition_type == 'custom_expression':
                # Evaluate custom expression (would need a safe expression evaluator)
                expression = condition.get('expression', 'True')

                # Simple variable substitution
                try:
                    # This is a simplified version - would need proper expression evaluation
                    if 'context' in context:
                        if 'user_count' in str(expression):
                            # Replace with actual user count logic
                            pass
                except:
                    return False

        return True

    def _execute_action(self, action_config: Dict[str, Any], context: Dict[str, Any], triggered_by: str):
        """Execute a single action"""
        action_type = action_config.get('type')

        if action_type == AutomationAction.SEND_NOTIFICATION.value:
            self._execute_send_notification(action_config, context, triggered_by)
        elif action_type == AutomationAction.CREATE_TASK.value:
            self._execute_create_task(action_config, context, triggered_by)
        elif action_type == AutomationAction.UPDATE_WORKFLOW_STATUS.value:
            self._execute_update_workflow_status(action_config, context, triggered_by)
        elif action_type == AutomationAction.SEND_EMAIL.value:
            self._execute_send_email(action_config, context, triggered_by)

    def _execute_send_notification(self, action_config: Dict[str, Any], context: Dict[str, Any], triggered_by: str):
        """Execute send notification action"""
        user_ids = action_config.get('user_ids', [])
        title = action_config.get('title', 'Automated Notification')
        message = action_config.get('message', 'This is an automated notification')

        # Replace variables in message
        message = self._replace_variables(message, context)
        title = self._replace_variables(title, context)

        priority = NotificationPriority[action_config.get('priority', 'NORMAL')]

        for user_id in user_ids:
            notification_manager.send_notification(
                user_id,
                NotificationType.SYSTEM_ANNOUNCEMENT,
                title,
                message,
                priority=priority,
                data=context
            )

    def _execute_create_task(self, action_config: Dict[str, Any], context: Dict[str, Any], triggered_by: str):
        """Execute create task action"""
        document_id = action_config.get('document_id')
        if not document_id:
            return

        title = action_config.get('title', 'Automated Task')
        description = action_config.get('description', 'This task was created automatically')

        # Replace variables
        title = self._replace_variables(title, context)
        description = self._replace_variables(description, context)

        assigned_to = action_config.get('assigned_to', [])
        priority = action_config.get('priority', 'medium')

        # Calculate due date if specified
        due_date = None
        if 'due_in_days' in action_config:
            due_date = datetime.now() + timedelta(days=action_config['due_in_days'])

        task = collaborative_manager.create_task(
            document_id=document_id,
            title=title,
            description=description,
            assigned_to=assigned_to,
            created_by=triggered_by,
            priority=priority,
            due_date=due_date
        )

        print(f"Created automated task: {task.task_id}")

    def _execute_update_workflow_status(self, action_config: Dict[str, Any], context: Dict[str, Any], triggered_by: str):
        """Execute update workflow status action"""
        workflow_id = action_config.get('workflow_id')
        new_status = action_config.get('status')

        if workflow_id and new_status:
            # Find and update workflow
            if workflow_id in collaborative_manager.workflows:
                workflow = collaborative_manager.workflows[workflow_id]
                old_status = workflow.status
                workflow.status = WorkflowStatus(new_status)
                workflow.updated_at = datetime.now()

                # Notify participants
                participants = list(workflow.reviewers.keys()) + list(workflow.approvers.keys())
                for participant_id in participants:
                    notification_manager.send_notification(
                        participant_id,
                        NotificationType.WORKFLOW_STATUS_CHANGED,
                        "Workflow Status Updated",
                        f"Workflow '{workflow.title}' status changed from {old_status.value} to {new_status}",
                        data={'workflow_id': workflow_id}
                    )

    def _execute_send_email(self, action_config: Dict[str, Any], context: Dict[str, Any], triggered_by: str):
        """Execute send email action"""
        # This would integrate with the email system
        # For now, just log the action
        recipients = action_config.get('recipients', [])
        subject = action_config.get('subject', 'Automated Email')
        body = action_config.get('body', 'This is an automated email')

        # Replace variables
        subject = self._replace_variables(subject, context)
        body = self._replace_variables(body, context)

        print(f"Would send email to {recipients}: {subject}")

    def _replace_variables(self, text: str, context: Dict[str, Any]) -> str:
        """Replace variables in text with context values"""
        # Simple variable replacement
        if 'current_time' in context:
            text = text.replace('{{current_time}}', context['current_time'])

        if 'document_id' in context:
            text = text.replace('{{document_id}}', context['document_id'])

        if 'workflow_id' in context:
            text = text.replace('{{workflow_id}}', context['workflow_id'])

        return text

    def add_rule(self, rule: WorkflowRule) -> str:
        """Add a new automation rule"""
        with self._lock:
            self.rules[rule.rule_id] = rule
            return rule.rule_id

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an automation rule"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]

                # Remove associated scheduled tasks
                tasks_to_remove = [
                    task_id for task_id, task in self.scheduled_tasks.items()
                    if task.rule_id == rule_id
                ]
                for task_id in tasks_to_remove:
                    del self.scheduled_tasks[task_id]

                return True
            return False

    def get_rule(self, rule_id: str) -> Optional[WorkflowRule]:
        """Get a specific rule"""
        with self._lock:
            return self.rules.get(rule_id)

    def list_rules(self, active_only: bool = False) -> List[WorkflowRule]:
        """List all automation rules"""
        with self._lock:
            rules = list(self.rules.values())

            if active_only:
                rules = [r for r in rules if r.is_active]

            # Sort by creation date (newest first)
            rules.sort(key=lambda x: x.created_at, reverse=True)

            return rules

    def schedule_rule(self, rule_id: str, delay_minutes: int = 0) -> str:
        """Schedule a rule to run after a delay"""
        with self._lock:
            if rule_id not in self.rules:
                raise ValueError(f"Rule {rule_id} not found")

            task = ScheduledTask(
                task_id=str(uuid.uuid4()),
                rule_id=rule_id,
                scheduled_time=datetime.now() + timedelta(minutes=delay_minutes)
            )

            self.scheduled_tasks[task.task_id] = task
            return task.task_id

    def trigger_rule_manually(self, rule_id: str, context: Dict[str, Any] = None) -> bool:
        """Manually trigger a rule"""
        with self._lock:
            rule = self.rules.get(rule_id)
            if not rule or not rule.is_active:
                return False

            self._trigger_rule(rule, context or {})
            return True

    def _load_default_rules(self):
        """Load default automation rules"""
        # Rule 1: Send reminder for overdue workflows
        overdue_reminder_rule = WorkflowRule(
            rule_id=str(uuid.uuid4()),
            name="Overdue Workflow Reminder",
            description="Send reminders for workflows past their due date",
            trigger=AutomationTrigger.TIME_BASED,
            trigger_config={
                'type': 'daily',
                'hour': 9,
                'minute': 0
            },
            conditions=[
                {
                    'type': 'custom_expression',
                    'expression': 'workflow_overdue'
                }
            ],
            actions=[
                {
                    'type': AutomationAction.SEND_NOTIFICATION.value,
                    'user_ids': [],  # Would be populated dynamically
                    'title': 'Overdue Workflow Reminder',
                    'message': 'Workflow {{workflow_title}} is overdue. Please take action.',
                    'priority': NotificationPriority.HIGH.value
                }
            ],
            created_by='system'
        )

        # Rule 2: Auto-approve after period of inactivity
        auto_approve_rule = WorkflowRule(
            rule_id=str(uuid.uuid4()),
            name="Auto-approve Inactive Workflows",
            description="Automatically approve workflows that have been in review for too long",
            trigger=AutomationTrigger.CONDITION_BASED,
            trigger_config={
                'check_interval_minutes': 60
            },
            conditions=[
                {
                    'type': 'document_status',
                    'status': 'in_review'
                },
                {
                    'type': 'time_range',
                    'min_days_in_status': 7
                }
            ],
            actions=[
                {
                    'type': AutomationAction.UPDATE_WORKFLOW_STATUS.value,
                    'workflow_id': '',  # Would be set dynamically
                    'status': WorkflowStatus.APPROVED.value
                }
            ],
            created_by='system'
        )

        # Rule 3: Welcome new collaborators
        welcome_rule = WorkflowRule(
            rule_id=str(uuid.uuid4()),
            name="Welcome New Collaborators",
            description="Send welcome message when users are added to documents",
            trigger=AutomationTrigger.EVENT_BASED,
            trigger_config={
                'event_type': CollaborationEventType.PERMISSION_CHANGED.value
            },
            conditions=[],
            actions=[
                {
                    'type': AutomationAction.SEND_NOTIFICATION.value,
                    'user_ids': [],  # Would be set to new collaborator
                    'title': 'Welcome to the Document',
                    'message': 'You have been added as a collaborator to document {{document_id}}. Welcome!',
                    'priority': NotificationPriority.NORMAL.value
                }
            ],
            created_by='system'
        )

        self.add_rule(overdue_reminder_rule)
        self.add_rule(auto_approve_rule)
        self.add_rule(welcome_rule)

    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get automation engine statistics"""
        with self._lock:
            total_rules = len(self.rules)
            active_rules = len([r for r in self.rules.values() if r.is_active])
            scheduled_tasks = len(self.scheduled_tasks)
            running_tasks = len(self.running_tasks)

            # Trigger statistics
            total_triggers = sum(r.trigger_count for r in self.rules.values())

            # Recent activity
            recent_rules = [
                r for r in self.rules.values()
                if r.last_triggered and r.last_triggered > datetime.now() - timedelta(hours=24)
            ]

            return {
                'total_rules': total_rules,
                'active_rules': active_rules,
                'scheduled_tasks': scheduled_tasks,
                'running_tasks': running_tasks,
                'total_triggers': total_triggers,
                'recent_activity': len(recent_rules),
                'engine_status': 'running' if self._running else 'stopped'
            }

    def export_rules(self) -> Dict[str, Any]:
        """Export automation rules for backup"""
        with self._lock:
            return {
                'rules': [r.to_dict() for r in self.rules.values()],
                'scheduled_tasks': [t.to_dict() for t in self.scheduled_tasks.values()],
                'statistics': self.get_engine_statistics(),
                'exported_at': datetime.now().isoformat()
            }


# Global workflow automation engine
automation_engine = WorkflowAutomationEngine()


class TaskManager:
    """Enhanced task management system"""

    def __init__(self):
        self.task_templates: Dict[str, Dict[str, Any]] = {}
        self.task_dependencies: Dict[str, Set[str]] = {}
        self.task_history: List[Dict[str, Any]] = []

        # Threading lock
        self._lock = threading.Lock()

        # Load default task templates
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default task templates"""
        self.task_templates = {
            'document_review': {
                'name': 'Document Review',
                'description': 'Review document for accuracy and completeness',
                'estimated_hours': 2,
                'required_skills': ['review', 'attention_to_detail'],
                'checklist': [
                    'Review content accuracy',
                    'Check formatting consistency',
                    'Verify references and citations',
                    'Ensure compliance requirements'
                ]
            },
            'final_approval': {
                'name': 'Final Approval',
                'description': 'Provide final approval for document publication',
                'estimated_hours': 0.5,
                'required_skills': ['decision_making', 'authority'],
                'checklist': [
                    'Review all feedback',
                    'Confirm compliance',
                    'Approve for publication'
                ]
            },
            'quality_check': {
                'name': 'Quality Assurance Check',
                'description': 'Perform quality assurance review',
                'estimated_hours': 1,
                'required_skills': ['qa', 'testing'],
                'checklist': [
                    'Check for errors',
                    'Verify functionality',
                    'Test edge cases'
                ]
            }
        }

    def create_task_from_template(self, template_id: str, document_id: str,
                                 assigned_to: List[str], created_by: str,
                                 customizations: Dict[str, Any] = None) -> Optional[WorkflowTask]:
        """Create a task from a template"""
        with self._lock:
            if template_id not in self.task_templates:
                return None

            template = self.task_templates[template_id]
            customizations = customizations or {}

            # Create task with template data
            title = customizations.get('title', template['name'])
            description = customizations.get('description', template['description'])

            # Calculate due date based on estimated hours
            estimated_hours = template.get('estimated_hours', 1)
            due_date = None
            if 'due_in_hours' in customizations:
                due_date = datetime.now() + timedelta(hours=customizations['due_in_hours'])
            elif estimated_hours > 0:
                # Default: 2 business days for review tasks
                business_days = max(2, estimated_hours // 4)
                due_date = self._add_business_days(datetime.now(), business_days)

            task = collaborative_manager.create_task(
                document_id=document_id,
                title=title,
                description=description,
                assigned_to=assigned_to,
                created_by=created_by,
                priority=customizations.get('priority', 'medium'),
                due_date=due_date
            )

            # Add task to history
            self.task_history.append({
                'task_id': task.task_id,
                'template_id': template_id,
                'created_at': datetime.now().isoformat(),
                'created_by': created_by
            })

            # Keep only recent history (last 1000 items)
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-1000:]

            return task

    def _add_business_days(self, start_date: datetime, business_days: int) -> datetime:
        """Add business days to a date"""
        current_date = start_date
        added_days = 0

        while added_days < business_days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:  # Monday to Friday
                added_days += 1

        return current_date

    def get_task_recommendations(self, document_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get task recommendations based on document and user"""
        recommendations = []

        # Get document information
        workflow = next(
            (w for w in collaborative_manager.workflows.values() if w.document_id == document_id),
            None
        )

        if workflow:
            # Recommend based on workflow status
            if workflow.status == WorkflowStatus.DRAFT:
                recommendations.append({
                    'template_id': 'document_review',
                    'title': 'Initial Document Review',
                    'reason': 'Document is in draft status and needs review',
                    'priority': 'high'
                })

            elif workflow.status == WorkflowStatus.IN_REVIEW:
                recommendations.append({
                    'template_id': 'final_approval',
                    'title': 'Final Approval',
                    'reason': 'Document is ready for final approval',
                    'priority': 'normal'
                })

        return recommendations

    def get_task_analytics(self) -> Dict[str, Any]:
        """Get task management analytics"""
        with self._lock:
            total_tasks = len(collaborative_manager.tasks)

            # Task status breakdown
            status_counts = {}
            for task in collaborative_manager.tasks.values():
                status = task.status
                status_counts[status] = status_counts.get(status, 0) + 1

            # Priority breakdown
            priority_counts = {}
            for task in collaborative_manager.tasks.values():
                priority = task.priority
                priority_counts[priority] = priority_counts.get(priority, 0) + 1

            # Overdue tasks
            overdue_tasks = []
            current_time = datetime.now()
            for task in collaborative_manager.tasks.values():
                if (task.due_date and
                    task.due_date < current_time and
                    task.status != 'completed'):
                    overdue_tasks.append(task.task_id)

            return {
                'total_tasks': total_tasks,
                'tasks_by_status': status_counts,
                'tasks_by_priority': priority_counts,
                'overdue_tasks': len(overdue_tasks),
                'overdue_task_ids': overdue_tasks,
                'task_templates': len(self.task_templates),
                'recent_task_history': len(self.task_history)
            }


# Global task manager
task_manager = TaskManager()