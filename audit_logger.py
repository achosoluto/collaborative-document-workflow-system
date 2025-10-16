"""
Audit Logging System for Version Control Operations
Provides comprehensive logging of all version control activities
"""

import json
import threading
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .version_control import version_manager, DocumentVersion
from .config import BASE_DIR

@dataclass
class AuditEvent:
    """Represents an audit event"""
    event_id: str
    timestamp: datetime
    user_id: str
    action: str  # create_version, update_lifecycle, rollback, etc.
    resource_type: str  # document, version, workflow, etc.
    resource_id: str
    details: Dict[str, Any] = field(default_factory=dict)

    # Context information
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None

    # Result information
    success: bool = True
    error_message: Optional[str] = None
    changes_made: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'session_id': self.session_id,
            'success': self.success,
            'error_message': self.error_message,
            'changes_made': self.changes_made
        }

class AuditLogger:
    """Main audit logging system"""

    def __init__(self):
        self.db_path = BASE_DIR / "data" / "audit_log.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory queue for high-performance logging
        self.event_queue = []
        self._lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)

        # Initialize database
        self._init_database()

        # Start event processor
        self._start_event_processor()

    def _init_database(self):
        """Initialize audit log database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    session_id TEXT,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    changes_made TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_events(action)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_events(resource_type, resource_id)")

    def _start_event_processor(self):
        """Start the event processing thread"""
        def process_events():
            while True:
                events_to_process = []

                with self._lock:
                    # Get events from queue (up to 100 at a time)
                    if self.event_queue:
                        events_to_process = self.event_queue[:100]
                        self.event_queue = self.event_queue[100:]

                # Process events in batch
                if events_to_process:
                    self._save_events_to_database(events_to_process)

                time.sleep(1)  # Process every second

        threading.Thread(target=process_events, daemon=True).start()

    def log_event(self, user_id: str, action: str, resource_type: str, resource_id: str,
                  details: Dict[str, Any] = None, **kwargs):
        """Log an audit event"""
        event = AuditEvent(
            event_id=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}",
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            **kwargs
        )

        # Add to queue for asynchronous processing
        with self._lock:
            self.event_queue.append(event)

        # Also log to standard logging
        self.logger.info(f"AUDIT: {user_id} {action} {resource_type} {resource_id}")

    def _save_events_to_database(self, events: List[AuditEvent]):
        """Save events to database in batch"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for event in events:
                    conn.execute("""
                        INSERT OR REPLACE INTO audit_events
                        (event_id, timestamp, user_id, action, resource_type, resource_id,
                         details, ip_address, user_agent, session_id, success, error_message, changes_made)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id, event.timestamp.isoformat(), event.user_id,
                        event.action, event.resource_type, event.resource_id,
                        json.dumps(event.details), event.ip_address, event.user_agent,
                        event.session_id, event.success, event.error_message,
                        json.dumps(event.changes_made)
                    ))

        except Exception as e:
            self.logger.error(f"Error saving audit events to database: {e}")

    def get_audit_trail(self, resource_type: str = None, resource_id: str = None,
                       user_id: str = None, action: str = None,
                       start_date: str = None, end_date: str = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit trail with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []

                if resource_type:
                    query += " AND resource_type = ?"
                    params.append(resource_type)

                if resource_id:
                    query += " AND resource_id = ?"
                    params.append(resource_id)

                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)

                if action:
                    query += " AND action = ?"
                    params.append(action)

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)

                events = []
                for row in cursor:
                    event_dict = dict(row)
                    # Parse JSON fields
                    for json_field in ['details', 'changes_made']:
                        if event_dict.get(json_field):
                            try:
                                event_dict[json_field] = json.loads(event_dict[json_field])
                            except:
                                pass
                    events.append(event_dict)

                return events

        except Exception as e:
            self.logger.error(f"Error getting audit trail: {e}")
            return []

    def get_user_activity_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get activity summary for a user"""
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

            events = self.get_audit_trail(user_id=user_id, start_date=start_date, limit=1000)

            # Summarize activities
            action_counts = {}
            resource_types = set()
            success_rate = 0

            for event in events:
                action = event['action']
                action_counts[action] = action_counts.get(action, 0) + 1
                resource_types.add(event['resource_type'])

                if event['success']:
                    success_rate += 1

            success_rate = (success_rate / len(events) * 100) if events else 0

            return {
                'user_id': user_id,
                'period_days': days,
                'total_actions': len(events),
                'action_counts': action_counts,
                'resource_types': list(resource_types),
                'success_rate': success_rate,
                'most_recent_action': events[0]['timestamp'] if events else None
            }

        except Exception as e:
            self.logger.error(f"Error getting user activity summary: {e}")
            return {}

    def get_resource_history(self, resource_type: str, resource_id: str) -> List[Dict[str, Any]]:
        """Get complete history for a specific resource"""
        return self.get_audit_trail(
            resource_type=resource_type,
            resource_id=resource_id,
            limit=500
        )

    def get_compliance_report(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Generate compliance report for audit activities"""
        try:
            events = self.get_audit_trail(
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )

            # Analyze events for compliance
            compliance_data = {
                'total_events': len(events),
                'unique_users': len(set(e['user_id'] for e in events)),
                'action_breakdown': {},
                'resource_breakdown': {},
                'security_events': [],
                'failed_operations': [],
                'high_risk_actions': []
            }

            high_risk_actions = {
                'delete_document', 'rollback', 'force_rollback',
                'archive_document', 'change_approval', 'system_config'
            }

            for event in events:
                action = event['action']
                resource_type = event['resource_type']

                # Count actions
                compliance_data['action_breakdown'][action] = compliance_data['action_breakdown'].get(action, 0) + 1

                # Count resource types
                compliance_data['resource_breakdown'][resource_type] = compliance_data['resource_breakdown'].get(resource_type, 0) + 1

                # Track security events
                if action in ['login', 'logout', 'permission_change']:
                    compliance_data['security_events'].append(event)

                # Track failed operations
                if not event['success']:
                    compliance_data['failed_operations'].append(event)

                # Track high-risk actions
                if action in high_risk_actions:
                    compliance_data['high_risk_actions'].append(event)

            return compliance_data

        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return {}

class AuditEventManager:
    """Manages audit events across all version control operations"""

    def __init__(self):
        self.audit_logger = AuditLogger()

    def log_version_created(self, version: DocumentVersion, user_id: str, **context):
        """Log version creation event"""
        self.audit_logger.log_event(
            user_id=user_id,
            action='create_version',
            resource_type='version',
            resource_id=version.version_id,
            details={
                'doc_id': version.doc_id,
                'version_number': version.version_number,
                'change_type': version.change_type,
                'file_path': version.file_path,
                'file_size': version.file_size
            },
            changes_made={
                'new_version': version.version_number,
                'file_hash': version.file_hash
            },
            **context
        )

    def log_version_modified(self, version_id: str, user_id: str,
                           changes: Dict[str, Any], **context):
        """Log version modification event"""
        self.audit_logger.log_event(
            user_id=user_id,
            action='modify_version',
            resource_type='version',
            resource_id=version_id,
            details={'changes': changes},
            changes_made=changes,
            **context
        )

    def log_lifecycle_transition(self, version_id: str, user_id: str,
                               old_status: str, new_status: str, reason: str = None, **context):
        """Log lifecycle status transition"""
        self.audit_logger.log_event(
            user_id=user_id,
            action='lifecycle_transition',
            resource_type='version',
            resource_id=version_id,
            details={
                'old_status': old_status,
                'new_status': new_status,
                'reason': reason
            },
            changes_made={
                'lifecycle_status': {
                    'from': old_status,
                    'to': new_status
                }
            },
            **context
        )

    def log_approval_action(self, request_id: str, user_id: str,
                          action: str, notes: str = None, **context):
        """Log approval workflow action"""
        self.audit_logger.log_event(
            user_id=user_id,
            action=f'approval_{action}',
            resource_type='approval_request',
            resource_id=request_id,
            details={
                'approval_action': action,
                'notes': notes
            },
            changes_made={
                'approval_status': action
            },
            **context
        )

    def log_rollback_operation(self, plan_id: str, user_id: str,
                             success: bool, error_message: str = None, **context):
        """Log rollback operation"""
        self.audit_logger.log_event(
            user_id=user_id,
            action='execute_rollback',
            resource_type='rollback_plan',
            resource_id=plan_id,
            details={
                'success': success,
                'error_message': error_message
            },
            success=success,
            error_message=error_message,
            **context
        )

    def log_relationship_change(self, source_doc_id: str, target_doc_id: str,
                              user_id: str, relationship_type: str, action: str, **context):
        """Log document relationship changes"""
        self.audit_logger.log_event(
            user_id=user_id,
            action=f'relationship_{action}',
            resource_type='document_relationship',
            resource_id=f"{source_doc_id}_{target_doc_id}",
            details={
                'source_doc_id': source_doc_id,
                'target_doc_id': target_doc_id,
                'relationship_type': relationship_type
            },
            changes_made={
                'relationship_action': action,
                'relationship_type': relationship_type
            },
            **context
        )

    def log_impact_analysis(self, assessment_id: str, user_id: str,
                          analysis_type: str, **context):
        """Log impact analysis execution"""
        self.audit_logger.log_event(
            user_id=user_id,
            action='run_impact_analysis',
            resource_type='impact_assessment',
            resource_id=assessment_id,
            details={
                'analysis_type': analysis_type
            },
            **context
        )

    def log_backup_operation(self, backup_name: str, user_id: str,
                           operation: str, success: bool, **context):
        """Log backup/recovery operations"""
        self.audit_logger.log_event(
            user_id=user_id,
            action=f'backup_{operation}',
            resource_type='backup',
            resource_id=backup_name,
            details={'operation': operation},
            success=success,
            **context
        )

    def log_search_operation(self, user_id: str, query: str,
                           result_count: int, **context):
        """Log search operations"""
        self.audit_logger.log_event(
            user_id=user_id,
            action='search',
            resource_type='search_query',
            resource_id=f"search_{id(self)}",
            details={
                'query': query,
                'result_count': result_count
            },
            **context
        )

class AuditReportGenerator:
    """Generates various audit reports"""

    def __init__(self):
        self.audit_logger = AuditLogger()

    def generate_daily_report(self, date: str = None) -> Dict[str, Any]:
        """Generate daily audit report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        start_date = f"{date}T00:00:00"
        end_date = f"{date}T23:59:59"

        events = self.audit_logger.get_audit_trail(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )

        # Generate report
        report = {
            'report_date': date,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_events': len(events),
                'unique_users': len(set(e['user_id'] for e in events)),
                'successful_operations': len([e for e in events if e['success']]),
                'failed_operations': len([e for e in events if not e['success']])
            },
            'top_actions': self._get_top_actions(events),
            'top_users': self._get_top_users(events),
            'security_events': [e for e in events if e['action'] in ['login', 'logout', 'permission_change']],
            'system_changes': [e for e in events if e['resource_type'] in ['system', 'configuration']]
        }

        return report

    def _get_top_actions(self, events: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequent actions"""
        action_counts = {}
        for event in events:
            action = event['action']
            action_counts[action] = action_counts.get(action, 0) + 1

        # Sort by count
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {'action': action, 'count': count}
            for action, count in sorted_actions[:limit]
        ]

    def _get_top_users(self, events: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Get most active users"""
        user_counts = {}
        for event in events:
            user = event['user_id']
            user_counts[user] = user_counts.get(user, 0) + 1

        # Sort by count
        sorted_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {'user_id': user, 'event_count': count}
            for user, count in sorted_users[:limit]
        ]

    def generate_user_activity_report(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate user activity report"""
        summary = self.audit_logger.get_user_activity_summary(user_id, days)

        # Get detailed events for the user
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        events = self.audit_logger.get_audit_trail(
            user_id=user_id,
            start_date=start_date,
            limit=1000
        )

        # Analyze user behavior
        behavior_analysis = self._analyze_user_behavior(events)

        return {
            'user_id': user_id,
            'report_period_days': days,
            'summary': summary,
            'behavior_analysis': behavior_analysis,
            'recent_events': events[:50]  # Last 50 events
        }

    def _analyze_user_behavior(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        if not events:
            return {}

        # Time analysis
        timestamps = [datetime.fromisoformat(e['timestamp']) for e in events]
        hours = [ts.hour for ts in timestamps]

        # Most active hours
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        most_active_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        # Action patterns
        actions_by_hour = {}
        for event in events:
            hour = datetime.fromisoformat(event['timestamp']).hour
            action = event['action']
            if hour not in actions_by_hour:
                actions_by_hour[hour] = {}
            actions_by_hour[hour][action] = actions_by_hour[hour].get(action, 0) + 1

        return {
            'most_active_hours': most_active_hours,
            'action_patterns_by_hour': actions_by_hour,
            'total_unique_actions': len(set(e['action'] for e in events)),
            'average_events_per_day': len(events) / 30
        }

# Global audit event manager
audit_event_manager = AuditEventManager()

# Integration with version control operations
def integrate_audit_logging():
    """Integrate audit logging with version control operations"""

    # Hook into version creation
    original_create_version = version_manager.create_version

    def create_version_with_audit(doc_id: str, file_path: str, change_type: str = "auto",
                                change_description: str = None, created_by: str = None):
        version = original_create_version(doc_id, file_path, change_type, change_description, created_by)

        if version:
            audit_event_manager.log_version_created(version, created_by or "system")

        return version

    version_manager.create_version = create_version_with_audit

    # Hook into lifecycle transitions
    original_transition = advanced_lifecycle_manager.transition_document

    def transition_with_audit(version_id: str, new_status: str, user_id: str, reason: str = None):
        version = version_manager.db.get_version(version_id)
        old_status = version.lifecycle_status if version else "unknown"

        success = original_transition(version_id, new_status, user_id, reason)

        if success:
            audit_event_manager.log_lifecycle_transition(
                version_id, user_id, old_status, new_status, reason
            )

        return success

    advanced_lifecycle_manager.transition_document = transition_with_audit

    # Hook into rollback operations
    original_execute_rollback = rollback_manager.execute_rollback

    def execute_rollback_with_audit(plan_id: str, executed_by: str, force: bool = False):
        # Get plan details for logging
        plan = rollback_manager.rollback_plans.get(plan_id)

        result = original_execute_rollback(plan_id, executed_by, force)

        success = result == "success"

        audit_event_manager.log_rollback_operation(
            plan_id, executed_by, success,
            None if success else result
        )

        return result

    rollback_manager.execute_rollback = execute_rollback_with_audit

# Initialize audit logging integration
integrate_audit_logging()

# Global instances
audit_logger = AuditLogger()
audit_report_generator = AuditReportGenerator()