"""
Version Rollback Management System
Provides safe rollback capabilities with conflict detection and resolution
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
import shutil
import json
import threading
from pathlib import Path

from .version_control import version_manager, DocumentVersion
from .version_comparison import version_detection_engine
from .approval_workflow import workflow_notification_manager

@dataclass
class RollbackPlan:
    """Plan for rolling back a document to a previous version"""
    plan_id: str
    doc_id: str
    target_version_id: str
    current_version_id: str
    rollback_type: str  # full, selective, emergency

    # Analysis results
    affected_documents: List[str] = field(default_factory=list)
    risk_level: str = "low"
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)

    # Plan metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = None
    status: str = "planned"  # planned, approved, executing, completed, failed

@dataclass
class RollbackConflict:
    """Represents a conflict that may arise during rollback"""
    conflict_id: str
    conflict_type: str  # dependency, reference, data_loss, compatibility
    severity: str  # low, medium, high, critical
    description: str
    affected_components: List[str]
    resolution_options: List[str]
    recommended_action: str

@dataclass
class RollbackResult:
    """Result of a rollback operation"""
    success: bool
    rollback_plan_id: str
    new_version_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    executed_at: datetime = field(default_factory=datetime.now)

class RollbackManager:
    """Manages document rollback operations"""

    def __init__(self):
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        self.rollback_results: Dict[str, RollbackResult] = {}
        self._lock = threading.Lock()

        # Backup directory for rollback safety
        self.backup_dir = Path(__file__).parent / "data" / "rollbacks"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_rollback_plan(self, doc_id: str, target_version_id: str,
                           rollback_type: str = "full", created_by: str = None) -> Optional[str]:
        """Create a rollback plan for a document"""

        # Get target and current versions
        target_version = version_manager.db.get_version(target_version_id)
        if not target_version:
            return None

        current_versions = version_manager.db.get_document_versions(doc_id)
        current_version = current_versions[0] if current_versions else None

        if not current_version:
            return None

        # Create plan ID
        plan_id = f"rollback_{doc_id}_{target_version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Analyze rollback impact
        impact_analysis = version_manager.analyze_impact(doc_id, target_version_id)

        # Detect conflicts
        conflicts = self._detect_rollback_conflicts(doc_id, target_version, current_version)

        # Determine risk level
        risk_level = self._assess_rollback_risk(conflicts, impact_analysis)

        # Generate required actions
        required_actions = self._generate_rollback_actions(conflicts, impact_analysis)

        # Create rollback plan
        plan = RollbackPlan(
            plan_id=plan_id,
            doc_id=doc_id,
            target_version_id=target_version_id,
            current_version_id=current_version.version_id,
            rollback_type=rollback_type,
            affected_documents=impact_analysis.affected_documents,
            risk_level=risk_level,
            conflicts=conflicts,
            required_actions=required_actions,
            created_by=created_by
        )

        with self._lock:
            self.rollback_plans[plan_id] = plan

        return plan_id

    def execute_rollback(self, plan_id: str, executed_by: str,
                        force: bool = False) -> str:
        """Execute a rollback plan"""
        plan = self.rollback_plans.get(plan_id)
        if not plan:
            return "Rollback plan not found"

        # Check if plan is approved (if required)
        if not force and plan.risk_level in ["high", "critical"]:
            if plan.status != "approved":
                return "High-risk rollback requires approval"

        # Update plan status
        plan.status = "executing"

        try:
            # Create backup of current state
            backup_result = self._create_rollback_backup(plan)
            if not backup_result['success']:
                plan.status = "failed"
                return f"Backup failed: {backup_result['error']}"

            # Execute the rollback
            result = self._perform_rollback(plan, executed_by)

            # Update plan status
            plan.status = "completed" if result['success'] else "failed"

            # Store result
            rollback_result = RollbackResult(
                success=result['success'],
                rollback_plan_id=plan_id,
                new_version_id=result.get('new_version_id'),
                errors=result.get('errors', []),
                warnings=result.get('warnings', [])
            )

            with self._lock:
                self.rollback_results[plan_id] = rollback_result

            # Send notifications
            self._notify_rollback_completion(plan, rollback_result, executed_by)

            return "success" if result['success'] else f"failed: {result['errors']}"

        except Exception as e:
            plan.status = "failed"
            return f"Exception during rollback: {str(e)}"

    def _detect_rollback_conflicts(self, doc_id: str, target_version: DocumentVersion,
                                 current_version: DocumentVersion) -> List[Dict[str, Any]]:
        """Detect potential conflicts in rollback"""
        conflicts = []

        # Check for data loss
        if target_version.file_size < current_version.file_size * 0.8:  # Significant size reduction
            conflicts.append({
                'conflict_type': 'data_loss',
                'severity': 'high',
                'description': 'Target version is significantly smaller than current version',
                'affected_components': ['content'],
                'resolution_options': ['Accept data loss', 'Merge changes', 'Cancel rollback'],
                'recommended_action': 'Review content differences before proceeding'
            })

        # Check for dependency conflicts
        relationships = version_manager.db.get_document_relationships(doc_id)
        for rel in relationships:
            if rel.relationship_type == "dependency":
                conflicts.append({
                    'conflict_type': 'dependency',
                    'severity': 'medium',
                    'description': f'Document has dependency relationship with {rel.target_doc_id}',
                    'affected_components': [rel.target_doc_id],
                    'resolution_options': ['Update dependent documents', 'Review dependencies'],
                    'recommended_action': 'Check dependent documents for compatibility'
                })

        # Check for version timeline conflicts
        versions = version_manager.db.get_document_versions(doc_id)
        target_index = next(i for i, v in enumerate(versions) if v.version_id == target_version.version_id)
        current_index = next(i for i, v in enumerate(versions) if v.version_id == current_version.version_id)

        if target_index > current_index:
            conflicts.append({
                'conflict_type': 'timeline',
                'severity': 'low',
                'description': 'Rolling back to a newer version in timeline',
                'affected_components': ['version_history'],
                'resolution_options': ['Proceed with rollback', 'Cancel rollback'],
                'recommended_action': 'Verify this is intended'
            })

        return conflicts

    def _assess_rollback_risk(self, conflicts: List[Dict[str, Any]],
                            impact_analysis) -> str:
        """Assess overall risk level of rollback"""
        max_conflict_severity = 0
        max_impact_risk = 0

        # Assess conflict severity
        severity_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        for conflict in conflicts:
            severity = conflict.get('severity', 'low')
            max_conflict_severity = max(max_conflict_severity, severity_scores.get(severity, 1))

        # Assess impact risk
        impact_risk_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        max_impact_risk = impact_risk_scores.get(impact_analysis.risk_level, 1)

        # Overall risk assessment
        overall_risk = max(max_conflict_severity, max_impact_risk)
        if overall_risk >= 4:
            return "critical"
        elif overall_risk >= 3:
            return "high"
        elif overall_risk >= 2:
            return "medium"
        else:
            return "low"

    def _generate_rollback_actions(self, conflicts: List[Dict[str, Any]],
                                 impact_analysis) -> List[str]:
        """Generate required actions for rollback"""
        actions = []

        # Add actions based on conflicts
        for conflict in conflicts:
            if conflict['severity'] in ['high', 'critical']:
                actions.append(f"Resolve {conflict['conflict_type']} conflict: {conflict['description']}")

        # Add actions based on impact analysis
        if impact_analysis.risk_level in ['high', 'critical']:
            actions.append("Notify stakeholders of high-risk rollback")
            actions.append("Create backup before proceeding")

        actions.extend(impact_analysis.recommendations)

        return actions

    def _create_rollback_backup(self, plan: RollbackPlan) -> Dict[str, Any]:
        """Create backup before rollback"""
        try:
            backup_id = f"backup_{plan.plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_dir / backup_id
            backup_path.mkdir(exist_ok=True)

            # Backup current document file
            current_version = version_manager.db.get_version(plan.current_version_id)
            if current_version and os.path.exists(current_version.file_path):
                backup_file = backup_path / f"current_{Path(current_version.file_path).name}"
                shutil.copy2(current_version.file_path, backup_file)

            # Backup version database records
            backup_db = backup_path / "version_data.json"
            with open(backup_db, 'w') as f:
                # Save current version and related data
                backup_data = {
                    'current_version': current_version.to_dict() if current_version else None,
                    'rollback_plan': plan.__dict__,
                    'backup_timestamp': datetime.now().isoformat()
                }
                json.dump(backup_data, f, indent=2, default=str)

            return {
                'success': True,
                'backup_id': backup_id,
                'backup_path': str(backup_path)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _perform_rollback(self, plan: RollbackPlan, executed_by: str) -> Dict[str, Any]:
        """Perform the actual rollback operation"""
        errors = []
        warnings = []

        try:
            # Get target version
            target_version = version_manager.db.get_version(plan.target_version_id)
            if not target_version:
                return {
                    'success': False,
                    'errors': ['Target version not found']
                }

            # Create new version based on target
            new_version = version_manager.create_version(
                doc_id=plan.doc_id,
                file_path=target_version.file_path,
                change_type="rollback",
                change_description=f"Rollback to version {target_version.version_number}",
                created_by=executed_by
            )

            if not new_version:
                return {
                    'success': False,
                    'errors': ['Failed to create rollback version']
                }

            # Update version metadata
            new_version.parent_version_id = plan.current_version_id
            version_manager.db.save_version(new_version)

            # Check for data loss or significant changes
            if target_version.file_size < plan.current_version_id and os.path.exists(target_version.file_path):
                current_size = os.path.getsize(target_version.file_path)
                size_reduction = (1 - current_size / target_version.file_size) * 100
                if size_reduction > 20:  # More than 20% size reduction
                    warnings.append(f"Document size reduced by {size_reduction".1f"}% during rollback")

            return {
                'success': True,
                'new_version_id': new_version.version_id,
                'warnings': warnings
            }

        except Exception as e:
            errors.append(f"Rollback failed: {str(e)}")
            return {
                'success': False,
                'errors': errors
            }

    def _notify_rollback_completion(self, plan: RollbackPlan,
                                  result: RollbackResult, executed_by: str):
        """Send notifications about rollback completion"""
        notification_data = {
            'plan_id': plan.plan_id,
            'doc_id': plan.doc_id,
            'target_version_id': plan.target_version_id,
            'rollback_type': plan.rollback_type,
            'executed_by': executed_by,
            'success': result.success,
            'risk_level': plan.risk_level
        }

        if result.success:
            notification_data.update({
                'new_version_id': result.new_version_id,
                'message': f"Document {plan.doc_id} successfully rolled back to previous version"
            })
        else:
            notification_data.update({
                'errors': result.errors,
                'message': f"Rollback failed for document {plan.doc_id}"
            })

        # Send notifications to stakeholders
        workflow_notification_manager.notification_manager.send_notification(
            "rollback_completed",
            notification_data,
            recipients=["document_owner", "stakeholders"]
        )

    def get_rollback_preview(self, doc_id: str, target_version_id: str) -> Dict[str, Any]:
        """Get preview of what would happen during rollback"""
        # Create temporary plan for analysis
        plan_id = self.create_rollback_plan(doc_id, target_version_id, "full", "preview_user")

        if not plan_id:
            return {'error': 'Could not create rollback plan'}

        plan = self.rollback_plans.get(plan_id)
        if not plan:
            return {'error': 'Rollback plan not found'}

        # Get version comparison
        target_version = version_manager.db.get_version(target_version_id)
        current_versions = version_manager.db.get_document_versions(doc_id)
        current_version = current_versions[0] if current_versions else None

        comparison = None
        if target_version and current_version:
            comparison = version_manager.compare_versions(target_version_id, current_version.version_id)

        return {
            'plan_id': plan_id,
            'doc_id': doc_id,
            'target_version': target_version.to_dict() if target_version else None,
            'current_version': current_version.to_dict() if current_version else None,
            'risk_level': plan.risk_level,
            'conflicts': plan.conflicts,
            'affected_documents': plan.affected_documents,
            'required_actions': plan.required_actions,
            'comparison': comparison.to_dict() if comparison else None
        }

    def get_rollback_history(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get rollback history for a document"""
        history = []

        # Find all rollback results for this document
        for result in self.rollback_results.values():
            plan = self.rollback_plans.get(result.rollback_plan_id)
            if plan and plan.doc_id == doc_id:
                history.append({
                    'plan_id': result.rollback_plan_id,
                    'executed_at': result.executed_at,
                    'success': result.success,
                    'target_version_id': plan.target_version_id,
                    'rollback_type': plan.rollback_type,
                    'risk_level': plan.risk_level,
                    'errors': result.errors,
                    'warnings': result.warnings
                })

        # Sort by execution time (newest first)
        history.sort(key=lambda x: x['executed_at'], reverse=True)

        return history

class EmergencyRollbackManager:
    """Handles emergency rollback situations"""

    def __init__(self):
        self.rollback_manager = RollbackManager()

    def emergency_rollback(self, doc_id: str, target_version_id: str,
                         reason: str, executed_by: str) -> Dict[str, Any]:
        """Perform emergency rollback with minimal checks"""
        try:
            # Create emergency rollback plan
            plan_id = self.rollback_manager.create_rollback_plan(
                doc_id, target_version_id, "emergency", executed_by
            )

            if not plan_id:
                return {
                    'success': False,
                    'error': 'Could not create emergency rollback plan'
                }

            # Execute immediately with force flag
            result_message = self.rollback_manager.execute_rollback(plan_id, executed_by, force=True)

            success = result_message == "success"

            return {
                'success': success,
                'plan_id': plan_id,
                'result': result_message,
                'emergency_reason': reason
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Emergency rollback failed: {str(e)}'
            }

    def get_emergency_rollback_options(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get available emergency rollback options for a document"""
        versions = version_manager.db.get_document_versions(doc_id)

        options = []
        for version in versions[:10]:  # Last 10 versions
            days_old = (datetime.now() - version.created_at).days

            options.append({
                'version_id': version.version_id,
                'version_number': version.version_number,
                'created_at': version.created_at,
                'created_by': version.created_by,
                'days_old': days_old,
                'change_type': version.change_type,
                'is_emergency_candidate': days_old <= 7  # Recent versions are better for emergency rollback
            })

        return options

class RollbackScheduler:
    """Schedules automated rollback validations and cleanup"""

    def __init__(self):
        self.rollback_manager = RollbackManager()
        self._lock = threading.Lock()

    def schedule_rollback_validation(self, plan_id: str, validate_after_hours: int = 24):
        """Schedule validation of a rollback after execution"""
        def validate_rollback():
            time.sleep(validate_after_hours * 3600)  # Convert to seconds

            plan = self.rollback_manager.rollback_plans.get(plan_id)
            if not plan and plan.status == "completed":
                # Perform validation checks
                self._validate_rollback_success(plan)

        threading.Thread(target=validate_rollback, daemon=True).start()

    def _validate_rollback_success(self, plan: RollbackPlan):
        """Validate that rollback was successful"""
        try:
            # Check if new version exists and is accessible
            new_version = version_manager.db.get_version(plan.target_version_id)
            if not new_version:
                print(f"WARNING: Rollback validation failed for plan {plan.plan_id} - target version not found")
                return

            # Check if file exists and is readable
            if not os.path.exists(new_version.file_path):
                print(f"WARNING: Rollback validation failed for plan {plan.plan_id} - file not found")
                return

            # Check file integrity
            current_hash = version_manager._calculate_file_hash(new_version.file_path)
            if current_hash != new_version.file_hash:
                print(f"WARNING: Rollback validation failed for plan {plan.plan_id} - file integrity check failed")

            print(f"Rollback validation successful for plan {plan.plan_id}")

        except Exception as e:
            print(f"Error during rollback validation for plan {plan.plan_id}: {e}")

    def cleanup_old_backups(self, retention_days: int = 30):
        """Clean up old rollback backups"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            for backup_dir in self.rollback_manager.backup_dir.iterdir():
                if backup_dir.is_dir():
                    # Check backup creation date from metadata
                    metadata_file = backup_dir / "version_data.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                backup_date = datetime.fromisoformat(metadata['backup_timestamp'])

                                if backup_date < cutoff_date:
                                    # Remove old backup
                                    shutil.rmtree(backup_dir)
                                    print(f"Removed old backup: {backup_dir.name}")
                        except:
                            # If metadata is corrupted, remove directory
                            shutil.rmtree(backup_dir)

        except Exception as e:
            print(f"Error during backup cleanup: {e}")

# Global instances
rollback_manager = RollbackManager()
emergency_rollback_manager = EmergencyRollbackManager()
rollback_scheduler = RollbackScheduler()

# Schedule periodic cleanup
def schedule_backup_cleanup():
    """Schedule periodic cleanup of old backups"""
    import schedule

    def cleanup_task():
        rollback_scheduler.cleanup_old_backups()

    # Run cleanup weekly
    schedule.every().monday.at("02:00").do(cleanup_task)

    def run_schedule():
        while True:
            schedule.run_pending()
            time.sleep(3600)

    threading.Thread(target=run_schedule, daemon=True).start()

# Initialize cleanup scheduler
schedule_backup_cleanup()