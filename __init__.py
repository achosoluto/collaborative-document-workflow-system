"""
Version Control System Package
Comprehensive document versioning and tracking system
"""

from .version_control import version_manager, DocumentVersion
from .document_monitor import document_monitor, lifecycle_manager, notification_system
from .version_comparison import document_comparator, version_detection_engine
from .lifecycle_manager import advanced_lifecycle_manager
from .approval_workflow import approval_workflow_manager, workflow_notification_manager
from .rollback_manager import rollback_manager, emergency_rollback_manager
from .relationship_analyzer import impact_analyzer, relationship_analyzer
from .integration import integration_manager
from .backup_recovery import backup_manager, recovery_manager, integrity_checker
from .audit_logger import audit_logger, audit_event_manager

__version__ = "1.0.0"
__all__ = [
    'version_manager',
    'document_monitor',
    'lifecycle_manager',
    'notification_system',
    'document_comparator',
    'version_detection_engine',
    'advanced_lifecycle_manager',
    'approval_workflow_manager',
    'workflow_notification_manager',
    'rollback_manager',
    'emergency_rollback_manager',
    'impact_analyzer',
    'relationship_analyzer',
    'integration_manager',
    'backup_manager',
    'recovery_manager',
    'integrity_checker',
    'audit_logger',
    'audit_event_manager'
]

def initialize_version_control_system():
    """Initialize the complete version control system"""
    print("Initializing Version Control System...")

    # Start document monitoring
    document_monitor.start_monitoring()

    # Initialize integrations
    integration_manager.initialize_integration()

    print("Version Control System initialized successfully!")
    print(f"Version: {__version__}")

def get_system_status():
    """Get comprehensive system status"""
    return {
        'version_control': {
            'status': 'active',
            'total_versions': 'N/A',  # Would query database
            'total_documents': 'N/A'   # Would query database
        },
        'document_monitoring': {
            'status': 'active' if document_monitor.is_running else 'inactive',
            'tracked_documents': len(document_monitor.get_tracked_documents())
        },
        'lifecycle_management': {
            'status': 'active',
            'active_workflows': len([w for w in advanced_lifecycle_manager.workflow_instances.values() if w.status == 'active'])
        },
        'backup_system': {
            'status': 'active',
            'total_backups': len(backup_manager.list_backups())
        },
        'audit_system': {
            'status': 'active',
            'total_events': 'N/A'  # Would query database
        }
    }