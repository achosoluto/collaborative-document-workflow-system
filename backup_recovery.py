"""
Automated Backup and Recovery System
Provides comprehensive backup and recovery capabilities for the version control system
"""

import os
import json
import shutil
import hashlib
import threading
import schedule
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import tarfile
import gzip
import logging

from .version_control import version_manager
from .config import BASE_DIR

class BackupManager:
    """Manages automated backups of the version control system"""

    def __init__(self):
        self.backup_dir = BASE_DIR / "data" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup configuration
        self.backup_interval_hours = 24
        self.retention_days = 30
        self.max_backups_per_day = 7

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Start backup scheduler
        self._start_backup_scheduler()

    def _start_backup_scheduler(self):
        """Start the automated backup scheduler"""
        def backup_task():
            self.create_full_backup()
            self.cleanup_old_backups()

        # Schedule daily backups at 2 AM
        schedule.every().day.at("02:00").do(backup_task)

        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour

        threading.Thread(target=run_scheduler, daemon=True).start()
        self.logger.info("Backup scheduler started")

    def create_full_backup(self, backup_name: str = None) -> str:
        """Create a full backup of the version control system"""
        if backup_name is None:
            backup_name = f"full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)

        try:
            self.logger.info(f"Creating full backup: {backup_name}")

            # Backup version control database
            db_backup_path = backup_path / "version_control.db"
            shutil.copy2(version_manager.db.db_path, db_backup_path)

            # Backup document catalog
            catalog_path = BASE_DIR / "document_catalog.json"
            if catalog_path.exists():
                catalog_backup_path = backup_path / "document_catalog.json"
                shutil.copy2(catalog_path, catalog_backup_path)

            # Backup vector store if it exists
            vector_store_path = BASE_DIR / "data" / "vector_store"
            if vector_store_path.exists():
                vector_backup_path = backup_path / "vector_store"
                shutil.copytree(vector_store_path, vector_backup_path, dirs_exist_ok=True)

            # Create backup metadata
            metadata = {
                'backup_name': backup_name,
                'backup_type': 'full',
                'created_at': datetime.now().isoformat(),
                'version_control_db_size': os.path.getsize(db_backup_path),
                'catalog_size': os.path.getsize(catalog_backup_path) if catalog_path.exists() else 0,
                'vector_store_size': self._get_directory_size(vector_store_path) if vector_store_path.exists() else 0,
                'total_size': self._get_directory_size(backup_path)
            }

            metadata_path = backup_path / "backup_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Create compressed archive
            archive_path = self._create_backup_archive(backup_path, backup_name)

            self.logger.info(f"Full backup completed: {backup_name}")
            return str(archive_path)

        except Exception as e:
            self.logger.error(f"Error creating full backup: {e}")
            # Clean up failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise

    def create_incremental_backup(self, since_backup: str = None) -> str:
        """Create an incremental backup since the last backup"""
        backup_name = f"incremental_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)

        try:
            self.logger.info(f"Creating incremental backup: {backup_name}")

            # Find the most recent backup
            if since_backup is None:
                since_backup = self._get_latest_backup_name()

            if since_backup:
                # Copy files that have changed since the last backup
                self._copy_changed_files(since_backup, backup_path)
            else:
                # No previous backup, create full backup
                return self.create_full_backup(backup_name)

            # Create backup metadata
            metadata = {
                'backup_name': backup_name,
                'backup_type': 'incremental',
                'based_on': since_backup,
                'created_at': datetime.now().isoformat(),
                'total_size': self._get_directory_size(backup_path)
            }

            metadata_path = backup_path / "backup_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Create compressed archive
            archive_path = self._create_backup_archive(backup_path, backup_name)

            self.logger.info(f"Incremental backup completed: {backup_name}")
            return str(archive_path)

        except Exception as e:
            self.logger.error(f"Error creating incremental backup: {e}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise

    def _copy_changed_files(self, since_backup: str, target_path: Path):
        """Copy files that have changed since the specified backup"""
        source_backup = self.backup_dir / since_backup

        if not source_backup.exists():
            raise ValueError(f"Source backup not found: {since_backup}")

        # For simplicity, copy all files (in a real implementation, you'd check modification times)
        for file_path in source_backup.rglob('*'):
            if file_path.is_file() and file_path.name != "backup_metadata.json":
                relative_path = file_path.relative_to(source_backup)
                target_file = target_path / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target_file)

    def _create_backup_archive(self, backup_path: Path, backup_name: str) -> Path:
        """Create a compressed archive of the backup"""
        archive_path = self.backup_dir / f"{backup_name}.tar.gz"

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_path, arcname=backup_path.name)

        # Remove the uncompressed backup directory
        shutil.rmtree(backup_path)

        return archive_path

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of a directory"""
        total_size = 0
        if path.exists():
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size

    def _get_latest_backup_name(self) -> Optional[str]:
        """Get the name of the most recent backup"""
        try:
            backup_archives = list(self.backup_dir.glob("*.tar.gz"))
            if not backup_archives:
                return None

            # Sort by modification time (newest first)
            backup_archives.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_backup = backup_archives[0]

            # Extract backup name from archive name
            return latest_backup.stem

        except Exception as e:
            self.logger.error(f"Error getting latest backup: {e}")
            return None

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []

        try:
            for archive_path in self.backup_dir.glob("*.tar.gz"):
                try:
                    # Extract metadata from archive
                    with tarfile.open(archive_path, "r:gz") as tar:
                        # Look for metadata file in archive
                        metadata_member = None
                        for member in tar.getmembers():
                            if member.name.endswith("backup_metadata.json"):
                                metadata_member = member
                                break

                        if metadata_member:
                            metadata_file = tar.extractfile(metadata_member)
                            metadata = json.load(metadata_file)

                            backups.append({
                                'backup_name': metadata.get('backup_name'),
                                'backup_type': metadata.get('backup_type'),
                                'created_at': metadata.get('created_at'),
                                'size_bytes': archive_path.stat().st_size,
                                'size_human': self._format_size(archive_path.stat().st_size),
                                'archive_path': str(archive_path)
                            })

                except Exception as e:
                    self.logger.error(f"Error reading backup {archive_path}: {e}")

        except Exception as e:
            self.logger.error(f"Error listing backups: {e}")

        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return backups

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes".1f"} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes".1f"} TB"

    def cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)

            for archive_path in self.backup_dir.glob("*.tar.gz"):
                # Check if backup is too old
                if archive_path.stat().st_mtime < cutoff_date.timestamp():
                    self.logger.info(f"Removing old backup: {archive_path.name}")
                    archive_path.unlink()

        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {e}")

class RecoveryManager:
    """Manages recovery operations from backups"""

    def __init__(self):
        self.backup_dir = BASE_DIR / "data" / "backups"
        self.recovery_dir = BASE_DIR / "data" / "recovery"
        self.recovery_dir.mkdir(parents=True, exist_ok=True)

    def recover_from_backup(self, backup_name: str, recovery_type: str = "full") -> Dict[str, Any]:
        """Recover system from a backup"""
        try:
            archive_path = self.backup_dir / f"{backup_name}.tar.gz"

            if not archive_path.exists():
                return {
                    'success': False,
                    'error': f'Backup not found: {backup_name}'
                }

            self.logger.info(f"Starting recovery from backup: {backup_name}")

            # Extract backup to recovery directory
            extraction_path = self.recovery_dir / backup_name
            if extraction_path.exists():
                shutil.rmtree(extraction_path)
            extraction_path.mkdir()

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extraction_path)

            # Validate backup integrity
            validation_result = self._validate_backup_integrity(extraction_path)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f'Backup validation failed: {validation_result["errors"]}'
                }

            # Perform recovery based on type
            if recovery_type == "full":
                recovery_result = self._perform_full_recovery(extraction_path)
            elif recovery_type == "database_only":
                recovery_result = self._perform_database_recovery(extraction_path)
            else:
                return {
                    'success': False,
                    'error': f'Unknown recovery type: {recovery_type}'
                }

            if recovery_result['success']:
                self.logger.info(f"Recovery completed successfully from backup: {backup_name}")

                # Create post-recovery validation point
                self._create_recovery_validation_point(extraction_path)

            return recovery_result

        except Exception as e:
            self.logger.error(f"Error during recovery: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _validate_backup_integrity(self, extraction_path: Path) -> Dict[str, Any]:
        """Validate the integrity of an extracted backup"""
        errors = []
        warnings = []

        try:
            # Check for required files
            required_files = [
                "version_control.db",
                "backup_metadata.json"
            ]

            for required_file in required_files:
                file_path = extraction_path / required_file
                if not file_path.exists():
                    errors.append(f"Required file missing: {required_file}")

            # Validate database integrity
            db_path = extraction_path / "version_control.db"
            if db_path.exists():
                try:
                    with sqlite3.connect(db_path) as conn:
                        # Run basic integrity check
                        conn.execute("PRAGMA integrity_check")
                        conn.execute("PRAGMA foreign_key_check")
                except sqlite3.Error as e:
                    errors.append(f"Database integrity check failed: {e}")

            # Check file sizes are reasonable
            for file_path in extraction_path.rglob('*'):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    if size == 0:
                        warnings.append(f"Empty file: {file_path.name}")
                    elif size > 100 * 1024 * 1024:  # 100MB
                        warnings.append(f"Large file (>100MB): {file_path.name}")

            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }

        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': []
            }

    def _perform_full_recovery(self, extraction_path: Path) -> Dict[str, Any]:
        """Perform full system recovery"""
        try:
            # Create backup of current state
            current_backup = self._create_pre_recovery_backup()

            # Stop version control operations during recovery
            # (In a real implementation, you'd have a maintenance mode)

            # Restore version control database
            source_db = extraction_path / "version_control.db"
            target_db = version_manager.db.db_path

            # Backup current database
            if target_db.exists():
                backup_db = target_db.parent / f"version_control_pre_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                shutil.copy2(target_db, backup_db)

            # Restore database
            shutil.copy2(source_db, target_db)

            # Restore document catalog
            source_catalog = extraction_path / "document_catalog.json"
            target_catalog = BASE_DIR / "document_catalog.json"

            if source_catalog.exists():
                if target_catalog.exists():
                    backup_catalog = BASE_DIR / f"document_catalog_pre_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    shutil.copy2(target_catalog, backup_catalog)
                shutil.copy2(source_catalog, target_catalog)

            # Restore vector store
            source_vector = extraction_path / "vector_store"
            target_vector = BASE_DIR / "data" / "vector_store"

            if source_vector.exists():
                if target_vector.exists():
                    backup_vector = BASE_DIR / "data" / f"vector_store_pre_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.move(target_vector, backup_vector)
                shutil.copytree(source_vector, target_vector)

            return {
                'success': True,
                'recovery_type': 'full',
                'pre_recovery_backup': current_backup,
                'message': 'Full recovery completed successfully'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Full recovery failed: {str(e)}'
            }

    def _perform_database_recovery(self, extraction_path: Path) -> Dict[str, Any]:
        """Perform database-only recovery"""
        try:
            source_db = extraction_path / "version_control.db"
            target_db = version_manager.db.db_path

            # Backup current database
            if target_db.exists():
                backup_db = target_db.parent / f"version_control_pre_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                shutil.copy2(target_db, backup_db)

            # Restore database
            shutil.copy2(source_db, target_db)

            return {
                'success': True,
                'recovery_type': 'database_only',
                'message': 'Database recovery completed successfully'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Database recovery failed: {str(e)}'
            }

    def _create_pre_recovery_backup(self) -> str:
        """Create a backup of current state before recovery"""
        backup_name = f"pre_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_name

        try:
            # Create minimal backup of current critical files
            backup_path.mkdir(exist_ok=True)

            # Backup current database
            if version_manager.db.db_path.exists():
                shutil.copy2(version_manager.db.db_path, backup_path / "version_control.db")

            # Backup current catalog
            catalog_path = BASE_DIR / "document_catalog.json"
            if catalog_path.exists():
                shutil.copy2(catalog_path, backup_path / "document_catalog.json")

            return backup_name

        except Exception as e:
            self.logger.error(f"Error creating pre-recovery backup: {e}")
            return None

    def _create_recovery_validation_point(self, recovery_path: Path):
        """Create a validation point after successful recovery"""
        validation_name = f"post_recovery_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        validation_path = self.recovery_dir / validation_name
        validation_path.mkdir(exist_ok=True)

        try:
            # Copy key files for validation
            source_db = version_manager.db.db_path
            if source_db.exists():
                shutil.copy2(source_db, validation_path / "version_control_validated.db")

            # Create validation metadata
            metadata = {
                'validation_name': validation_name,
                'created_at': datetime.now().isoformat(),
                'based_on_backup': recovery_path.name,
                'system_status': 'post_recovery'
            }

            with open(validation_path / "validation_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Created recovery validation point: {validation_name}")

        except Exception as e:
            self.logger.error(f"Error creating recovery validation point: {e}")

class IntegrityChecker:
    """Checks integrity of version control data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check_system_integrity(self) -> Dict[str, Any]:
        """Check overall system integrity"""
        issues = []
        warnings = []

        try:
            # Check database integrity
            db_issues = self._check_database_integrity()
            issues.extend(db_issues)

            # Check file integrity
            file_issues = self._check_file_integrity()
            issues.extend(file_issues)

            # Check relationship integrity
            rel_issues = self._check_relationship_integrity()
            issues.extend(rel_issues)

            # Check for orphaned data
            orphan_issues = self._check_for_orphaned_data()
            warnings.extend(orphan_issues)

            return {
                'is_healthy': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'checked_at': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'is_healthy': False,
                'issues': [f'Integrity check failed: {str(e)}'],
                'warnings': [],
                'checked_at': datetime.now().isoformat()
            }

    def _check_database_integrity(self) -> List[str]:
        """Check database integrity"""
        issues = []

        try:
            with sqlite3.connect(version_manager.db.db_path) as conn:
                # Run SQLite integrity check
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_results = cursor.fetchall()

                for result in integrity_results:
                    if result[0] != 'ok':
                        issues.append(f"Database integrity issue: {result[0]}")

                # Check foreign key constraints
                cursor = conn.execute("PRAGMA foreign_key_check")
                fk_results = cursor.fetchall()

                for result in fk_results:
                    issues.append(f"Foreign key violation in table {result[0]}, row {result[1]}")

        except Exception as e:
            issues.append(f"Database integrity check failed: {str(e)}")

        return issues

    def _check_file_integrity(self) -> List[str]:
        """Check file integrity"""
        issues = []

        try:
            # Check version files exist
            versions = []
            with sqlite3.connect(version_manager.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM document_versions")
                versions = cursor.fetchall()

            for version_row in versions:
                file_path = version_row['file_path']
                if not os.path.exists(file_path):
                    issues.append(f"Missing version file: {file_path}")
                else:
                    # Check file hash if available
                    stored_hash = version_row['file_hash']
                    if stored_hash:
                        current_hash = self._calculate_file_hash(file_path)
                        if current_hash != stored_hash:
                            issues.append(f"File hash mismatch for {file_path}")

        except Exception as e:
            issues.append(f"File integrity check failed: {str(e)}")

        return issues

    def _check_relationship_integrity(self) -> List[str]:
        """Check relationship integrity"""
        issues = []

        try:
            relationships = []
            with sqlite3.connect(version_manager.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM document_relationships")
                relationships = cursor.fetchall()

            for rel_row in relationships:
                source_id = rel_row['source_doc_id']
                target_id = rel_row['target_doc_id']

                # Check if referenced documents exist
                for doc_id in [source_id, target_id]:
                    versions = version_manager.db.get_document_versions(doc_id)
                    if not versions:
                        issues.append(f"Relationship references non-existent document: {doc_id}")

        except Exception as e:
            issues.append(f"Relationship integrity check failed: {str(e)}")

        return issues

    def _check_for_orphaned_data(self) -> List[str]:
        """Check for orphaned data"""
        warnings = []

        try:
            # Check for versions without corresponding documents in catalog
            catalog_path = BASE_DIR / "document_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    catalog = json.load(f)

                catalog_doc_ids = set(catalog.keys())

                # Get all document IDs from version control
                vc_doc_ids = set()
                with sqlite3.connect(version_manager.db.db_path) as conn:
                    cursor = conn.execute("SELECT DISTINCT doc_id FROM document_versions")
                    vc_doc_ids = {row[0] for row in cursor.fetchall()}

                orphaned_in_vc = vc_doc_ids - catalog_doc_ids
                if orphaned_in_vc:
                    warnings.append(f"Found {len(orphaned_in_vc)} documents in version control but not in catalog")

        except Exception as e:
            warnings.append(f"Orphaned data check failed: {str(e)}")

        return warnings

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

class DisasterRecoveryManager:
    """Manages disaster recovery scenarios"""

    def __init__(self):
        self.backup_manager = BackupManager()
        self.recovery_manager = RecoveryManager()
        self.integrity_checker = IntegrityChecker()

    def create_disaster_recovery_plan(self) -> Dict[str, Any]:
        """Create a comprehensive disaster recovery plan"""
        plan = {
            'plan_id': f"dr_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'created_at': datetime.now().isoformat(),
            'recovery_objectives': {
                'rto_hours': 4,  # Recovery Time Objective
                'rpo_hours': 1   # Recovery Point Objective
            },
            'backup_strategy': {
                'full_backup_interval_hours': 24,
                'incremental_backup_interval_hours': 6,
                'retention_days': 30
            },
            'recovery_procedures': [
                'Assess damage and determine recovery scope',
                'Restore from most recent backup',
                'Validate system integrity',
                'Restore user access',
                'Notify stakeholders'
            ],
            'critical_data': [
                'Version control database',
                'Document catalog',
                'Vector store',
                'User permissions'
            ],
            'contact_information': {
                'technical_lead': 'admin@example.com',
                'backup_admin': 'backup@example.com'
            }
        }

        # Save plan
        plan_path = BASE_DIR / "data" / "disaster_recovery_plan.json"
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2)

        return plan

    def execute_disaster_recovery(self, backup_name: str = None) -> Dict[str, Any]:
        """Execute disaster recovery procedure"""
        try:
            # Determine which backup to use
            if backup_name is None:
                backups = self.backup_manager.list_backups()
                if not backups:
                    return {
                        'success': False,
                        'error': 'No backups available for recovery'
                    }
                backup_name = backups[0]['backup_name']  # Use most recent

            # Execute recovery
            recovery_result = self.recovery_manager.recover_from_backup(backup_name, "full")

            if recovery_result['success']:
                # Run integrity check after recovery
                integrity_result = self.integrity_checker.check_system_integrity()

                return {
                    'success': True,
                    'recovery_result': recovery_result,
                    'integrity_check': integrity_result,
                    'message': 'Disaster recovery completed successfully'
                }
            else:
                return recovery_result

        except Exception as e:
            return {
                'success': False,
                'error': f'Disaster recovery failed: {str(e)}'
            }

# Global instances
backup_manager = BackupManager()
recovery_manager = RecoveryManager()
integrity_checker = IntegrityChecker()
disaster_recovery_manager = DisasterRecoveryManager()

# Schedule periodic integrity checks
def schedule_integrity_checks():
    """Schedule periodic integrity checks"""
    def integrity_check_task():
        result = integrity_checker.check_system_integrity()
        if not result['is_healthy']:
            print(f"WARNING: System integrity issues detected: {result['issues']}")

    # Run integrity check daily at 3 AM
    schedule.every().day.at("03:00").do(integrity_check_task)

    def run_schedule():
        while True:
            schedule.run_pending()
            time.sleep(3600)

    threading.Thread(target=run_schedule, daemon=True).start()

# Initialize integrity check scheduler
schedule_integrity_checks()