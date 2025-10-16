"""
Document Monitoring and Change Detection System
Automatically detects document changes and creates new versions
"""

import os
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import threading
import schedule
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
import logging

from .version_control import version_manager, DocumentVersion
from .config import DocumentMetadata

class DocumentChangeDetector:
    """Detects changes in documents and triggers version creation"""

    def __init__(self):
        self.document_hashes: Dict[str, str] = {}
        self.document_mtimes: Dict[str, float] = {}
        self.document_sizes: Dict[str, int] = {}
        self._lock = threading.Lock()

        # Load existing document catalog
        self._load_document_catalog()

    def _load_document_catalog(self):
        """Load document catalog to initialize tracking"""
        try:
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    import json
                    catalog = json.load(f)

                for doc_id, doc_data in catalog.items():
                    file_path = doc_data.get('absolute_path')
                    if file_path and os.path.exists(file_path):
                        self._update_document_info(file_path)
        except Exception as e:
            print(f"Error loading document catalog: {e}")

    def _update_document_info(self, file_path: str):
        """Update stored information about a document"""
        try:
            stat = os.stat(file_path)
            with self._lock:
                self.document_mtimes[file_path] = stat.st_mtime
                self.document_sizes[file_path] = stat.st_size

                # Calculate hash for text files
                if self._is_text_file(file_path):
                    file_hash = self._calculate_file_hash(file_path)
                    self.document_hashes[file_path] = file_hash
        except Exception as e:
            print(f"Error updating document info for {file_path}: {e}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        except:
            return ""
        return hash_sha256.hexdigest()

    def _is_text_file(self, file_path: str) -> bool:
        """Check if file is a text file that can be hashed"""
        text_extensions = {'.txt', '.md', '.html', '.htm', '.xml', '.json', '.csv', '.py', '.js', '.css'}
        return Path(file_path).suffix.lower() in text_extensions

    def check_for_changes(self) -> List[Dict[str, str]]:
        """Check all tracked documents for changes"""
        changed_documents = []

        with self._lock:
            files_to_check = list(self.document_mtimes.keys())

        for file_path in files_to_check:
            if os.path.exists(file_path):
                change_info = self._check_document_change(file_path)
                if change_info:
                    changed_documents.append(change_info)
            else:
                # File was deleted
                changed_documents.append({
                    'file_path': file_path,
                    'change_type': 'deleted',
                    'doc_id': self._get_doc_id_from_path(file_path)
                })

        return changed_documents

    def _check_document_change(self, file_path: str) -> Optional[Dict[str, str]]:
        """Check if a specific document has changed"""
        try:
            stat = os.stat(file_path)
            current_mtime = stat.st_mtime
            current_size = stat.st_size

            with self._lock:
                prev_mtime = self.document_mtimes.get(file_path, 0)
                prev_size = self.document_sizes.get(file_path, 0)

            # Check for changes
            if current_mtime != prev_mtime or current_size != prev_size:
                # Update stored info
                self._update_document_info(file_path)

                # Determine change type
                change_type = "modified"
                if prev_mtime == 0:
                    change_type = "created"

                return {
                    'file_path': file_path,
                    'change_type': change_type,
                    'doc_id': self._get_doc_id_from_path(file_path)
                }

        except Exception as e:
            print(f"Error checking document {file_path}: {e}")

        return None

    def _get_doc_id_from_path(self, file_path: str) -> Optional[str]:
        """Get document ID from file path by checking catalog"""
        try:
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    import json
                    catalog = json.load(f)

                for doc_id, doc_data in catalog.items():
                    if doc_data.get('absolute_path') == file_path:
                        return doc_id
        except:
            pass
        return None

    def add_document_to_tracking(self, file_path: str, doc_id: str):
        """Add a document to change tracking"""
        self._update_document_info(file_path)

    def remove_document_from_tracking(self, file_path: str):
        """Remove a document from change tracking"""
        with self._lock:
            self.document_hashes.pop(file_path, None)
            self.document_mtimes.pop(file_path, None)
            self.document_sizes.pop(file_path, None)

class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events for document monitoring"""

    def __init__(self, change_detector: DocumentChangeDetector):
        self.change_detector = change_detector
        self.recent_events: Set[str] = set()
        self._lock = threading.Lock()

    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            with self._lock:
                # Debounce rapid events
                if event.src_path not in self.recent_events:
                    self.recent_events.add(event.src_path)
                    # Remove from recent events after 1 second
                    threading.Timer(1.0, self._clear_recent_event, [event.src_path]).start()

                    # Queue change check
                    threading.Thread(
                        target=self._handle_document_change,
                        args=(event.src_path, "modified"),
                        daemon=True
                    ).start()

    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            threading.Thread(
                target=self._handle_document_change,
                args=(event.src_path, "created"),
                daemon=True
            ).start()

    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory:
            threading.Thread(
                target=self._handle_document_change,
                args=(event.src_path, "deleted"),
                daemon=True
            ).start()

    def _handle_document_change(self, file_path: str, change_type: str):
        """Handle a document change event"""
        try:
            # Get document ID
            doc_id = self.change_detector._get_doc_id_from_path(file_path)
            if not doc_id:
                return

            # Create new version
            version = version_manager.create_version(
                doc_id=doc_id,
                file_path=file_path,
                change_type="auto",
                change_description=f"Automatic version creation due to {change_type}"
            )

            if version:
                print(f"Created new version {version.version_number} for document {doc_id}")

                # Update document catalog with new version info
                self._update_document_catalog(doc_id, version)

        except Exception as e:
            print(f"Error handling document change for {file_path}: {e}")

    def _update_document_catalog(self, doc_id: str, version: DocumentVersion):
        """Update document catalog with new version information"""
        try:
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    import json
                    catalog = json.load(f)

                if doc_id in catalog:
                    # Update version information in catalog
                    catalog[doc_id]['current_version'] = version.version_number
                    catalog[doc_id]['last_version_update'] = version.created_at.isoformat()
                    catalog[doc_id]['version_count'] = len(version_manager.db.get_document_versions(doc_id))

                    # Save updated catalog
                    with open(catalog_path, 'w') as f:
                        json.dump(catalog, f, indent=2)

        except Exception as e:
            print(f"Error updating document catalog: {e}")

    def _clear_recent_event(self, file_path: str):
        """Clear a file path from recent events"""
        with self._lock:
            self.recent_events.discard(file_path)

class DocumentMonitor:
    """Main document monitoring system"""

    def __init__(self, watch_paths: List[str] = None):
        self.watch_paths = watch_paths or self._get_default_watch_paths()
        self.change_detector = DocumentChangeDetector()
        self.observer = Observer()
        self.event_handler = FileChangeHandler(self.change_detector)
        self.is_running = False
        self._lock = threading.Lock()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_default_watch_paths(self) -> List[str]:
        """Get default paths to watch for document changes"""
        default_paths = [
            str(Path(__file__).parent.parent / "DTP docs"),
            str(Path(__file__).parent.parent / "Excel_to_PDF")
        ]

        # Filter to existing paths
        return [path for path in default_paths if os.path.exists(path)]

    def start_monitoring(self):
        """Start monitoring document changes"""
        with self._lock:
            if self.is_running:
                return

            # Start file system observer
            for path in self.watch_paths:
                if os.path.exists(path):
                    self.observer.schedule(self.event_handler, path, recursive=True)
                    self.logger.info(f"Monitoring path: {path}")

            self.observer.start()
            self.is_running = True
            self.logger.info("Document monitoring started")

            # Start periodic change detection
            self._start_periodic_checks()

    def stop_monitoring(self):
        """Stop monitoring document changes"""
        with self._lock:
            if not self.is_running:
                return

            self.observer.stop()
            self.observer.join()
            self.is_running = False
            self.logger.info("Document monitoring stopped")

    def _start_periodic_checks(self):
        """Start periodic checks for changes (backup to file system events)"""
        def periodic_check():
            if self.is_running:
                try:
                    changed_docs = self.change_detector.check_for_changes()
                    for change in changed_docs:
                        if change['change_type'] != 'deleted':
                            self.event_handler._handle_document_change(
                                change['file_path'], change['change_type']
                            )
                except Exception as e:
                    self.logger.error(f"Error in periodic check: {e}")

        # Schedule periodic checks every 30 seconds
        schedule.every(30).seconds.do(periodic_check)

        def run_schedule():
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)

        threading.Thread(target=run_schedule, daemon=True).start()

    def add_watch_path(self, path: str):
        """Add a new path to watch"""
        if os.path.exists(path) and path not in self.watch_paths:
            self.watch_paths.append(path)
            if self.is_running:
                self.observer.schedule(self.event_handler, path, recursive=True)
            self.logger.info(f"Added watch path: {path}")

    def remove_watch_path(self, path: str):
        """Remove a path from watching"""
        if path in self.watch_paths:
            self.watch_paths.remove(path)
            # Note: Observer doesn't support removing individual paths easily
            # This would require recreating the observer
            self.logger.info(f"Removed watch path: {path}")

    def get_tracked_documents(self) -> Dict[str, Dict[str, any]]:
        """Get information about currently tracked documents"""
        with self.change_detector._lock:
            tracked = {}
            for file_path in self.change_detector.document_mtimes.keys():
                tracked[file_path] = {
                    'last_modified': self.change_detector.document_mtimes[file_path],
                    'size': self.change_detector.document_sizes.get(file_path, 0),
                    'hash': self.change_detector.document_hashes.get(file_path, ''),
                    'doc_id': self.change_detector._get_doc_id_from_path(file_path)
                }
            return tracked

class VersionLifecycleManager:
    """Manages document lifecycle states"""

    def __init__(self):
        self.lifecycle_stages = {
            'draft': {'order': 1, 'description': 'Initial draft version'},
            'review': {'order': 2, 'description': 'Under review'},
            'approved': {'order': 3, 'description': 'Approved for use'},
            'published': {'order': 4, 'description': 'Published and active'},
            'deprecated': {'order': 5, 'description': 'Deprecated, still available'},
            'archived': {'order': 6, 'description': 'Archived, hidden from users'},
            'deleted': {'order': 7, 'description': 'Marked for deletion'}
        }

    def update_lifecycle_status(self, version_id: str, new_status: str,
                              user: str = None) -> bool:
        """Update lifecycle status of a version"""
        if new_status not in self.lifecycle_stages:
            return False

        version = version_manager.db.get_version(version_id)
        if not version:
            return False

        # Validate transition
        if not self._validate_lifecycle_transition(version.lifecycle_status, new_status):
            return False

        # Update version
        version.lifecycle_status = new_status
        success = version_manager.db.save_version(version)

        if success:
            print(f"Updated lifecycle status of version {version_id} to {new_status}")

            # If archiving/deleting, also update document catalog
            if new_status in ['archived', 'deleted']:
                self._update_document_availability(version.doc_id, new_status)

        return success

    def _validate_lifecycle_transition(self, current_status: str, new_status: str) -> bool:
        """Validate if a lifecycle transition is allowed"""
        current_order = self.lifecycle_stages.get(current_status, {}).get('order', 0)
        new_order = self.lifecycle_stages.get(new_status, {}).get('order', 0)

        # Allow forward transitions and some backward transitions
        if new_order >= current_order:
            return True

        # Allow backward transitions only for certain cases
        allowed_backward = {
            'published': ['approved', 'review', 'draft'],
            'approved': ['review', 'draft'],
            'deprecated': ['published', 'approved', 'review', 'draft']
        }

        return new_status in allowed_backward.get(current_status, [])

    def _update_document_availability(self, doc_id: str, status: str):
        """Update document availability in catalog"""
        try:
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    import json
                    catalog = json.load(f)

                if doc_id in catalog:
                    catalog[doc_id]['lifecycle_status'] = status
                    catalog[doc_id]['is_available'] = status not in ['archived', 'deleted']

                    with open(catalog_path, 'w') as f:
                        json.dump(catalog, f, indent=2)

        except Exception as e:
            print(f"Error updating document availability: {e}")

    def get_lifecycle_history(self, doc_id: str) -> List[Dict[str, any]]:
        """Get lifecycle history for a document"""
        versions = version_manager.db.get_document_versions(doc_id)
        history = []

        for version in versions:
            history.append({
                'version_id': version.version_id,
                'version_number': version.version_number,
                'lifecycle_status': version.lifecycle_status,
                'status_changed_at': version.created_at,
                'status_order': self.lifecycle_stages.get(version.lifecycle_status, {}).get('order', 0)
            })

        # Sort by status order and date
        history.sort(key=lambda x: (x['status_order'], x['status_changed_at']), reverse=True)

        return history

class ChangeNotificationSystem:
    """Handles notifications for document changes"""

    def __init__(self):
        self.notification_handlers = []
        self._lock = threading.Lock()

    def add_notification_handler(self, handler):
        """Add a notification handler"""
        with self._lock:
            self.notification_handlers.append(handler)

    def notify_version_created(self, version: DocumentVersion):
        """Notify about new version creation"""
        notification = {
            'type': 'version_created',
            'version_id': version.version_id,
            'doc_id': version.doc_id,
            'version_number': version.version_number,
            'created_at': version.created_at,
            'change_type': version.change_type,
            'description': version.change_description
        }

        self._send_notifications(notification)

    def notify_lifecycle_change(self, version_id: str, old_status: str, new_status: str):
        """Notify about lifecycle status change"""
        notification = {
            'type': 'lifecycle_change',
            'version_id': version_id,
            'old_status': old_status,
            'new_status': new_status,
            'timestamp': datetime.now()
        }

        self._send_notifications(notification)

    def notify_impact_analysis(self, analysis, affected_users: List[str] = None):
        """Notify about impact analysis results"""
        notification = {
            'type': 'impact_analysis',
            'analysis_id': analysis.analysis_id,
            'doc_id': analysis.doc_id,
            'risk_level': analysis.risk_level,
            'impact_scope': analysis.impact_scope,
            'affected_documents': analysis.affected_documents,
            'recommendations': analysis.recommendations,
            'timestamp': datetime.now()
        }

        self._send_notifications(notification)

    def _send_notifications(self, notification: Dict):
        """Send notification to all handlers"""
        with self._lock:
            for handler in self.notification_handlers:
                try:
                    threading.Thread(
                        target=handler,
                        args=(notification,),
                        daemon=True
                    ).start()
                except Exception as e:
                    print(f"Error in notification handler: {e}")

# Global instances
document_monitor = DocumentMonitor()
lifecycle_manager = VersionLifecycleManager()
notification_system = ChangeNotificationSystem()

def start_document_monitoring():
    """Start the document monitoring system"""
    document_monitor.start_monitoring()

def stop_document_monitoring():
    """Stop the document monitoring system"""
    document_monitor.stop_monitoring()