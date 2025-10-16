"""
Integration Module for Version Control System
Integrates version control with existing metadata framework and search capabilities
"""

import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from .version_control import version_manager
from .config import DocumentMetadata

class MetadataIntegrator:
    """Integrates version control data with document metadata"""

    def __init__(self):
        self.catalog_path = Path(__file__).parent.parent / "document_catalog.json"
        self._lock = threading.Lock()

    def update_document_catalog(self, doc_id: str = None):
        """Update document catalog with version control information"""
        try:
            if not self.catalog_path.exists():
                return False

            with self._lock:
                # Load current catalog
                with open(self.catalog_path, 'r') as f:
                    catalog = json.load(f)

            # Update specific document or all documents
            if doc_id:
                self._update_single_document(catalog, doc_id)
            else:
                for document_id in catalog.keys():
                    self._update_single_document(catalog, document_id)

            # Save updated catalog
            with open(self.catalog_path, 'w') as f:
                json.dump(catalog, f, indent=2)

            return True

        except Exception as e:
            print(f"Error updating document catalog: {e}")
            return False

    def _update_single_document(self, catalog: Dict, doc_id: str):
        """Update version control information for a single document"""
        if doc_id not in catalog:
            return

        # Get version information
        versions = version_manager.db.get_document_versions(doc_id)

        if versions:
            # Sort versions by creation date (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)
            latest_version = versions[0]

            # Update catalog entry
            catalog[doc_id].update({
                'version_count': len(versions),
                'current_version': latest_version.version_number,
                'current_version_id': latest_version.version_id,
                'last_version_update': latest_version.created_at.isoformat(),
                'lifecycle_status': latest_version.lifecycle_status,
                'approval_status': latest_version.approval_status,
                'version_history': [
                    {
                        'version_id': v.version_id,
                        'version_number': v.version_number,
                        'created_at': v.created_at.isoformat(),
                        'change_type': v.change_type,
                        'lifecycle_status': v.lifecycle_status
                    }
                    for v in versions[:10]  # Last 10 versions
                ]
            })

            # Add relationship information
            relationships = version_manager.db.get_document_relationships(doc_id)
            if relationships:
                catalog[doc_id]['relationships'] = [
                    {
                        'target_doc_id': rel.target_doc_id if rel.source_doc_id == doc_id else rel.source_doc_id,
                        'relationship_type': rel.relationship_type,
                        'strength': rel.strength
                    }
                    for rel in relationships
                ]

        # Add change request information
        change_requests = version_manager.db.get_change_requests(doc_id)
        if change_requests:
            pending_requests = [cr for cr in change_requests if cr.status == 'pending']
            catalog[doc_id]['pending_change_requests'] = len(pending_requests)

    def get_enhanced_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata enhanced with version control information"""
        try:
            # Load from catalog
            if not self.catalog_path.exists():
                return None

            with open(self.catalog_path, 'r') as f:
                catalog = json.load(f)

            if doc_id not in catalog:
                return None

            doc_data = catalog[doc_id].copy()

            # Add version control enhancements
            versions = version_manager.db.get_document_versions(doc_id)
            if versions:
                doc_data['version_control'] = {
                    'total_versions': len(versions),
                    'latest_version': versions[0].to_dict(),
                    'version_timeline': [v.to_dict() for v in versions[:5]]
                }

            # Add relationship information
            relationships = version_manager.db.get_document_relationships(doc_id)
            if relationships:
                doc_data['relationships'] = {
                    'total_relationships': len(relationships),
                    'relationship_details': [rel.to_dict() for rel in relationships]
                }

            # Add lifecycle information
            lifecycle_info = advanced_lifecycle_manager.get_document_lifecycle_info(doc_id)
            if lifecycle_info:
                doc_data['lifecycle'] = lifecycle_info

            return doc_data

        except Exception as e:
            print(f"Error getting enhanced metadata for {doc_id}: {e}")
            return None

class SearchIntegrator:
    """Integrates version control with search functionality"""

    def __init__(self):
        self.metadata_integrator = MetadataIntegrator()

    def search_documents_with_versions(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search documents including version information"""
        # This would integrate with the existing search engine
        # For now, return enhanced results from catalog

        try:
            if not self.metadata_integrator.catalog_path.exists():
                return []

            with open(self.metadata_integrator.catalog_path, 'r') as f:
                catalog = json.load(f)

            results = []
            query_lower = query.lower()

            for doc_id, doc_data in catalog.items():
                # Basic text search in filename and content
                searchable_text = (doc_data.get('file_name', '') + ' ' +
                                 doc_data.get('content_preview', '')).lower()

                if query_lower in searchable_text:
                    # Apply filters
                    if self._matches_filters(doc_data, filters):
                        # Enhance with version control data
                        enhanced_data = self.metadata_integrator.get_enhanced_document_metadata(doc_id)
                        if enhanced_data:
                            results.append(enhanced_data)

            return results

        except Exception as e:
            print(f"Error searching documents with versions: {e}")
            return []

    def _matches_filters(self, doc_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches search filters"""
        if not filters:
            return True

        for key, value in filters.items():
            if key == 'lifecycle_status':
                if doc_data.get('lifecycle_status') != value:
                    return False
            elif key == 'has_versions':
                version_count = doc_data.get('version_count', 0)
                if value and version_count == 0:
                    return False
                elif not value and version_count > 0:
                    return False
            elif key == 'file_extension':
                if doc_data.get('file_extension') != value:
                    return False

        return True

    def get_version_search_results(self, query: str, version_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search within specific versions of documents"""
        results = []

        try:
            # Get all documents with versions
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            if not catalog_path.exists():
                return results

            with open(catalog_path, 'r') as f:
                catalog = json.load(f)

            for doc_id, doc_data in catalog.items():
                # Get all versions for this document
                versions = version_manager.db.get_document_versions(doc_id)

                for version in versions:
                    # Search in version content if available
                    if version.content_snapshot:
                        content_lower = version.content_snapshot.lower()
                        if query.lower() in content_lower:
                            # Check version filters
                            if self._matches_version_filters(version, version_filters):
                                results.append({
                                    'document': doc_data,
                                    'version': version.to_dict(),
                                    'matched_in_version': True
                                })

        except Exception as e:
            print(f"Error searching versions: {e}")

        return results

    def _matches_version_filters(self, version: DocumentVersion, filters: Dict[str, Any]) -> bool:
        """Check if version matches filters"""
        if not filters:
            return True

        for key, value in filters.items():
            if key == 'change_type':
                if version.change_type != value:
                    return False
            elif key == 'lifecycle_status':
                if version.lifecycle_status != value:
                    return False
            elif key == 'created_after':
                if version.created_at < datetime.fromisoformat(value):
                    return False
            elif key == 'created_before':
                if version.created_at > datetime.fromisoformat(value):
                    return False

        return True

class VersionSearchEnhancer:
    """Enhances search results with version control information"""

    def __init__(self):
        self.search_integrator = SearchIntegrator()

    def enhance_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance search results with version control data"""
        enhanced_results = []

        for result in search_results:
            doc_id = result.get('doc_id')
            if doc_id:
                # Get enhanced metadata
                enhanced_metadata = self.search_integrator.metadata_integrator.get_enhanced_document_metadata(doc_id)
                if enhanced_metadata:
                    # Merge with original result
                    enhanced_result = result.copy()
                    enhanced_result.update({
                        'version_control_info': enhanced_metadata.get('version_control', {}),
                        'lifecycle_info': enhanced_metadata.get('lifecycle', {}),
                        'relationships': enhanced_metadata.get('relationships', {})
                    })
                    enhanced_results.append(enhanced_result)
                else:
                    enhanced_results.append(result)
            else:
                enhanced_results.append(result)

        return enhanced_results

    def add_version_facets(self, search_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Add version-related facets to search results"""
        facets = {
            'lifecycle_status': {},
            'change_type': {},
            'has_versions': {},
            'approval_status': {}
        }

        for result in search_results:
            # Lifecycle status facet
            lifecycle_status = result.get('lifecycle_status', 'unknown')
            facets['lifecycle_status'][lifecycle_status] = facets['lifecycle_status'].get(lifecycle_status, 0) + 1

            # Change type facet (from latest version)
            version_control = result.get('version_control_info', {})
            if version_control and 'latest_version' in version_control:
                change_type = version_control['latest_version'].get('change_type', 'unknown')
                facets['change_type'][change_type] = facets['change_type'].get(change_type, 0) + 1

            # Has versions facet
            has_versions = result.get('version_count', 0) > 1
            has_versions_str = 'yes' if has_versions else 'no'
            facets['has_versions'][has_versions_str] = facets['has_versions'].get(has_versions_str, 0) + 1

            # Approval status facet
            approval_status = result.get('approval_status', 'unknown')
            facets['approval_status'][approval_status] = facets['approval_status'].get(approval_status, 0) + 1

        return facets

class CatalogSynchronizer:
    """Synchronizes version control data with document catalog"""

    def __init__(self):
        self.metadata_integrator = MetadataIntegrator()
        self.is_running = False
        self._lock = threading.Lock()

    def start_synchronization(self, interval_minutes: int = 30):
        """Start periodic catalog synchronization"""
        def sync_loop():
            while self.is_running:
                try:
                    self._perform_synchronization()
                    time.sleep(interval_minutes * 60)
                except Exception as e:
                    print(f"Error in catalog synchronization: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying

        with self._lock:
            if not self.is_running:
                self.is_running = True
                threading.Thread(target=sync_loop, daemon=True).start()
                print(f"Catalog synchronization started (interval: {interval_minutes} minutes)")

    def stop_synchronization(self):
        """Stop periodic catalog synchronization"""
        with self._lock:
            self.is_running = False
            print("Catalog synchronization stopped")

    def _perform_synchronization(self):
        """Perform full catalog synchronization"""
        print("Starting catalog synchronization...")

        # Update all documents in catalog
        success = self.metadata_integrator.update_document_catalog()

        if success:
            print("Catalog synchronization completed successfully")
        else:
            print("Catalog synchronization failed")

    def force_synchronization(self, doc_id: str = None):
        """Force immediate synchronization"""
        print(f"Forcing catalog synchronization for document: {doc_id or 'all'}")
        return self.metadata_integrator.update_document_catalog(doc_id)

class VersionAwareDocumentProcessor:
    """Document processor that integrates with version control"""

    def __init__(self):
        self.catalog_synchronizer = CatalogSynchronizer()

    def process_document_with_versioning(self, file_path: str, doc_id: str = None) -> Dict[str, Any]:
        """Process a document with version control integration"""
        try:
            # Extract document ID if not provided
            if not doc_id:
                doc_id = self._extract_doc_id_from_path(file_path)

            # Create initial version if document doesn't exist in version control
            versions = version_manager.db.get_document_versions(doc_id)
            if not versions:
                version = version_manager.create_version(
                    doc_id=doc_id,
                    file_path=file_path,
                    change_type="initial",
                    change_description="Initial version created"
                )

                if version:
                    print(f"Created initial version {version.version_number} for document {doc_id}")

            # Update catalog with version information
            self.catalog_synchronizer.force_synchronization(doc_id)

            return {
                'success': True,
                'doc_id': doc_id,
                'version_count': len(version_manager.db.get_document_versions(doc_id))
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_doc_id_from_path(self, file_path: str) -> str:
        """Extract document ID from file path"""
        # This would use the existing document ID generation logic
        # For now, use a simple hash-based approach
        import hashlib
        path_str = str(Path(file_path).resolve())
        return hashlib.md5(path_str.encode()).hexdigest()

class IntegrationManager:
    """Main integration manager for all systems"""

    def __init__(self):
        self.metadata_integrator = MetadataIntegrator()
        self.search_integrator = SearchIntegrator()
        self.catalog_synchronizer = CatalogSynchronizer()
        self.document_processor = VersionAwareDocumentProcessor()

    def initialize_integration(self):
        """Initialize all integrations"""
        print("Initializing version control integration...")

        # Start catalog synchronization
        self.catalog_synchronizer.start_synchronization()

        # Update existing catalog with version information
        self.catalog_synchronizer.force_synchronization()

        print("Version control integration initialized")

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all integrated systems"""
        try:
            # Get catalog statistics
            catalog_path = Path(__file__).parent.parent / "document_catalog.json"
            catalog_stats = {'exists': False, 'document_count': 0}

            if catalog_path.exists():
                catalog_stats['exists'] = True
                with open(catalog_path, 'r') as f:
                    catalog = json.load(f)
                catalog_stats['document_count'] = len(catalog)

            # Get version control statistics
            version_stats = {
                'total_versions': 0,
                'total_relationships': 0,
                'total_change_requests': 0
            }

            # This would query the version control database for stats
            # For now, use placeholder values

            return {
                'catalog': catalog_stats,
                'version_control': version_stats,
                'integration_status': 'active',
                'last_sync': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'error': str(e),
                'integration_status': 'error'
            }

# Global integration manager
integration_manager = IntegrationManager()

# Auto-initialize integration when module is imported
integration_manager.initialize_integration()