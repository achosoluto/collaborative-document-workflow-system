"""
Version Control System for Document Management
Provides comprehensive document versioning, change tracking, and lifecycle management
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import hashlib
import difflib
import os
from pathlib import Path
import sqlite3
import threading
from .config import search_config, BASE_DIR

@dataclass
class DocumentVersion:
    """Represents a specific version of a document"""

    version_id: str
    doc_id: str
    version_number: str
    file_path: str
    file_hash: str
    file_size: int
    created_at: datetime
    created_by: Optional[str] = None

    # Version metadata
    change_type: str = "unknown"  # major, minor, patch, auto
    change_description: Optional[str] = None
    parent_version_id: Optional[str] = None

    # Content comparison
    content_snapshot: Optional[str] = None
    diff_from_parent: Optional[str] = None

    # Lifecycle status
    lifecycle_status: str = "active"  # active, deprecated, archived
    approval_status: str = "pending"  # pending, approved, rejected

    # Relationships
    related_versions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'version_id': self.version_id,
            'doc_id': self.doc_id,
            'version_number': self.version_number,
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'file_size': self.file_size,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'change_type': self.change_type,
            'change_description': self.change_description,
            'parent_version_id': self.parent_version_id,
            'content_snapshot': self.content_snapshot,
            'diff_from_parent': self.diff_from_parent,
            'lifecycle_status': self.lifecycle_status,
            'approval_status': self.approval_status,
            'related_versions': self.related_versions,
            'tags': self.tags
        }

@dataclass
class DocumentRelationship:
    """Represents relationships between documents"""

    relationship_id: str
    source_doc_id: str
    target_doc_id: str
    relationship_type: str  # dependency, reference, related, parent, child
    strength: float = 1.0  # 0.0 to 1.0

    # Metadata
    created_at: datetime
    created_by: Optional[str] = None
    description: Optional[str] = None
    is_bidirectional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'relationship_id': self.relationship_id,
            'source_doc_id': self.source_doc_id,
            'target_doc_id': self.target_doc_id,
            'relationship_type': self.relationship_type,
            'strength': self.strength,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'description': self.description,
            'is_bidirectional': self.is_bidirectional
        }

@dataclass
class ChangeRequest:
    """Represents a change request for document modification"""

    request_id: str
    doc_id: str
    requested_by: str
    request_type: str  # create, update, delete, rollback
    target_version_id: Optional[str] = None

    # Request details
    title: str
    description: str
    priority: str = "medium"  # low, medium, high, critical
    status: str = "pending"  # pending, approved, rejected, implemented

    # Dates
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    implemented_at: Optional[datetime] = None

    # Approval workflow
    assigned_reviewer: Optional[str] = None
    review_notes: Optional[str] = None
    approval_chain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'request_id': self.request_id,
            'doc_id': self.doc_id,
            'requested_by': self.requested_by,
            'request_type': self.request_type,
            'target_version_id': self.target_version_id,
            'title': self.title,
            'description': self.description,
            'priority': self.priority,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'implemented_at': self.implemented_at.isoformat() if self.implemented_at else None,
            'assigned_reviewer': self.assigned_reviewer,
            'review_notes': self.review_notes,
            'approval_chain': self.approval_chain
        }

@dataclass
class ImpactAnalysis:
    """Represents impact analysis for document changes"""

    analysis_id: str
    doc_id: str
    version_id: str
    analysis_type: str  # dependency, reference, content, metadata

    # Impact assessment
    affected_documents: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high, critical
    impact_scope: str = "minimal"  # minimal, moderate, significant, extensive

    # Analysis details
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'analysis_id': self.analysis_id,
            'doc_id': self.doc_id,
            'version_id': self.version_id,
            'analysis_type': self.analysis_type,
            'affected_documents': self.affected_documents,
            'risk_level': self.risk_level,
            'impact_scope': self.impact_scope,
            'analysis_results': self.analysis_results,
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat()
        }

class VersionControlDB:
    """Database manager for version control system"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = BASE_DIR / "data" / "version_control.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    version_number TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    change_type TEXT NOT NULL,
                    change_description TEXT,
                    parent_version_id TEXT,
                    content_snapshot TEXT,
                    diff_from_parent TEXT,
                    lifecycle_status TEXT NOT NULL,
                    approval_status TEXT NOT NULL,
                    related_versions TEXT,
                    tags TEXT,
                    UNIQUE(doc_id, version_number)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_relationships (
                    relationship_id TEXT PRIMARY KEY,
                    source_doc_id TEXT NOT NULL,
                    target_doc_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    description TEXT,
                    is_bidirectional BOOLEAN NOT NULL,
                    UNIQUE(source_doc_id, target_doc_id, relationship_type)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS change_requests (
                    request_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    target_version_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    implemented_at TEXT,
                    assigned_reviewer TEXT,
                    review_notes TEXT,
                    approval_chain TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS impact_analyses (
                    analysis_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    affected_documents TEXT,
                    risk_level TEXT NOT NULL,
                    impact_scope TEXT NOT NULL,
                    analysis_results TEXT,
                    recommendations TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_versions_doc_id ON document_versions(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_versions_created_at ON document_versions(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON document_relationships(source_doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON document_relationships(target_doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_change_requests_doc_id ON change_requests(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_change_requests_status ON change_requests(status)")

    def save_version(self, version: DocumentVersion) -> bool:
        """Save a document version to the database"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO document_versions
                        (version_id, doc_id, version_number, file_path, file_hash, file_size,
                         created_at, created_by, change_type, change_description, parent_version_id,
                         content_snapshot, diff_from_parent, lifecycle_status, approval_status,
                         related_versions, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        version.version_id, version.doc_id, version.version_number,
                        version.file_path, version.file_hash, version.file_size,
                        version.created_at.isoformat(), version.created_by, version.change_type,
                        version.change_description, version.parent_version_id,
                        version.content_snapshot, version.diff_from_parent,
                        version.lifecycle_status, version.approval_status,
                        json.dumps(version.related_versions), json.dumps(version.tags)
                    ))
            return True
        except Exception as e:
            print(f"Error saving version: {e}")
            return False

    def get_version(self, version_id: str) -> Optional[DocumentVersion]:
        """Retrieve a specific version"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM document_versions WHERE version_id = ?",
                    (version_id,)
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_version(row)
        except Exception as e:
            print(f"Error getting version: {e}")
        return None

    def get_document_versions(self, doc_id: str) -> List[DocumentVersion]:
        """Get all versions for a document"""
        versions = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM document_versions WHERE doc_id = ? ORDER BY created_at DESC",
                    (doc_id,)
                )
                for row in cursor:
                    versions.append(self._row_to_version(row))
        except Exception as e:
            print(f"Error getting document versions: {e}")
        return versions

    def save_relationship(self, relationship: DocumentRelationship) -> bool:
        """Save a document relationship"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO document_relationships
                        (relationship_id, source_doc_id, target_doc_id, relationship_type,
                         strength, created_at, created_by, description, is_bidirectional)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        relationship.relationship_id, relationship.source_doc_id,
                        relationship.target_doc_id, relationship.relationship_type,
                        relationship.strength, relationship.created_at.isoformat(),
                        relationship.created_by, relationship.description,
                        relationship.is_bidirectional
                    ))
            return True
        except Exception as e:
            print(f"Error saving relationship: {e}")
            return False

    def get_document_relationships(self, doc_id: str) -> List[DocumentRelationship]:
        """Get all relationships for a document"""
        relationships = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM document_relationships WHERE source_doc_id = ? OR target_doc_id = ?",
                    (doc_id, doc_id)
                )
                for row in cursor:
                    relationships.append(self._row_to_relationship(row))
        except Exception as e:
            print(f"Error getting relationships: {e}")
        return relationships

    def save_change_request(self, request: ChangeRequest) -> bool:
        """Save a change request"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO change_requests
                        (request_id, doc_id, requested_by, request_type, target_version_id,
                         title, description, priority, status, created_at, reviewed_at,
                         implemented_at, assigned_reviewer, review_notes, approval_chain)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        request.request_id, request.doc_id, request.requested_by,
                        request.request_type, request.target_version_id, request.title,
                        request.description, request.priority, request.status,
                        request.created_at.isoformat(),
                        request.reviewed_at.isoformat() if request.reviewed_at else None,
                        request.implemented_at.isoformat() if request.implemented_at else None,
                        request.assigned_reviewer, request.review_notes,
                        json.dumps(request.approval_chain)
                    ))
            return True
        except Exception as e:
            print(f"Error saving change request: {e}")
            return False

    def get_change_requests(self, doc_id: str = None, status: str = None) -> List[ChangeRequest]:
        """Get change requests with optional filtering"""
        requests = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                query = "SELECT * FROM change_requests WHERE 1=1"
                params = []

                if doc_id:
                    query += " AND doc_id = ?"
                    params.append(doc_id)

                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY created_at DESC"

                cursor = conn.execute(query, params)
                for row in cursor:
                    requests.append(self._row_to_change_request(row))
        except Exception as e:
            print(f"Error getting change requests: {e}")
        return requests

    def save_impact_analysis(self, analysis: ImpactAnalysis) -> bool:
        """Save impact analysis"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO impact_analyses
                        (analysis_id, doc_id, version_id, analysis_type, affected_documents,
                         risk_level, impact_scope, analysis_results, recommendations, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        analysis.analysis_id, analysis.doc_id, analysis.version_id,
                        analysis.analysis_type, json.dumps(analysis.affected_documents),
                        analysis.risk_level, analysis.impact_scope,
                        json.dumps(analysis.analysis_results),
                        json.dumps(analysis.recommendations),
                        analysis.created_at.isoformat()
                    ))
            return True
        except Exception as e:
            print(f"Error saving impact analysis: {e}")
            return False

    def _row_to_version(self, row) -> DocumentVersion:
        """Convert database row to DocumentVersion object"""
        return DocumentVersion(
            version_id=row['version_id'],
            doc_id=row['doc_id'],
            version_number=row['version_number'],
            file_path=row['file_path'],
            file_hash=row['file_hash'],
            file_size=row['file_size'],
            created_at=datetime.fromisoformat(row['created_at']),
            created_by=row['created_by'],
            change_type=row['change_type'],
            change_description=row['change_description'],
            parent_version_id=row['parent_version_id'],
            content_snapshot=row['content_snapshot'],
            diff_from_parent=row['diff_from_parent'],
            lifecycle_status=row['lifecycle_status'],
            approval_status=row['approval_status'],
            related_versions=json.loads(row['related_versions'] or '[]'),
            tags=json.loads(row['tags'] or '[]')
        )

    def _row_to_relationship(self, row) -> DocumentRelationship:
        """Convert database row to DocumentRelationship object"""
        return DocumentRelationship(
            relationship_id=row['relationship_id'],
            source_doc_id=row['source_doc_id'],
            target_doc_id=row['target_doc_id'],
            relationship_type=row['relationship_type'],
            strength=row['strength'],
            created_at=datetime.fromisoformat(row['created_at']),
            created_by=row['created_by'],
            description=row['description'],
            is_bidirectional=bool(row['is_bidirectional'])
        )

    def _row_to_change_request(self, row) -> ChangeRequest:
        """Convert database row to ChangeRequest object"""
        return ChangeRequest(
            request_id=row['request_id'],
            doc_id=row['doc_id'],
            requested_by=row['requested_by'],
            request_type=row['request_type'],
            target_version_id=row['target_version_id'],
            title=row['title'],
            description=row['description'],
            priority=row['priority'],
            status=row['status'],
            created_at=datetime.fromisoformat(row['created_at']),
            reviewed_at=datetime.fromisoformat(row['reviewed_at']) if row['reviewed_at'] else None,
            implemented_at=datetime.fromisoformat(row['implemented_at']) if row['implemented_at'] else None,
            assigned_reviewer=row['assigned_reviewer'],
            review_notes=row['review_notes'],
            approval_chain=json.loads(row['approval_chain'] or '[]')
        )

class VersionControlManager:
    """Main manager for version control operations"""

    def __init__(self):
        self.db = VersionControlDB()
        self._lock = threading.Lock()

    def create_version(self, doc_id: str, file_path: str, change_type: str = "auto",
                      change_description: str = None, created_by: str = None) -> Optional[DocumentVersion]:
        """Create a new version of a document"""

        # Get existing versions to determine version number
        existing_versions = self.db.get_document_versions(doc_id)
        if existing_versions:
            latest_version = max(existing_versions, key=lambda v: self._parse_version_number(v.version_number))
            next_version = self._increment_version(latest_version.version_number, change_type)
            parent_version_id = latest_version.version_id
        else:
            next_version = "1.0.0"
            parent_version_id = None

        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        file_size = os.path.getsize(file_path)

        # Create version object
        version = DocumentVersion(
            version_id=f"{doc_id}_v{next_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            doc_id=doc_id,
            version_number=next_version,
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            created_at=datetime.now(),
            created_by=created_by,
            change_type=change_type,
            change_description=change_description,
            parent_version_id=parent_version_id
        )

        # Generate diff if parent exists
        if parent_version_id:
            parent_version = self.db.get_version(parent_version_id)
            if parent_version and parent_version.content_snapshot:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        current_content = f.read()
                    version.content_snapshot = current_content
                    version.diff_from_parent = self._generate_diff(
                        parent_version.content_snapshot, current_content
                    )
                except:
                    version.content_snapshot = f"<binary file: {os.path.basename(file_path)}>"

        # Save version
        if self.db.save_version(version):
            return version
        return None

    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Compare two versions of a document"""
        v1 = self.db.get_version(version1_id)
        v2 = self.db.get_version(version2_id)

        if not v1 or not v2:
            return {'error': 'Version not found'}

        comparison = {
            'version1': v1.to_dict(),
            'version2': v2.to_dict(),
            'differences': []
        }

        # Compare metadata
        metadata_diff = self._compare_metadata(v1, v2)
        comparison['differences'].extend(metadata_diff)

        # Compare content if available
        if v1.content_snapshot and v2.content_snapshot:
            content_diff = self._generate_diff(v1.content_snapshot, v2.content_snapshot)
            comparison['content_diff'] = content_diff

        return comparison

    def rollback_to_version(self, doc_id: str, target_version_id: str,
                          created_by: str = None) -> Optional[DocumentVersion]:
        """Rollback a document to a specific version"""
        target_version = self.db.get_version(target_version_id)
        if not target_version:
            return None

        # Create new version based on target version
        rollback_version = self.create_version(
            doc_id=doc_id,
            file_path=target_version.file_path,
            change_type="rollback",
            change_description=f"Rollback to version {target_version.version_number}",
            created_by=created_by
        )

        if rollback_version:
            rollback_version.parent_version_id = target_version.version_id
            self.db.save_version(rollback_version)

        return rollback_version

    def get_document_history(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get complete history for a document"""
        versions = self.db.get_document_versions(doc_id)
        relationships = self.db.get_document_relationships(doc_id)

        history = {
            'document_id': doc_id,
            'versions': [v.to_dict() for v in versions],
            'relationships': [r.to_dict() for r in relationships],
            'timeline': self._build_timeline(versions)
        }

        return history

    def analyze_impact(self, doc_id: str, version_id: str = None) -> ImpactAnalysis:
        """Analyze the impact of changes to a document"""
        analysis = ImpactAnalysis(
            analysis_id=f"impact_{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            doc_id=doc_id,
            version_id=version_id or "current",
            analysis_type="comprehensive",
            created_at=datetime.now()
        )

        # Get document relationships
        relationships = self.db.get_document_relationships(doc_id)

        # Analyze affected documents
        affected_docs = []
        for rel in relationships:
            if rel.source_doc_id == doc_id:
                affected_docs.append({
                    'doc_id': rel.target_doc_id,
                    'relationship_type': rel.relationship_type,
                    'impact_level': self._calculate_impact_level(rel)
                })

        analysis.affected_documents = [doc['doc_id'] for doc in affected_docs]

        # Assess risk level
        analysis.risk_level = self._assess_risk_level(affected_docs)
        analysis.impact_scope = self._assess_impact_scope(affected_docs)

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(affected_docs, analysis.risk_level)

        # Save analysis
        self.db.save_impact_analysis(analysis)

        return analysis

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _parse_version_number(self, version_str: str) -> Tuple[int, int, int]:
        """Parse version string into tuple"""
        parts = version_str.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    def _increment_version(self, current_version: str, change_type: str) -> str:
        """Increment version number based on change type"""
        major, minor, patch = self._parse_version_number(current_version)

        if change_type == "major":
            return f"{major + 1}.0.0"
        elif change_type == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch or auto
            return f"{major}.{minor}.{patch + 1}"

    def _generate_diff(self, old_content: str, new_content: str) -> str:
        """Generate diff between two content strings"""
        return '\n'.join(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile='old',
            tofile='new'
        ))

    def _compare_metadata(self, v1: DocumentVersion, v2: DocumentVersion) -> List[Dict[str, Any]]:
        """Compare metadata between two versions"""
        differences = []

        fields_to_compare = ['file_size', 'change_type', 'lifecycle_status', 'approval_status']

        for field in fields_to_compare:
            val1 = getattr(v1, field)
            val2 = getattr(v2, field)
            if val1 != val2:
                differences.append({
                    'field': field,
                    'old_value': val1,
                    'new_value': val2
                })

        return differences

    def _build_timeline(self, versions: List[DocumentVersion]) -> List[Dict[str, Any]]:
        """Build timeline view of version history"""
        timeline = []
        for version in versions:
            timeline.append({
                'version_id': version.version_id,
                'version_number': version.version_number,
                'created_at': version.created_at,
                'change_type': version.change_type,
                'description': version.change_description,
                'created_by': version.created_by
            })
        return timeline

    def _calculate_impact_level(self, relationship: DocumentRelationship) -> str:
        """Calculate impact level for a relationship"""
        if relationship.relationship_type == "dependency":
            return "high"
        elif relationship.relationship_type == "reference":
            return "medium"
        else:
            return "low"

    def _assess_risk_level(self, affected_docs: List[Dict[str, Any]]) -> str:
        """Assess overall risk level"""
        if not affected_docs:
            return "low"

        high_impact_count = sum(1 for doc in affected_docs if doc['impact_level'] == 'high')
        if high_impact_count > 0:
            return "high"
        elif len(affected_docs) > 5:
            return "medium"
        else:
            return "low"

    def _assess_impact_scope(self, affected_docs: List[Dict[str, Any]]) -> str:
        """Assess impact scope"""
        if len(affected_docs) > 10:
            return "extensive"
        elif len(affected_docs) > 5:
            return "significant"
        elif len(affected_docs) > 2:
            return "moderate"
        else:
            return "minimal"

    def _generate_recommendations(self, affected_docs: List[Dict[str, Any]], risk_level: str) -> List[str]:
        """Generate recommendations based on impact analysis"""
        recommendations = []

        if risk_level == "high":
            recommendations.append("Review all high-impact dependent documents before proceeding")
            recommendations.append("Consider phased rollout to minimize disruption")

        if affected_docs:
            recommendations.append(f"Notify stakeholders of {len(affected_docs)} affected documents")

        recommendations.append("Create backup before implementing changes")
        recommendations.append("Test changes in staging environment first")

        return recommendations

# Global version control manager instance
version_manager = VersionControlManager()