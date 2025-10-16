"""
Version Control API Endpoints
Provides REST API interface for version control operations
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import json
from typing import Dict, Any, Optional

from .version_control import version_manager
from .version_comparison import document_comparator, version_detection_engine
from .lifecycle_manager import advanced_lifecycle_manager
from .approval_workflow import approval_workflow_manager, workflow_notification_manager
from .rollback_manager import rollback_manager
from .relationship_analyzer import impact_analyzer, relationship_analyzer
from .integration import integration_manager

# Create blueprint
version_api = Blueprint('version_api', __name__)

# Version Management Endpoints
@version_api.route('/documents/<doc_id>/versions', methods=['GET'])
def get_document_versions(doc_id: str):
    """Get all versions for a document"""
    try:
        versions = version_manager.db.get_document_versions(doc_id)

        return jsonify({
            'success': True,
            'document_id': doc_id,
            'versions': [version.to_dict() for version in versions],
            'total_versions': len(versions)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/documents/<doc_id>/versions', methods=['POST'])
def create_document_version(doc_id: str):
    """Create a new version for a document"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        change_type = data.get('change_type', 'auto')
        change_description = data.get('change_description')
        created_by = data.get('created_by')

        if not file_path:
            return jsonify({
                'success': False,
                'error': 'file_path is required'
            }), 400

        version = version_manager.create_version(
            doc_id=doc_id,
            file_path=file_path,
            change_type=change_type,
            change_description=change_description,
            created_by=created_by
        )

        if version:
            return jsonify({
                'success': True,
                'version': version.to_dict(),
                'message': f'Created version {version.version_number}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create version'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/versions/<version_id>', methods=['GET'])
def get_version(version_id: str):
    """Get a specific version"""
    try:
        version = version_manager.db.get_version(version_id)

        if version:
            return jsonify({
                'success': True,
                'version': version.to_dict()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Version not found'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/versions/<version1_id>/compare/<version2_id>', methods=['GET'])
def compare_versions(version1_id: str, version2_id: str):
    """Compare two versions"""
    try:
        comparison = document_comparator.compare_documents(version1_id, version2_id)

        return jsonify({
            'success': True,
            'comparison': comparison.__dict__
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Lifecycle Management Endpoints
@version_api.route('/versions/<version_id>/lifecycle', methods=['PUT'])
def update_lifecycle_status(version_id: str):
    """Update lifecycle status of a version"""
    try:
        data = request.get_json()
        new_status = data.get('status')
        user_id = data.get('user_id')
        reason = data.get('reason')

        if not new_status or not user_id:
            return jsonify({
                'success': False,
                'error': 'status and user_id are required'
            }), 400

        success = advanced_lifecycle_manager.transition_document(
            version_id, new_status, user_id, reason
        )

        if success:
            return jsonify({
                'success': True,
                'message': f'Updated lifecycle status to {new_status}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update lifecycle status'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/documents/<doc_id>/lifecycle', methods=['GET'])
def get_document_lifecycle(doc_id: str):
    """Get lifecycle information for a document"""
    try:
        lifecycle_info = advanced_lifecycle_manager.get_document_lifecycle_info(doc_id)

        return jsonify({
            'success': True,
            'lifecycle_info': lifecycle_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/workflows', methods=['POST'])
def start_workflow():
    """Start a workflow for a document"""
    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        version_id = data.get('version_id')
        workflow_id = data.get('workflow_id')

        if not doc_id or not version_id:
            return jsonify({
                'success': False,
                'error': 'doc_id and version_id are required'
            }), 400

        instance_id = advanced_lifecycle_manager.start_workflow(
            doc_id, version_id, workflow_id
        )

        if instance_id:
            return jsonify({
                'success': True,
                'workflow_instance_id': instance_id,
                'message': 'Workflow started successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start workflow'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/workflows/<instance_id>', methods=['GET'])
def get_workflow_status(instance_id: str):
    """Get workflow status"""
    try:
        status = advanced_lifecycle_manager.get_workflow_status(instance_id)

        if status:
            return jsonify({
                'success': True,
                'workflow_status': status
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Workflow instance not found'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/workflows/<instance_id>/complete-step', methods=['POST'])
def complete_workflow_step(instance_id: str):
    """Complete a workflow step"""
    try:
        data = request.get_json()
        step_id = data.get('step_id')
        user_id = data.get('user_id')
        action = data.get('action')
        notes = data.get('notes')

        if not step_id or not user_id or not action:
            return jsonify({
                'success': False,
                'error': 'step_id, user_id, and action are required'
            }), 400

        success = advanced_lifecycle_manager.complete_workflow_step(
            instance_id, step_id, user_id, action, notes
        )

        if success:
            return jsonify({
                'success': True,
                'message': 'Workflow step completed'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to complete workflow step'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Approval Workflow Endpoints
@version_api.route('/change-requests', methods=['POST'])
def create_change_request():
    """Create a change request"""
    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        request_type = data.get('request_type')
        title = data.get('title')
        description = data.get('description')
        requested_by = data.get('requested_by')
        priority = data.get('priority', 'medium')

        if not all([doc_id, request_type, title, description, requested_by]):
            return jsonify({
                'success': False,
                'error': 'doc_id, request_type, title, description, and requested_by are required'
            }), 400

        request = version_manager.db.save_change_request(type(request_type))

        return jsonify({
            'success': True,
            'request_id': request.request_id,
            'message': 'Change request created'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/change-requests/<doc_id>', methods=['GET'])
def get_change_requests(doc_id: str):
    """Get change requests for a document"""
    try:
        status_filter = request.args.get('status')
        requests = version_manager.db.get_change_requests(doc_id, status_filter)

        return jsonify({
            'success': True,
            'change_requests': [req.to_dict() for req in requests]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/approvals/pending/<user_id>', methods=['GET'])
def get_pending_approvals(user_id: str):
    """Get pending approvals for a user"""
    try:
        approvals = approval_workflow_manager.get_pending_approvals(user_id)

        return jsonify({
            'success': True,
            'pending_approvals': approvals
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/approvals/<request_id>/respond', methods=['POST'])
def respond_to_approval(request_id: str):
    """Respond to an approval request"""
    try:
        data = request.get_json()
        approver_id = data.get('approver_id')
        decision = data.get('decision')  # approved, rejected, delegated
        notes = data.get('notes')

        if not approver_id or not decision:
            return jsonify({
                'success': False,
                'error': 'approver_id and decision are required'
            }), 400

        success = approval_workflow_manager.respond_to_approval(
            request_id, approver_id, decision, notes
        )

        if success:
            return jsonify({
                'success': True,
                'message': f'Approval {decision} successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to process approval response'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Rollback Endpoints
@version_api.route('/documents/<doc_id>/rollback-plan', methods=['POST'])
def create_rollback_plan(doc_id: str):
    """Create a rollback plan"""
    try:
        data = request.get_json()
        target_version_id = data.get('target_version_id')
        rollback_type = data.get('rollback_type', 'full')
        created_by = data.get('created_by')

        if not target_version_id:
            return jsonify({
                'success': False,
                'error': 'target_version_id is required'
            }), 400

        plan_id = rollback_manager.create_rollback_plan(
            doc_id, target_version_id, rollback_type, created_by
        )

        if plan_id:
            return jsonify({
                'success': True,
                'plan_id': plan_id,
                'message': 'Rollback plan created'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create rollback plan'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/rollback/<plan_id>/execute', methods=['POST'])
def execute_rollback(plan_id: str):
    """Execute a rollback plan"""
    try:
        data = request.get_json()
        executed_by = data.get('executed_by')
        force = data.get('force', False)

        if not executed_by:
            return jsonify({
                'success': False,
                'error': 'executed_by is required'
            }), 400

        result = rollback_manager.execute_rollback(plan_id, executed_by, force)

        return jsonify({
            'success': result == 'success',
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/documents/<doc_id>/rollback-preview/<target_version_id>', methods=['GET'])
def get_rollback_preview(doc_id: str, target_version_id: str):
    """Get preview of rollback operation"""
    try:
        preview = rollback_manager.get_rollback_preview(doc_id, target_version_id)

        return jsonify({
            'success': True,
            'preview': preview
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/documents/<doc_id>/rollback-history', methods=['GET'])
def get_rollback_history(doc_id: str):
    """Get rollback history for a document"""
    try:
        history = rollback_manager.get_rollback_history(doc_id)

        return jsonify({
            'success': True,
            'rollback_history': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Relationship and Impact Analysis Endpoints
@version_api.route('/documents/<doc_id>/relationships', methods=['GET'])
def get_document_relationships(doc_id: str):
    """Get relationships for a document"""
    try:
        relationships = version_manager.db.get_document_relationships(doc_id)

        return jsonify({
            'success': True,
            'relationships': [rel.to_dict() for rel in relationships]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/documents/<doc_id>/impact-analysis', methods=['POST'])
def analyze_impact(doc_id: str):
    """Analyze impact of changes to a document"""
    try:
        data = request.get_json()
        version_id = data.get('version_id')
        change_description = data.get('change_description')

        assessment = impact_analyzer.analyze_change_impact(
            doc_id, version_id, change_description
        )

        if assessment:
            return jsonify({
                'success': True,
                'impact_assessment': impact_analyzer.get_impact_summary(assessment)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to analyze impact'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/documents/<doc_id>/dependency-graph', methods=['GET'])
def get_dependency_graph(doc_id: str):
    """Get dependency graph for a document"""
    try:
        max_depth = int(request.args.get('max_depth', 3))
        graph = dependency_resolver.get_dependency_graph(doc_id, max_depth)

        return jsonify({
            'success': True,
            'dependency_graph': graph
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/documents/<doc_id>/relationship-network', methods=['GET'])
def get_relationship_network(doc_id: str):
    """Get relationship network for a document"""
    try:
        # Get related document IDs
        relationships = version_manager.db.get_document_relationships(doc_id)
        related_ids = [rel.target_doc_id for rel in relationships]
        related_ids.append(doc_id)

        network = relationship_analyzer.build_relationship_network(related_ids)

        return jsonify({
            'success': True,
            'relationship_network': network
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Search and Integration Endpoints
@version_api.route('/search/enhanced', methods=['GET'])
def enhanced_search():
    """Enhanced search with version control information"""
    try:
        query = request.args.get('q', '')
        lifecycle_status = request.args.get('lifecycle_status')
        has_versions = request.args.get('has_versions')

        filters = {}
        if lifecycle_status:
            filters['lifecycle_status'] = lifecycle_status
        if has_versions:
            filters['has_versions'] = has_versions.lower() == 'true'

        results = integration_manager.search_integrator.search_documents_with_versions(query, filters)

        return jsonify({
            'success': True,
            'query': query,
            'filters': filters,
            'results': results,
            'total_results': len(results)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/search/versions', methods=['GET'])
def search_versions():
    """Search within document versions"""
    try:
        query = request.args.get('q', '')
        change_type = request.args.get('change_type')
        lifecycle_status = request.args.get('lifecycle_status')

        filters = {}
        if change_type:
            filters['change_type'] = change_type
        if lifecycle_status:
            filters['lifecycle_status'] = lifecycle_status

        results = integration_manager.search_integrator.get_version_search_results(query, filters)

        return jsonify({
            'success': True,
            'query': query,
            'filters': filters,
            'results': results,
            'total_results': len(results)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/system/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    try:
        status = integration_manager.get_system_status()

        return jsonify({
            'success': True,
            'system_status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/documents/<doc_id>/enhanced-metadata', methods=['GET'])
def get_enhanced_metadata(doc_id: str):
    """Get enhanced metadata for a document"""
    try:
        metadata = integration_manager.metadata_integrator.get_enhanced_document_metadata(doc_id)

        if metadata:
            return jsonify({
                'success': True,
                'enhanced_metadata': metadata
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Document not found'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Utility Endpoints
@version_api.route('/documents/<doc_id>/should-version', methods=['POST'])
def check_should_version(doc_id: str):
    """Check if a document should be versioned"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')

        if not file_path:
            return jsonify({
                'success': False,
                'error': 'file_path is required'
            }), 400

        # Get current version for comparison
        versions = version_manager.db.get_document_versions(doc_id)
        current_version = versions[0] if versions else None

        should_version, reason = version_detection_engine.should_create_version(
            file_path, current_version
        )

        return jsonify({
            'success': True,
            'should_create_version': should_version,
            'reason': reason,
            'current_version': current_version.version_number if current_version else None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@version_api.route('/notifications/test', methods=['POST'])
def test_notifications():
    """Test notification system"""
    try:
        data = request.get_json()
        notification_type = data.get('type', 'test')
        recipients = data.get('recipients', ['test@example.com'])

        # Send test notification
        workflow_notification_manager.notification_manager.send_notification(
            notification_type,
            {
                'test_message': 'This is a test notification',
                'timestamp': datetime.now().isoformat()
            },
            recipients
        )

        return jsonify({
            'success': True,
            'message': 'Test notification sent'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error Handlers
@version_api.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@version_api.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Register blueprint function for Flask app integration
def register_version_api(app):
    """Register the version control API blueprint with Flask app"""
    app.register_blueprint(version_api, url_prefix='/api/version-control')