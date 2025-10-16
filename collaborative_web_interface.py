"""
Collaborative Workflow Web Interface
Flask-based web interface for the collaborative document workflow system
"""

import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

from flask import Blueprint, render_template, request, jsonify, session

# Import collaborative components
from .collaborative_workflow import collaborative_manager, UserRole, WorkflowStatus
from .annotation_system import annotation_manager, AnnotationType, CommentStatus
from .notification_system import notification_manager
from .workflow_automation import automation_engine, task_manager
from .collaborative_dashboard import dashboard_manager
from .collaborative_integration import integration_manager

# Create Flask blueprint for collaborative features
collaborative_bp = Blueprint('collaborative', __name__, url_prefix='/collaborative')


@collaborative_bp.route('/dashboard')
def collaborative_dashboard():
    """Main collaborative dashboard"""
    return render_template('collaborative_dashboard.html')


@collaborative_bp.route('/document/<doc_id>')
def document_collaboration(doc_id):
    """Document collaboration interface"""
    return render_template('document_collaboration.html', doc_id=doc_id)


@collaborative_bp.route('/api/initialize_user', methods=['POST'])
def api_initialize_user():
    """Initialize a user in the collaborative system"""
    data = request.get_json()
    user_id = data.get('user_id')
    username = data.get('username')
    email = data.get('email')
    role = data.get('role', 'viewer')

    if not all([user_id, username, email]):
        return jsonify({'error': 'Missing required user information'}), 400

    try:
        user_role = UserRole(role)
        user = collaborative_manager.add_user(user_id, username, email, user_role)

        return jsonify({
            'success': True,
            'user': user.to_dict()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/document/<doc_id>/collaborators', methods=['GET'])
def api_get_document_collaborators(doc_id):
    """Get collaborators for a document"""
    try:
        collaborators = collaborative_manager.get_document_collaborators(doc_id)
        return jsonify({
            'success': True,
            'collaborators': [c.to_dict() for c in collaborators]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/document/<doc_id>/collaborators', methods=['POST'])
def api_add_document_collaborator(doc_id):
    """Add a collaborator to a document"""
    data = request.get_json()
    user_id = data.get('user_id')
    role = data.get('role', 'viewer')

    if not user_id:
        return jsonify({'error': 'User ID required'}), 400

    try:
        user_role = UserRole(role)
        collaborative_manager.add_document_collaborator(doc_id, user_id, user_role)

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/document/<doc_id>/annotations', methods=['GET'])
def api_get_document_annotations(doc_id):
    """Get annotations for a document"""
    try:
        annotation_type = request.args.get('type')
        status_filter = request.args.get('status')

        annotation_type_enum = None
        if annotation_type:
            annotation_type_enum = AnnotationType(annotation_type)

        status_enum = None
        if status_filter:
            status_enum = CommentStatus(status_filter)

        annotations = annotation_manager.get_document_annotations(
            doc_id, annotation_type_enum, status_enum
        )

        return jsonify({
            'success': True,
            'annotations': [a.to_dict() for a in annotations]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/document/<doc_id>/annotations', methods=['POST'])
def api_add_document_annotation(doc_id):
    """Add an annotation to a document"""
    data = request.get_json()
    annotation_type = data.get('type', 'comment')
    content = data.get('content')
    position = data.get('position', {})
    user_id = data.get('user_id')

    if not all([content, user_id]):
        return jsonify({'error': 'Content and user_id required'}), 400

    try:
        annotation_type_enum = AnnotationType(annotation_type)
        user = collaborative_manager.users.get(user_id)

        if not user:
            return jsonify({'error': 'User not found'}), 404

        annotation = annotation_manager.create_annotation(
            doc_id, annotation_type_enum, content, position, user_id, user.username
        )

        return jsonify({
            'success': True,
            'annotation': annotation.to_dict()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/annotations/<annotation_id>/reply', methods=['POST'])
def api_reply_to_annotation(annotation_id):
    """Reply to an annotation"""
    data = request.get_json()
    content = data.get('content')
    user_id = data.get('user_id')

    if not all([content, user_id]):
        return jsonify({'error': 'Content and user_id required'}), 400

    try:
        user = collaborative_manager.users.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        reply = annotation_manager.reply_to_annotation(
            annotation_id, content, user_id, user.username
        )

        return jsonify({
            'success': True,
            'annotation': reply.to_dict()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/document/<doc_id>/workflows', methods=['GET'])
def api_get_document_workflows(doc_id):
    """Get workflows for a document"""
    try:
        workflows = [
            w for w in collaborative_manager.workflows.values()
            if w.document_id == doc_id
        ]

        return jsonify({
            'success': True,
            'workflows': [w.to_dict() for w in workflows]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/document/<doc_id>/workflows', methods=['POST'])
def api_create_document_workflow(doc_id):
    """Create a workflow for a document"""
    data = request.get_json()
    title = data.get('title')
    description = data.get('description', '')
    user_id = data.get('user_id')
    reviewers = data.get('reviewers', [])
    approvers = data.get('approvers', [])

    if not all([title, user_id]):
        return jsonify({'error': 'Title and user_id required'}), 400

    try:
        user = collaborative_manager.users.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        workflow = collaborative_manager.create_workflow(
            doc_id, title, description, user_id, reviewers, approvers
        )

        return jsonify({
            'success': True,
            'workflow': workflow.to_dict()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/document/<doc_id>/tasks', methods=['GET'])
def api_get_document_tasks(doc_id):
    """Get tasks for a document"""
    try:
        tasks = [
            t for t in collaborative_manager.tasks.values()
            if t.document_id == doc_id
        ]

        return jsonify({
            'success': True,
            'tasks': [t.to_dict() for t in tasks]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/document/<doc_id>/tasks', methods=['POST'])
def api_create_document_task(doc_id):
    """Create a task for a document"""
    data = request.get_json()
    title = data.get('title')
    description = data.get('description', '')
    assigned_to = data.get('assigned_to', [])
    user_id = data.get('user_id')
    priority = data.get('priority', 'medium')

    if not all([title, user_id]):
        return jsonify({'error': 'Title and user_id required'}), 400

    try:
        task = collaborative_manager.create_task(
            doc_id, title, description, assigned_to, user_id, priority
        )

        return jsonify({
            'success': True,
            'task': task.to_dict()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/tasks/<task_id>/status', methods=['PUT'])
def api_update_task_status(task_id):
    """Update task status"""
    data = request.get_json()
    status = data.get('status')
    user_id = data.get('user_id')

    if not all([status, user_id]):
        return jsonify({'error': 'Status and user_id required'}), 400

    try:
        success = collaborative_manager.update_task_status(task_id, status, user_id)

        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Task not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/user/<user_id>/notifications', methods=['GET'])
def api_get_user_notifications(user_id):
    """Get notifications for a user"""
    try:
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        notifications = notification_manager.get_user_notifications(user_id, unread_only)

        return jsonify({
            'success': True,
            'notifications': [n.to_dict() for n in notifications]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/user/<user_id>/notifications/read', methods=['POST'])
def api_mark_notifications_read(user_id):
    """Mark notifications as read"""
    data = request.get_json()
    notification_ids = data.get('notification_ids')

    try:
        notification_manager.mark_as_read(user_id, notification_ids)
        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/user/<user_id>/dashboard', methods=['GET'])
def api_get_user_dashboard(user_id):
    """Get user's collaborative dashboard"""
    try:
        dashboard = dashboard_manager.get_user_dashboard(user_id)

        if not dashboard:
            # Create default dashboard
            dashboard = dashboard_manager.create_dashboard(user_id)

        # Get dashboard data
        dashboard_data = dashboard.to_dict()

        # Add widget data
        widgets_data = {}
        for widget in dashboard.widgets:
            if widget.is_visible:
                widget_data = dashboard_manager.get_widget_data(widget, user_id)
                widgets_data[widget.widget_id] = widget_data

        dashboard_data['widgets_data'] = widgets_data

        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/collaboration/stats', methods=['GET'])
def api_get_collaboration_stats():
    """Get comprehensive collaboration statistics"""
    try:
        document_id = request.args.get('document_id')

        if document_id:
            stats = collaborative_manager.get_collaboration_stats(document_id)
        else:
            stats = collaborative_manager.get_collaboration_stats()

        # Add additional stats
        stats.update({
            'automation_stats': automation_engine.get_engine_statistics(),
            'notification_stats': notification_manager.get_notification_statistics(),
            'annotation_stats': annotation_manager.get_annotation_statistics(),
            'dashboard_stats': dashboard_manager.get_dashboard_statistics()
        })

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/automation/rules', methods=['GET'])
def api_get_automation_rules():
    """Get automation rules"""
    try:
        active_only = request.args.get('active_only', 'false').lower() == 'true'
        rules = automation_engine.list_rules(active_only)

        return jsonify({
            'success': True,
            'rules': [r.to_dict() for r in rules]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/automation/rules', methods=['POST'])
def api_create_automation_rule():
    """Create an automation rule"""
    data = request.get_json()

    try:
        from .workflow_automation import WorkflowRule, AutomationTrigger

        rule = WorkflowRule(
            rule_id=str(uuid.uuid4()),
            name=data.get('name'),
            description=data.get('description', ''),
            trigger=AutomationTrigger(data.get('trigger')),
            trigger_config=data.get('trigger_config', {}),
            actions=data.get('actions', []),
            conditions=data.get('conditions', []),
            is_active=data.get('is_active', True),
            created_by=data.get('created_by', 'system')
        )

        rule_id = automation_engine.add_rule(rule)

        return jsonify({
            'success': True,
            'rule_id': rule_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/document/<doc_id>/activity', methods=['GET'])
def api_get_document_activity(doc_id):
    """Get activity feed for a document"""
    try:
        limit = int(request.args.get('limit', 50))
        activity = collaborative_manager.get_document_activity_feed(doc_id, limit)

        return jsonify({
            'success': True,
            'activity': [a.to_dict() for a in activity]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collaborative_bp.route('/api/websocket/token', methods=['POST'])
def api_get_websocket_token():
    """Get WebSocket authentication token"""
    data = request.get_json()
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({'error': 'User ID required'}), 400

    # Generate a simple token (in production, use proper JWT)
    token = f"ws_token_{user_id}_{int(time.time())}"

    return jsonify({
        'success': True,
        'token': token,
        'websocket_url': f"ws://localhost:8766/ws/{token}"
    })


def create_collaborative_templates():
    """Create HTML templates for collaborative features"""
    import os
    from pathlib import Path

    templates_dir = Path('search_engine/templates')

    # Main collaborative dashboard template
    collaborative_dashboard_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Collaborative Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .dashboard-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem 0;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 1.5rem;
                text-align: center;
                margin-bottom: 1rem;
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }
            .activity-item {
                padding: 1rem;
                border-bottom: 1px solid #f0f0f0;
                margin-bottom: 0.5rem;
            }
            .activity-type {
                display: inline-block;
                padding: 0.25rem 0.5rem;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 500;
                margin-left: 0.5rem;
            }
            .badge-collaborative { background: #e3f2fd; color: #1976d2; }
            .badge-workflow { background: #fff3e0; color: #f57c00; }
            .badge-annotation { background: #f3e5f5; color: #7b1fa2; }
            .badge-edit { background: #e8f5e8; color: #388e3c; }
        </style>
    </head>
    <body>
        <div class="dashboard-header">
            <div class="container">
                <h1>🚀 Collaborative Document Workflow</h1>
                <p class="mb-0">Real-time collaboration, workflows, and team management</p>
            </div>
        </div>

        <div class="container-fluid">
            <div id="loading" class="text-center">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>

            <div id="dashboard-content" style="display: none;">
                <!-- Metrics Row -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value text-primary" id="active-collaborators">0</div>
                            <div class="text-muted">Active Collaborators</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value text-success" id="active-workflows">0</div>
                            <div class="text-muted">Active Workflows</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value text-warning" id="pending-tasks">0</div>
                            <div class="text-muted">Pending Tasks</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value text-info" id="total-annotations">0</div>
                            <div class="text-muted">Total Annotations</div>
                        </div>
                    </div>
                </div>

                <!-- Main Content Row -->
                <div class="row">
                    <!-- Document Activity -->
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h5 class="mb-0">📄 Document Activity</h5>
                                <button class="btn btn-primary btn-sm" onclick="loadDocumentActivity()">Refresh</button>
                            </div>
                            <div class="card-body">
                                <div id="document-activity" style="max-height: 400px; overflow-y: auto;">
                                    <!-- Document activity will be loaded here -->
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Team Status -->
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">👥 Team Status</h5>
                            </div>
                            <div class="card-body">
                                <div id="team-status">
                                    <!-- Team status will be loaded here -->
                                </div>
                            </div>
                        </div>

                        <!-- Quick Actions -->
                        <div class="card mt-3">
                            <div class="card-header">
                                <h5 class="mb-0">⚡ Quick Actions</h5>
                            </div>
                            <div class="card-body">
                                <button class="btn btn-outline-primary btn-sm me-2" onclick="createWorkflow()">New Workflow</button>
                                <button class="btn btn-outline-secondary btn-sm" onclick="addCollaborator()">Add Collaborator</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Activity Feed -->
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h5 class="mb-0">📈 Recent Activity</h5>
                                <div class="btn-group btn-group-sm">
                                    <button class="btn btn-outline-primary active" onclick="filterActivity('all')">All</button>
                                    <button class="btn btn-outline-primary" onclick="filterActivity('comments')">Comments</button>
                                    <button class="btn btn-outline-primary" onclick="filterActivity('edits')">Edits</button>
                                    <button class="btn btn-outline-primary" onclick="filterActivity('workflows')">Workflows</button>
                                </div>
                            </div>
                            <div class="card-body">
                                <div id="activity-feed" style="max-height: 300px; overflow-y: auto;">
                                    <!-- Activity feed will be loaded here -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- WebSocket Connection Status -->
        <div id="websocket-status" class="position-fixed bottom-0 end-0 m-3 alert" style="display: none;">
            <span id="ws-status-text">WebSocket: Disconnected</span>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let websocket = null;
            let currentUserId = 'user_' + Date.now(); // In production, get from auth system

            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                initializeWebSocket();
                loadDashboardData();
                setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
            });

            function initializeWebSocket() {
                // Get WebSocket token
                fetch('/collaborative/api/websocket/token', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ user_id: currentUserId })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        connectWebSocket(data.websocket_url);
                    }
                })
                .catch(error => {
                    console.error('Error getting WebSocket token:', error);
                });
            }

            function connectWebSocket(url) {
                websocket = new WebSocket(url);

                websocket.onopen = function() {
                    updateWebSocketStatus('Connected', 'success');
                    subscribeToChannels();
                };

                websocket.onclose = function() {
                    updateWebSocketStatus('Disconnected', 'danger');
                    // Attempt to reconnect after 5 seconds
                    setTimeout(() => initializeWebSocket(), 5000);
                };

                websocket.onerror = function(error) {
                    updateWebSocketStatus('Error', 'warning');
                    console.error('WebSocket error:', error);
                };

                websocket.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };
            }

            function subscribeToChannels() {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    const subscriptionMessage = {
                        type: 'subscribe',
                        channels: ['user:' + currentUserId, 'notifications']
                    };
                    websocket.send(JSON.stringify(subscriptionMessage));
                }
            }

            function handleWebSocketMessage(data) {
                console.log('WebSocket message received:', data);

                // Handle different message types
                if (data.type === 'notification') {
                    showNotification(data.title, data.message);
                    updateNotificationCount();
                } else if (data.type === 'document_edited') {
                    loadDocumentActivity();
                } else if (data.type === 'comment_added') {
                    loadActivityFeed();
                }
            }

            function updateWebSocketStatus(status, type) {
                const statusEl = document.getElementById('websocket-status');
                const statusText = document.getElementById('ws-status-text');

                statusText.textContent = 'WebSocket: ' + status;
                statusEl.className = `position-fixed bottom-0 end-0 m-3 alert alert-${type}`;
                statusEl.style.display = 'block';
            }

            function loadDashboardData() {
                // Load collaboration stats
                fetch('/collaborative/api/collaboration/stats')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            updateMetrics(data.stats);
                        }
                    })
                    .catch(error => console.error('Error loading stats:', error));

                // Load activity feed
                loadActivityFeed();

                // Load team status
                loadTeamStatus();

                document.getElementById('loading').style.display = 'none';
                document.getElementById('dashboard-content').style.display = 'block';
            }

            function updateMetrics(stats) {
                document.getElementById('active-collaborators').textContent = stats.online_users || 0;
                document.getElementById('active-workflows').textContent = stats.total_workflows || 0;
                document.getElementById('pending-tasks').textContent = stats.total_tasks || 0;
                document.getElementById('total-annotations').textContent = stats.total_comments || 0;
            }

            function loadActivityFeed(filter = 'all') {
                // This would load activity feed data
                // For now, show placeholder
                const activityFeed = document.getElementById('activity-feed');
                activityFeed.innerHTML = '<p class="text-muted">Activity feed loading...</p>';
            }

            function loadTeamStatus() {
                // This would load team status data
                // For now, show placeholder
                const teamStatus = document.getElementById('team-status');
                teamStatus.innerHTML = '<p class="text-muted">Team status loading...</p>';
            }

            function loadDocumentActivity() {
                // This would load document activity data
                // For now, show placeholder
                const docActivity = document.getElementById('document-activity');
                docActivity.innerHTML = '<p class="text-muted">Document activity loading...</p>';
            }

            function showNotification(title, message) {
                // Simple notification display
                alert(title + ': ' + message);
            }

            function filterActivity(type) {
                // Update button states
                document.querySelectorAll('[onclick*="filterActivity"]').forEach(btn => {
                    btn.classList.remove('active');
                });
                event.target.classList.add('active');

                loadActivityFeed(type);
            }

            function createWorkflow() {
                // This would open a workflow creation modal
                alert('Workflow creation would open here');
            }

            function addCollaborator() {
                // This would open a collaborator addition modal
                alert('Collaborator addition would open here');
            }

            // Cleanup on page unload
            window.addEventListener('beforeunload', function() {
                if (websocket) {
                    websocket.close();
                }
            });
        </script>
    </body>
    </html>
    """

    # Document collaboration template
    document_collaboration_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document Collaboration</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .document-header {
                background: #f8f9fa;
                padding: 1rem;
                border-bottom: 1px solid #dee2e6;
            }
            .collaboration-panel {
                position: fixed;
                right: 0;
                top: 0;
                width: 300px;
                height: 100vh;
                background: white;
                border-left: 1px solid #dee2e6;
                padding: 1rem;
                overflow-y: auto;
            }
            .annotation {
                margin-bottom: 1rem;
                padding: 0.75rem;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            .annotation-author {
                font-weight: bold;
                color: #495057;
            }
            .annotation-time {
                font-size: 0.8rem;
                color: #6c757d;
            }
            .cursor-overlay {
                position: absolute;
                pointer-events: none;
                z-index: 1000;
            }
            .user-cursor {
                position: absolute;
                width: 2px;
                height: 20px;
                background: #007bff;
            }
            .user-cursor::before {
                content: '';
                position: absolute;
                top: -8px;
                left: -6px;
                background: #007bff;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10px;
                white-space: nowrap;
            }
        </style>
    </head>
    <body>
        <div class="document-header">
            <div class="container-fluid">
                <div class="row">
                    <div class="col-md-8">
                        <h4 id="document-title">Loading Document...</h4>
                        <div id="document-info">
                            <!-- Document info will be loaded here -->
                        </div>
                    </div>
                    <div class="col-md-4 text-end">
                        <button class="btn btn-primary btn-sm" onclick="addComment()">💬 Add Comment</button>
                        <button class="btn btn-secondary btn-sm" onclick="showWorkflows()">📋 Workflows</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="container-fluid">
            <div class="row">
                <!-- Main Document Area -->
                <div class="col-md-8">
                    <div id="document-viewer" style="min-height: 600px; background: white; padding: 2rem;">
                        <!-- Document content would be rendered here -->
                        <div class="text-center text-muted">
                            <h5>Document Viewer</h5>
                            <p>Document rendering would be implemented based on document type</p>
                        </div>
                    </div>

                    <!-- Cursor overlay for real-time collaboration -->
                    <div id="cursor-overlay" class="cursor-overlay">
                        <!-- User cursors will be displayed here -->
                    </div>
                </div>

                <!-- Collaboration Panel -->
                <div class="col-md-4">
                    <div class="collaboration-panel">
                        <!-- Online Collaborators -->
                        <div class="card mb-3">
                            <div class="card-header">
                                <h6 class="mb-0">👥 Online Collaborators</h6>
                            </div>
                            <div class="card-body">
                                <div id="online-collaborators">
                                    <!-- Online users will be listed here -->
                                </div>
                            </div>
                        </div>

                        <!-- Document Comments/Annotations -->
                        <div class="card mb-3">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h6 class="mb-0">💬 Comments & Annotations</h6>
                                <button class="btn btn-primary btn-sm" onclick="addComment()">Add</button>
                            </div>
                            <div class="card-body">
                                <div id="document-comments">
                                    <!-- Comments will be loaded here -->
                                </div>
                            </div>
                        </div>

                        <!-- Active Workflows -->
                        <div class="card">
                            <div class="card-header">
                                <h6 class="mb-0">📋 Active Workflows</h6>
                            </div>
                            <div class="card-body">
                                <div id="active-workflows">
                                    <!-- Workflows will be loaded here -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Comment Modal -->
        <div class="modal fade" id="commentModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Add Comment</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="commentForm">
                            <div class="mb-3">
                                <label for="commentContent" class="form-label">Comment</label>
                                <textarea class="form-control" id="commentContent" rows="3" required></textarea>
                            </div>
                            <div class="mb-3">
                                <label for="commentType" class="form-label">Type</label>
                                <select class="form-control" id="commentType">
                                    <option value="comment">Comment</option>
                                    <option value="question">Question</option>
                                    <option value="suggestion">Suggestion</option>
                                    <option value="todo">To Do</option>
                                </select>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="submitComment()">Add Comment</button>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            const docId = window.location.pathname.split('/').pop();
            let currentUserId = 'user_' + Date.now();

            document.addEventListener('DOMContentLoaded', function() {
                // Initialize user if not exists
                initializeUser();

                // Load document collaboration data
                loadDocumentCollaboration();

                // Initialize real-time editing (if supported)
                initializeCollaborativeEditing();

                // Set up periodic refresh
                setInterval(loadDocumentCollaboration, 30000);
            });

            function initializeUser() {
                // Initialize user in collaborative system
                fetch('/collaborative/api/initialize_user', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        user_id: currentUserId,
                        username: 'User ' + currentUserId.substring(5),
                        email: currentUserId + '@example.com',
                        role: 'editor'
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('User initialized:', data.user);
                    }
                })
                .catch(error => console.error('Error initializing user:', error));
            }

            function loadDocumentCollaboration() {
                // Load collaborators
                loadCollaborators();

                // Load comments/annotations
                loadComments();

                // Load workflows
                loadWorkflows();

                // Load document info
                loadDocumentInfo();
            }

            function loadCollaborators() {
                fetch(`/collaborative/api/document/${docId}/collaborators`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            displayCollaborators(data.collaborators);
                        }
                    })
                    .catch(error => console.error('Error loading collaborators:', error));
            }

            function displayCollaborators(collaborators) {
                const container = document.getElementById('online-collaborators');

                if (collaborators.length === 0) {
                    container.innerHTML = '<p class="text-muted">No collaborators yet</p>';
                    return;
                }

                const onlineUsers = collaborators.filter(c => c.is_online);
                const html = onlineUsers.map(collaborator => `
                    <div class="d-flex align-items-center mb-2">
                        <div class="badge bg-${collaborator.is_online ? 'success' : 'secondary'} me-2">
                            ${collaborator.is_online ? '●' : '○'}
                        </div>
                        <div>
                            <div class="fw-bold">${collaborator.username}</div>
                            <small class="text-muted">${collaborator.role}</small>
                        </div>
                    </div>
                `).join('');

                container.innerHTML = html;
            }

            function loadComments() {
                fetch(`/collaborative/api/document/${docId}/annotations`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            displayComments(data.annotations);
                        }
                    })
                    .catch(error => console.error('Error loading comments:', error));
            }

            function displayComments(annotations) {
                const container = document.getElementById('document-comments');

                if (annotations.length === 0) {
                    container.innerHTML = '<p class="text-muted">No comments yet</p>';
                    return;
                }

                const html = annotations.map(annotation => `
                    <div class="annotation">
                        <div class="annotation-author">${annotation.author_name}</div>
                        <div class="annotation-time">${new Date(annotation.created_at).toLocaleString()}</div>
                        <div class="mt-1">${annotation.content}</div>
                        <small class="text-muted">${annotation.type}</small>
                    </div>
                `).join('');

                container.innerHTML = html;
            }

            function loadWorkflows() {
                fetch(`/collaborative/api/document/${docId}/workflows`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            displayWorkflows(data.workflows);
                        }
                    })
                    .catch(error => console.error('Error loading workflows:', error));
            }

            function displayWorkflows(workflows) {
                const container = document.getElementById('active-workflows');

                if (workflows.length === 0) {
                    container.innerHTML = '<p class="text-muted">No active workflows</p>';
                    return;
                }

                const html = workflows.map(workflow => `
                    <div class="alert alert-${getWorkflowAlertClass(workflow.status)}">
                        <strong>${workflow.title}</strong><br>
                        <small>Status: ${workflow.status} | Created: ${new Date(workflow.created_at).toLocaleDateString()}</small>
                    </div>
                `).join('');

                container.innerHTML = html;
            }

            function getWorkflowAlertClass(status) {
                const statusMap = {
                    'draft': 'secondary',
                    'in_review': 'warning',
                    'approved': 'success',
                    'rejected': 'danger'
                };
                return statusMap[status] || 'secondary';
            }

            function loadDocumentInfo() {
                // Load basic document information
                // This would integrate with the main search system
                document.getElementById('document-title').textContent = `Document: ${docId}`;
            }

            function addComment() {
                // Show comment modal
                const modal = new bootstrap.Modal(document.getElementById('commentModal'));
                modal.show();
            }

            function submitComment() {
                const content = document.getElementById('commentContent').value;
                const type = document.getElementById('commentType').value;

                if (!content.trim()) {
                    alert('Please enter a comment');
                    return;
                }

                // Submit comment
                fetch(`/collaborative/api/document/${docId}/annotations`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        type: type,
                        content: content,
                        user_id: currentUserId,
                        position: { x: 100, y: 100 } // Default position
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Close modal and reload comments
                        bootstrap.Modal.getInstance(document.getElementById('commentModal')).hide();
                        document.getElementById('commentForm').reset();
                        loadComments();
                    } else {
                        alert('Error adding comment: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error submitting comment:', error);
                    alert('Error submitting comment');
                });
            }

            function showWorkflows() {
                // This would open workflows modal
                alert('Workflows management would open here');
            }

            function initializeCollaborativeEditing() {
                // Initialize real-time collaborative editing features
                // This would include cursor tracking, operational transforms, etc.

                console.log('Collaborative editing initialized for document:', docId);
            }

            // Track cursor position for collaborative editing
            let cursorTimeout;
            document.addEventListener('mousemove', function(e) {
                // Send cursor position updates
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    clearTimeout(cursorTimeout);
                    cursorTimeout = setTimeout(() => {
                        const message = {
                            type: 'cursor_position',
                            position: {
                                x: e.clientX,
                                y: e.clientY,
                                document_id: docId
                            }
                        };
                        websocket.send(JSON.stringify(message));
                    }, 100); // Throttle cursor updates
                }
            });
        </script>
    </body>
    </html>
    """

    # Write templates to files
    (templates_dir / 'collaborative_dashboard.html').write_text(collaborative_dashboard_html)
    (templates_dir / 'document_collaboration.html').write_text(document_collaboration_html)

    print("Collaborative templates created")


def initialize_collaborative_system():
    """Initialize the complete collaborative workflow system"""
    print("Initializing Collaborative Document Workflow System...")

    # Initialize all managers
    print("✓ Collaborative workflow manager initialized")

    # Start automation engine
    automation_engine.start_engine()
    print("✓ Workflow automation engine started")

    # Create templates
    create_collaborative_templates()
    print("✓ Web interface templates created")

    print("🎉 Collaborative Document Workflow System initialized successfully!")


# Initialize when module is imported
initialize_collaborative_system()