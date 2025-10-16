"""
Web Interface for Version Control Management
Provides a comprehensive web interface for all version control operations
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from datetime import datetime
import json

from .version_api import version_api
from .version_control import version_manager
from .lifecycle_manager import advanced_lifecycle_manager
from .approval_workflow import approval_workflow_manager
from .rollback_manager import rollback_manager
from .relationship_analyzer import impact_analyzer, relationship_analyzer
from .integration import integration_manager

# Create blueprint
version_web = Blueprint('version_web', __name__)

@version_web.route('/documents/<doc_id>/versions')
def document_versions(doc_id):
    """Display all versions for a document"""
    try:
        versions = version_manager.db.get_document_versions(doc_id)

        # Get document metadata
        enhanced_metadata = integration_manager.metadata_integrator.get_enhanced_document_metadata(doc_id)

        return render_template('document_versions.html',
                             doc_id=doc_id,
                             document=enhanced_metadata,
                             versions=versions)
    except Exception as e:
        flash(f'Error loading document versions: {str(e)}', 'error')
        return redirect(url_for('search'))

@version_web.route('/versions/<version_id>')
def version_details(version_id):
    """Display details for a specific version"""
    try:
        version = version_manager.db.get_version(version_id)
        if not version:
            flash('Version not found', 'error')
            return redirect(url_for('search'))

        # Get related versions for comparison
        related_versions = version_manager.db.get_document_versions(version.doc_id)

        return render_template('version_details.html',
                             version=version,
                             related_versions=related_versions)
    except Exception as e:
        flash(f'Error loading version details: {str(e)}', 'error')
        return redirect(url_for('search'))

@version_web.route('/versions/compare', methods=['GET', 'POST'])
def compare_versions():
    """Compare two document versions"""
    if request.method == 'POST':
        version1_id = request.form.get('version1_id')
        version2_id = request.form.get('version2_id')

        if not version1_id or not version2_id:
            flash('Please select two versions to compare', 'error')
            return redirect(url_for('compare_versions'))

        try:
            comparison = document_comparator.compare_documents(version1_id, version2_id)

            return render_template('version_comparison.html',
                                 comparison=comparison,
                                 version1_id=version1_id,
                                 version2_id=version2_id)
        except Exception as e:
            flash(f'Error comparing versions: {str(e)}', 'error')
            return redirect(url_for('compare_versions'))

    # GET request - show version selection form
    # Get all documents with versions for selection
    catalog_path = Path(__file__).parent.parent / "document_catalog.json"
    documents = []

    if catalog_path.exists():
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)

        for doc_id, doc_data in catalog.items():
            versions = version_manager.db.get_document_versions(doc_id)
            if len(versions) > 1:  # Only include documents with multiple versions
                documents.append({
                    'doc_id': doc_id,
                    'file_name': doc_data.get('file_name', doc_id),
                    'versions': versions
                })

    return render_template('version_compare_form.html', documents=documents)

@version_web.route('/documents/<doc_id>/lifecycle')
def document_lifecycle(doc_id):
    """Display lifecycle information for a document"""
    try:
        lifecycle_info = advanced_lifecycle_manager.get_document_lifecycle_info(doc_id)

        if not lifecycle_info:
            flash('Document not found', 'error')
            return redirect(url_for('search'))

        return render_template('document_lifecycle.html',
                             doc_id=doc_id,
                             lifecycle_info=lifecycle_info)
    except Exception as e:
        flash(f'Error loading lifecycle information: {str(e)}', 'error')
        return redirect(url_for('search'))

@version_web.route('/versions/<version_id>/lifecycle/transition', methods=['POST'])
def transition_lifecycle(version_id):
    """Transition a version to a new lifecycle status"""
    try:
        new_status = request.form.get('new_status')
        user_id = request.form.get('user_id', 'system')
        reason = request.form.get('reason')

        if not new_status:
            flash('New status is required', 'error')
            return redirect(url_for('version_details', version_id=version_id))

        success = advanced_lifecycle_manager.transition_document(
            version_id, new_status, user_id, reason
        )

        if success:
            flash(f'Successfully transitioned to {new_status}', 'success')
        else:
            flash('Failed to transition lifecycle status', 'error')

        return redirect(url_for('version_details', version_id=version_id))

    except Exception as e:
        flash(f'Error transitioning lifecycle: {str(e)}', 'error')
        return redirect(url_for('version_details', version_id=version_id))

@version_web.route('/workflows')
def list_workflows():
    """List all active workflows"""
    try:
        # Get all workflow instances
        workflow_instances = []
        for instance_id, instance in advanced_lifecycle_manager.workflow_instances.items():
            if instance.status == 'active':
                status = advanced_lifecycle_manager.get_workflow_status(instance_id)
                if status:
                    workflow_instances.append(status)

        return render_template('workflow_list.html',
                             workflows=workflow_instances)
    except Exception as e:
        flash(f'Error loading workflows: {str(e)}', 'error')
        return redirect(url_for('search'))

@version_web.route('/workflows/<instance_id>')
def workflow_details(instance_id):
    """Display workflow details"""
    try:
        status = advanced_lifecycle_manager.get_workflow_status(instance_id)
        if not status:
            flash('Workflow not found', 'error')
            return redirect(url_for('list_workflows'))

        return render_template('workflow_details.html',
                             workflow_status=status)
    except Exception as e:
        flash(f'Error loading workflow details: {str(e)}', 'error')
        return redirect(url_for('list_workflows'))

@version_web.route('/approvals')
def approval_dashboard():
    """Display approval dashboard"""
    try:
        # Get pending approvals (would need user authentication)
        user_id = request.args.get('user_id', 'current_user')
        pending_approvals = approval_workflow_manager.get_pending_approvals(user_id)

        # Get recent change requests
        all_requests = version_manager.db.get_change_requests()
        recent_requests = all_requests[:20]  # Last 20 requests

        return render_template('approval_dashboard.html',
                             pending_approvals=pending_approvals,
                             recent_requests=recent_requests)
    except Exception as e:
        flash(f'Error loading approval dashboard: {str(e)}', 'error')
        return redirect(url_for('search'))

@version_web.route('/approvals/<request_id>/respond', methods=['POST'])
def respond_to_approval_request(request_id):
    """Respond to an approval request"""
    try:
        approver_id = request.form.get('approver_id')
        decision = request.form.get('decision')  # approved, rejected, delegated
        notes = request.form.get('notes')

        if not approver_id or not decision:
            flash('Approver ID and decision are required', 'error')
            return redirect(url_for('approval_dashboard'))

        success = approval_workflow_manager.respond_to_approval(
            request_id, approver_id, decision, notes
        )

        if success:
            flash(f'Approval {decision} successfully', 'success')
        else:
            flash('Failed to process approval response', 'error')

        return redirect(url_for('approval_dashboard'))

    except Exception as e:
        flash(f'Error processing approval: {str(e)}', 'error')
        return redirect(url_for('approval_dashboard'))

@version_web.route('/documents/<doc_id>/rollback')
def rollback_document(doc_id):
    """Display rollback options for a document"""
    try:
        # Get rollback history
        history = rollback_manager.get_rollback_history(doc_id)

        # Get available versions for rollback
        versions = version_manager.db.get_document_versions(doc_id)

        return render_template('document_rollback.html',
                             doc_id=doc_id,
                             rollback_history=history,
                             available_versions=versions)
    except Exception as e:
        flash(f'Error loading rollback information: {str(e)}', 'error')
        return redirect(url_for('document_versions', doc_id=doc_id))

@version_web.route('/documents/<doc_id>/rollback/create-plan', methods=['POST'])
def create_rollback_plan(doc_id):
    """Create a rollback plan"""
    try:
        target_version_id = request.form.get('target_version_id')
        rollback_type = request.form.get('rollback_type', 'full')
        created_by = request.form.get('created_by', 'system')

        if not target_version_id:
            flash('Target version is required', 'error')
            return redirect(url_for('rollback_document', doc_id=doc_id))

        plan_id = rollback_manager.create_rollback_plan(
            doc_id, target_version_id, rollback_type, created_by
        )

        if plan_id:
            flash('Rollback plan created successfully', 'success')
            return redirect(url_for('execute_rollback', plan_id=plan_id))
        else:
            flash('Failed to create rollback plan', 'error')
            return redirect(url_for('rollback_document', doc_id=doc_id))

    except Exception as e:
        flash(f'Error creating rollback plan: {str(e)}', 'error')
        return redirect(url_for('rollback_document', doc_id=doc_id))

@version_web.route('/rollback/<plan_id>/execute', methods=['POST'])
def execute_rollback_plan(plan_id):
    """Execute a rollback plan"""
    try:
        executed_by = request.form.get('executed_by', 'system')
        force = request.form.get('force', 'false').lower() == 'true'

        result = rollback_manager.execute_rollback(plan_id, executed_by, force)

        if result == 'success':
            flash('Rollback executed successfully', 'success')
        else:
            flash(f'Rollback failed: {result}', 'error')

        return redirect(url_for('rollback_result', plan_id=plan_id))

    except Exception as e:
        flash(f'Error executing rollback: {str(e)}', 'error')
        return redirect(url_for('list_rollbacks'))

@version_web.route('/rollback/<plan_id>/preview')
def preview_rollback(plan_id):
    """Preview a rollback plan"""
    try:
        plan = rollback_manager.rollback_plans.get(plan_id)
        if not plan:
            flash('Rollback plan not found', 'error')
            return redirect(url_for('list_rollbacks'))

        preview = rollback_manager.get_rollback_preview(plan.doc_id, plan.target_version_id)

        return render_template('rollback_preview.html',
                             plan=plan,
                             preview=preview)
    except Exception as e:
        flash(f'Error loading rollback preview: {str(e)}', 'error')
        return redirect(url_for('list_rollbacks'))

@version_web.route('/rollbacks')
def list_rollbacks():
    """List all rollback plans and results"""
    try:
        # Get all rollback plans
        rollback_plans = []
        for plan_id, plan in rollback_manager.rollback_plans.items():
            result = rollback_manager.rollback_results.get(plan_id)
            rollback_plans.append({
                'plan': plan,
                'result': result
            })

        # Sort by creation date (newest first)
        rollback_plans.sort(key=lambda x: x['plan'].created_at, reverse=True)

        return render_template('rollback_list.html',
                             rollback_plans=rollback_plans)
    except Exception as e:
        flash(f'Error loading rollbacks: {str(e)}', 'error')
        return redirect(url_for('search'))

@version_web.route('/documents/<doc_id>/relationships')
def document_relationships(doc_id):
    """Display document relationships"""
    try:
        relationships = version_manager.db.get_document_relationships(doc_id)

        # Build relationship network
        related_ids = [rel.target_doc_id for rel in relationships]
        related_ids.append(doc_id)
        network = relationship_analyzer.build_relationship_network(related_ids)

        return render_template('document_relationships.html',
                             doc_id=doc_id,
                             relationships=relationships,
                             network=network)
    except Exception as e:
        flash(f'Error loading relationships: {str(e)}', 'error')
        return redirect(url_for('document_versions', doc_id=doc_id))

@version_web.route('/documents/<doc_id>/impact-analysis', methods=['GET', 'POST'])
def impact_analysis(doc_id):
    """Display impact analysis for a document"""
    if request.method == 'POST':
        version_id = request.form.get('version_id')
        change_description = request.form.get('change_description')

        if not version_id:
            flash('Version ID is required', 'error')
            return redirect(url_for('impact_analysis', doc_id=doc_id))

        try:
            assessment = impact_analyzer.analyze_change_impact(
                doc_id, version_id, change_description
            )

            if assessment:
                return render_template('impact_analysis_result.html',
                                     doc_id=doc_id,
                                     assessment=assessment,
                                     summary=impact_analyzer.get_impact_summary(assessment))
            else:
                flash('Failed to analyze impact', 'error')

        except Exception as e:
            flash(f'Error analyzing impact: {str(e)}', 'error')

    # GET request - show impact analysis form
    versions = version_manager.db.get_document_versions(doc_id)

    return render_template('impact_analysis_form.html',
                         doc_id=doc_id,
                         versions=versions)

@version_web.route('/documents/<doc_id>/dependency-graph')
def dependency_graph(doc_id):
    """Display dependency graph for a document"""
    try:
        max_depth = int(request.args.get('max_depth', 3))
        graph = dependency_resolver.get_dependency_graph(doc_id, max_depth)

        return render_template('dependency_graph.html',
                             doc_id=doc_id,
                             graph=graph)
    except Exception as e:
        flash(f'Error loading dependency graph: {str(e)}', 'error')
        return redirect(url_for('document_relationships', doc_id=doc_id))

@version_web.route('/search/versions')
def search_versions():
    """Search interface for document versions"""
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

        return render_template('version_search_results.html',
                             query=query,
                             filters=filters,
                             results=results)
    except Exception as e:
        flash(f'Error searching versions: {str(e)}', 'error')
        return redirect(url_for('search'))

@version_web.route('/system/status')
def system_status():
    """Display system status"""
    try:
        status = integration_manager.get_system_status()

        # Get additional status information
        version_stats = {
            'total_documents': 0,
            'total_versions': 0,
            'active_workflows': len([i for i in advanced_lifecycle_manager.workflow_instances.values() if i.status == 'active']),
            'pending_approvals': 0  # Would need to count from approval system
        }

        return render_template('system_status.html',
                             status=status,
                             version_stats=version_stats)
    except Exception as e:
        flash(f'Error loading system status: {str(e)}', 'error')
        return redirect(url_for('search'))

@version_web.route('/documents/<doc_id>/create-version', methods=['POST'])
def create_new_version(doc_id):
    """Create a new version for a document"""
    try:
        file_path = request.form.get('file_path')
        change_type = request.form.get('change_type', 'auto')
        change_description = request.form.get('change_description')
        created_by = request.form.get('created_by', 'system')

        if not file_path:
            flash('File path is required', 'error')
            return redirect(url_for('document_versions', doc_id=doc_id))

        version = version_manager.create_version(
            doc_id=doc_id,
            file_path=file_path,
            change_type=change_type,
            change_description=change_description,
            created_by=created_by
        )

        if version:
            flash(f'Created new version {version.version_number}', 'success')
        else:
            flash('Failed to create new version', 'error')

        return redirect(url_for('document_versions', doc_id=doc_id))

    except Exception as e:
        flash(f'Error creating version: {str(e)}', 'error')
        return redirect(url_for('document_versions', doc_id=doc_id))

# Template filters and utilities
@version_web.app_template_filter('datetime')
def format_datetime(value):
    """Format datetime for display"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime('%Y-%m-%d %H:%M:%S')

@version_web.app_template_filter('timesince')
def timesince(value):
    """Show time since datetime"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)

    now = datetime.now()
    diff = now - value

    if diff.days > 0:
        return f'{diff.days} days ago'
    elif diff.seconds > 3600:
        return f'{diff.seconds // 3600} hours ago'
    elif diff.seconds > 60:
        return f'{diff.seconds // 60} minutes ago'
    else:
        return 'Just now'

@version_web.app_template_filter('status_badge')
def status_badge_class(status):
    """Get CSS class for status badges"""
    classes = {
        'draft': 'badge-secondary',
        'under_review': 'badge-warning',
        'approved': 'badge-info',
        'published': 'badge-success',
        'deprecated': 'badge-warning',
        'archived': 'badge-dark',
        'retired': 'badge-danger',
        'pending': 'badge-warning',
        'completed': 'badge-success',
        'failed': 'badge-danger'
    }
    return classes.get(status, 'badge-secondary')

@version_web.app_template_filter('risk_badge')
def risk_badge_class(risk):
    """Get CSS class for risk badges"""
    classes = {
        'low': 'badge-success',
        'medium': 'badge-warning',
        'high': 'badge-danger',
        'critical': 'badge-dark'
    }
    return classes.get(risk, 'badge-secondary')

# Register blueprint function for Flask app integration
def register_version_web(app):
    """Register the version control web interface with Flask app"""
    app.register_blueprint(version_web, url_prefix='/version-control')

    # Register template filters
    app.add_template_filter(format_datetime)
    app.add_template_filter(timesince)
    app.add_template_filter(status_badge_class)
    app.add_template_filter(risk_badge_class)