# 🚀 Collaborative Document Workflow System

A comprehensive real-time collaborative document management system that integrates seamlessly with existing search, version control, compliance, and summarization systems.

## 📋 Overview

This system provides:

- **Real-time collaborative editing** with operational transforms
- **Document review and approval workflows** with role-based permissions
- **Commenting and annotation system** with threaded discussions
- **Team collaboration features** with notifications
- **Workflow automation and task management**
- **Integration with existing systems** (search, version control, compliance)
- **Collaborative dashboard and reporting**
- **WebSocket support** for real-time features

## 🏗️ System Architecture

### Core Components

1. **Collaborative Workflow Manager** (`collaborative_workflow.py`)
   - User management and presence tracking
   - Document collaboration coordination
   - Event management and activity feeds

2. **Real-time Collaborative Editor** (`collaborative_editor.py`)
   - WebSocket-based real-time editing
   - Operational transform engine
   - Document state management

3. **Annotation System** (`annotation_system.py`)
   - Threaded commenting system
   - Document annotations with positioning
   - Rich content support (mentions, reactions, attachments)

4. **Notification System** (`notification_system.py`)
   - Multi-channel notifications (in-app, email, push)
   - User preferences and quiet hours
   - Activity feeds and real-time updates

5. **Workflow Automation Engine** (`workflow_automation.py`)
   - Intelligent automation rules
   - Scheduled tasks and triggers
   - Task templates and management

6. **Collaborative Dashboard** (`collaborative_dashboard.py`)
   - Real-time widgets and metrics
   - Customizable dashboards
   - Analytics and reporting

7. **WebSocket Manager** (`websocket_manager.py`)
   - Real-time communication
   - Channel-based subscriptions
   - Connection management

8. **Integration Layer** (`collaborative_integration.py`)
   - Search system integration
   - Version control integration
   - Compliance system integration

## 🚀 Quick Start

### 1. Installation

```python
# Import the collaborative system
from search_engine.collaborative_workflow import collaborative_manager
from search_engine.collaborative_integration import integration_manager

# Initialize the system
integration_manager.initialize_all_integrations()
```

### 2. Basic Usage

```python
# Add users to the system
collaborative_manager.add_user(
    user_id="user123",
    username="John Doe",
    email="john@example.com",
    role=UserRole.EDITOR
)

# Add collaborators to a document
collaborative_manager.add_document_collaborator(
    document_id="doc456",
    user_id="user123",
    role=UserRole.EDITOR
)

# Create a workflow
workflow = collaborative_manager.create_workflow(
    document_id="doc456",
    title="Document Review Process",
    description="Review and approval workflow",
    created_by="user123",
    reviewers=["reviewer1", "reviewer2"],
    approvers=["approver1"]
)

# Add comments
comment = collaborative_manager.add_comment(
    document_id="doc456",
    user_id="user123",
    content="Please review this section",
    position={"x": 100, "y": 200}
)

# Create tasks
task = collaborative_manager.create_task(
    document_id="doc456",
    title="Review document content",
    description="Please review for accuracy",
    assigned_to=["reviewer1"],
    created_by="user123",
    priority="high"
)
```

## 📚 API Reference

### User Management

```python
# Add user
user = collaborative_manager.add_user(user_id, username, email, role)

# Update user presence
collaborative_manager.update_user_presence(user_id, is_online=True, current_document=doc_id)

# Get document collaborators
collaborators = collaborative_manager.get_document_collaborators(document_id)
```

### Document Collaboration

```python
# Add collaborator to document
collaborative_manager.add_document_collaborator(document_id, user_id, role)

# Remove collaborator
collaborative_manager.remove_document_collaborator(document_id, user_id)
```

### Comments and Annotations

```python
# Add comment
comment = annotation_manager.create_annotation(
    document_id, AnnotationType.COMMENT, content, position, author_id, author_name
)

# Reply to comment
reply = annotation_manager.reply_to_annotation(parent_id, content, author_id, author_name)

# Resolve comment
annotation_manager.resolve_annotation(annotation_id, user_id)

# Add reaction
annotation_manager.add_reaction(annotation_id, "👍", user_id)
```

### Workflow Management

```python
# Create workflow
workflow = collaborative_manager.create_workflow(
    document_id, title, description, created_by, reviewers, approvers
)

# Create task
task = collaborative_manager.create_task(
    document_id, title, description, assigned_to, created_by, priority, due_date
)

# Update task status
collaborative_manager.update_task_status(task_id, "completed", user_id)
```

### Real-time Features

```python
# Initialize collaborative editing for document
editor_manager.initialize_document_for_collaboration(document_id, initial_content)

# Start WebSocket server
websocket_manager.start_server()

# Broadcast events
websocket_manager.send_collaborative_event(event)
```

## 🔧 Configuration

### Environment Variables

```bash
# WebSocket Configuration
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8766

# Email Notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Automation Settings
AUTOMATION_SCHEDULER_INTERVAL=60  # seconds
AUTOMATION_MAX_RETRIES=3

# Cache Settings
CACHE_TTL=300  # seconds
MAX_CACHE_ENTRIES=1000
```

### Default Automation Rules

The system includes several default automation rules:

1. **Overdue Workflow Reminder** - Sends reminders for overdue workflows
2. **Auto-approve Inactive Workflows** - Approves workflows after inactivity period
3. **Welcome New Collaborators** - Sends welcome messages to new collaborators

## 🌐 Web Interface

### Routes

- `/collaborative/dashboard` - Main collaborative dashboard
- `/collaborative/document/<doc_id>` - Document collaboration interface
- `/collaborative/api/*` - REST API endpoints

### JavaScript Integration

```javascript
// Initialize WebSocket connection
const ws = new WebSocket('ws://localhost:8766/ws/your-token');

// Subscribe to channels
ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['document:doc123', 'user:user123']
}));

// Handle real-time updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'document_edited') {
        // Handle document edit
        updateDocumentContent(data);
    } else if (data.type === 'comment_added') {
        // Handle new comment
        addCommentToUI(data);
    }
};

// Send collaborative edit
function sendEdit(operation) {
    ws.send(JSON.stringify({
        type: 'collaborative_edit',
        operation: operation
    }));
}
```

## 📊 Dashboard Widgets

### Available Widget Types

1. **Collaboration Overview** - Active collaborators, recent activity
2. **Active Workflows** - Workflow status and progress
3. **Document Activity** - Recent document interactions
4. **Team Status** - Online team members and status
5. **Notification Summary** - Unread notifications and alerts
6. **Task Summary** - Pending and overdue tasks
7. **Compliance Status** - Document compliance metrics
8. **Activity Feed** - Real-time activity stream

### Creating Custom Widgets

```python
widget = DashboardWidget(
    widget_id=str(uuid.uuid4()),
    title='Custom Metric',
    type='metric',
    position={'x': 0, 'y': 0, 'width': 3, 'height': 2},
    config={'metric_type': 'collaboration_score'},
    refresh_interval=60
)
```

## 🤖 Automation Rules

### Creating Custom Rules

```python
rule = WorkflowRule(
    rule_id=str(uuid.uuid4()),
    name="Custom Approval Rule",
    description="Auto-approve after 3 days",
    trigger=AutomationTrigger.CONDITION_BASED,
    conditions=[
        {
            'type': 'time_range',
            'min_days_in_status': 3
        }
    ],
    actions=[
        {
            'type': 'update_workflow_status',
            'status': 'approved'
        }
    ]
)

automation_engine.add_rule(rule)
```

### Available Triggers

- **TIME_BASED** - Daily schedules or intervals
- **EVENT_BASED** - Document events (edit, comment, etc.)
- **CONDITION_BASED** - When conditions are met
- **MANUAL** - Triggered manually

### Available Actions

- **SEND_NOTIFICATION** - Send notifications to users
- **CREATE_TASK** - Create new workflow tasks
- **UPDATE_WORKFLOW_STATUS** - Change workflow status
- **SEND_EMAIL** - Send email notifications
- **RUN_SCRIPT** - Execute custom scripts

## 🔒 Security & Permissions

### User Roles

- **VIEWER** - Read-only access
- **EDITOR** - Can edit documents
- **REVIEWER** - Can review and comment
- **APPROVER** - Can approve workflows
- **ADMIN** - Full system access

### Permission Checks

```python
# Check if user can perform action
user = collaborative_manager.users[user_id]
document_permissions = user.permissions.get(document_id, '')

can_edit = document_permissions in ['editor', 'approver', 'admin']
can_approve = document_permissions in ['approver', 'admin']
```

## 📈 Monitoring & Analytics

### Key Metrics

```python
# Get system statistics
stats = collaborative_manager.get_collaboration_stats()

# Get automation statistics
auto_stats = automation_engine.get_engine_statistics()

# Get dashboard statistics
dash_stats = dashboard_manager.get_dashboard_statistics()

# Get WebSocket connection stats
ws_stats = websocket_manager.get_connection_stats()
```

### Performance Monitoring

- **Response Times** - Track API response times
- **User Activity** - Monitor active users and sessions
- **System Health** - Monitor background processes
- **Error Rates** - Track and alert on errors

## 🔧 Troubleshooting

### Common Issues

1. **WebSocket Connection Issues**
   ```python
   # Check connection status
   stats = websocket_manager.get_connection_stats()
   print(f"Active connections: {stats['total_clients']}")
   ```

2. **Automation Rules Not Triggering**
   ```python
   # Check rule status
   rules = automation_engine.list_rules(active_only=True)
   for rule in rules:
       print(f"Rule: {rule.name}, Last triggered: {rule.last_triggered}")
   ```

3. **Notifications Not Sending**
   ```python
   # Check notification system
   stats = notification_manager.get_notification_statistics()
   print(f"Total notifications: {stats['total_notifications']}")
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for specific components
logger = logging.getLogger('search_engine.collaborative_workflow')
logger.setLevel(logging.DEBUG)
```

## 🚀 Deployment

### Production Deployment

1. **Install Dependencies**
   ```bash
   pip install websockets flask python-multipart
   ```

2. **Configure Environment**
   ```bash
   export WEBSOCKET_HOST=0.0.0.0
   export WEBSOCKET_PORT=8766
   export SMTP_USERNAME=your-email@gmail.com
   export SMTP_PASSWORD=your-app-password
   ```

3. **Start Services**
   ```python
   # Initialize integration
   integration_manager.initialize_all_integrations()

   # Start web application
   app.run(host='0.0.0.0', port=5000)
   ```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY search_engine/ ./search_engine/
COPY document_catalog.json .

EXPOSE 5000 8766

CMD ["python", "-m", "search_engine.collaborative_integration"]
```

## 📋 API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/collaborative/api/initialize_user` | Initialize user in system |
| GET | `/collaborative/api/document/{id}/collaborators` | Get document collaborators |
| POST | `/collaborative/api/document/{id}/collaborators` | Add document collaborator |
| GET | `/collaborative/api/document/{id}/annotations` | Get document annotations |
| POST | `/collaborative/api/document/{id}/annotations` | Add document annotation |
| GET | `/collaborative/api/document/{id}/workflows` | Get document workflows |
| POST | `/collaborative/api/document/{id}/workflows` | Create document workflow |
| GET | `/collaborative/api/user/{id}/notifications` | Get user notifications |
| POST | `/collaborative/api/user/{id}/notifications/read` | Mark notifications as read |

### WebSocket Events

#### Client → Server

```javascript
// Subscribe to channels
{
    "type": "subscribe",
    "channels": ["document:doc123", "user:user123"]
}

// Send collaborative edit
{
    "type": "collaborative_edit",
    "operation": {
        "type": "insert",
        "position": 100,
        "content": "new text"
    }
}

// Update cursor position
{
    "type": "cursor_position",
    "position": { "x": 150, "y": 200 }
}
```

#### Server → Client

```javascript
// Document edited
{
    "type": "document_edited",
    "document_id": "doc123",
    "user_id": "user456",
    "operation": { /* operation details */ }
}

// New comment
{
    "type": "comment_added",
    "document_id": "doc123",
    "annotation": { /* annotation details */ }
}

// Notification
{
    "type": "notification",
    "title": "New Comment",
    "message": "You have a new comment"
}
```

## 🔗 Integration Examples

### With Existing Search System

```python
# Enhanced search with collaborative context
from search_engine.collaborative_integration import integration_manager

# Search with collaborative filters
results = integration_manager.search_integrator.search_with_collaboration_context(
    query="invoice processing",
    user_id="user123",
    collaboration_filters={"has_collaborators": True}
)
```

### With Version Control

```python
# Create collaborative version
version_info = integration_manager.version_integrator.create_collaborative_version(
    doc_id="doc123",
    user_id="user456",
    change_type="major",
    description="Added collaborative review section"
)
```

### With Compliance System

```python
# Check collaborative compliance
compliance = integration_manager.compliance_integrator.check_collaborative_compliance(
    document_id="doc123",
    user_id="user456"
)

if not compliance['is_compliant']:
    print("Compliance violations:", compliance['violations'])
```

## 🎯 Best Practices

### User Management
- Always initialize users before collaborative actions
- Use appropriate roles based on user responsibilities
- Regularly update user presence for accurate online status

### Document Collaboration
- Set clear collaboration guidelines for each document
- Use workflows for structured review processes
- Leverage automation rules to reduce manual work

### Performance Optimization
- Use appropriate cache TTL values for your use case
- Monitor WebSocket connection counts
- Implement proper cleanup for inactive sessions

### Security
- Validate all user permissions before actions
- Use HTTPS for production deployments
- Implement proper authentication for WebSocket connections

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the debug logs for detailed error information
3. Monitor system statistics for performance issues
4. Check integration status for connectivity problems

## 🗺️ Roadmap

### Planned Features

- [ ] Advanced operational transform algorithms
- [ ] Mobile-responsive collaborative interface
- [ ] Advanced workflow visualization
- [ ] Integration with external collaboration tools (Slack, Teams)
- [ ] Machine learning-powered task recommendations
- [ ] Advanced analytics and reporting
- [ ] Plugin system for custom integrations

### Recent Updates

- ✅ Real-time collaborative editing with operational transforms
- ✅ Comprehensive annotation and commenting system
- ✅ Workflow automation engine with intelligent rules
- ✅ Integration with existing search, version control, and compliance systems
- ✅ WebSocket-based real-time communication
- ✅ Role-based permissions and security
- ✅ Comprehensive dashboard and reporting

---

**Built with ❤️ for seamless team collaboration**