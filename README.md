# 🤝 Collaborative Document Workflow System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/achosoluto/collaborative-document-workflow-system/issues)

A comprehensive real-time collaborative document management system that integrates seamlessly with existing search, version control, compliance, and summarization systems.

## 🌟 Features

- **🔄 Real-time Collaborative Editing** - Multiple users can edit documents simultaneously with operational transforms
- **📝 Document Review & Approval Workflows** - Structured review processes with role-based permissions
- **💬 Advanced Commenting System** - Threaded discussions with mentions, reactions, and document positioning
- **🚀 Team Collaboration Features** - Multi-channel notifications and real-time presence tracking
- **🤖 Intelligent Workflow Automation** - Rule-based triggers and automated task management
- **📊 Interactive Dashboard & Reporting** - Customizable analytics widgets with live metrics
- **🌐 WebSocket Support** - Real-time bidirectional communication for instant updates
- **🔗 System Integration** - Seamless connection with search, version control, compliance, and summarization systems

## 🏗️ Architecture

### Core Components

- **Collaborative Workflow Manager** - User management and coordination
- **Real-time Collaborative Editor** - WebSocket-based document editing with OT algorithms
- **Annotation System** - Threaded commenting with rich content support
- **Notification System** - Multi-channel notifications with user preferences
- **Workflow Automation Engine** - Intelligent rule-based task automation
- **Dashboard System** - Real-time widgets and comprehensive analytics
- **WebSocket Manager** - Real-time communication and event broadcasting
- **Integration Layer** - Cross-platform compatibility and system interoperability

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- WebSocket library
- Flask
- Additional dependencies listed in `requirements.txt`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/achosoluto/collaborative-document-workflow-system.git
   cd collaborative-document-workflow-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   # WebSocket Configuration
   export WEBSOCKET_HOST=localhost
   export WEBSOCKET_PORT=8766

   # Email Notifications (optional)
   export SMTP_SERVER=smtp.gmail.com
   export SMTP_PORT=587
   export SMTP_USERNAME=your-email@gmail.com
   export SMTP_PASSWORD=your-app-password
   ```

4. **Initialize the system**
   ```python
   from search_engine.collaborative_integration import integration_manager

   # Initialize all integrations
   integration_manager.initialize_all_integrations()
   ```

5. **Start the collaborative system**
   ```bash
   python -m search_engine.collaborative_integration
   ```

### Basic Usage

```python
from search_engine.collaborative_workflow import collaborative_manager

# Add users to the system
user = collaborative_manager.add_user(
    user_id="user123",
    username="John Doe",
    email="john@example.com",
    role="editor"
)

# Add collaborators to a document
collaborative_manager.add_document_collaborator(
    document_id="doc456",
    user_id="user123",
    role="editor"
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

# Add comments with threading
comment = collaborative_manager.add_comment(
    document_id="doc456",
    user_id="user123",
    content="Please review this section",
    position={"x": 100, "y": 200}
)
```

## 📋 User Roles & Permissions

| Role | Permissions |
|------|-------------|
| **Viewer** | Read-only access to documents |
| **Editor** | Full document editing capabilities |
| **Reviewer** | Document review and commenting |
| **Approver** | Workflow approval and sign-off |
| **Admin** | Full system administration |

## 🌐 Web Interface

Access the collaborative interface through:
- **Dashboard**: `/collaborative/dashboard`
- **Document Collaboration**: `/collaborative/document/<doc_id>`
- **API Endpoints**: `/collaborative/api/*`

### JavaScript Integration Example

```javascript
// Initialize WebSocket connection
const ws = new WebSocket('ws://localhost:8766/ws/your-token');

// Subscribe to document channels
ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['document:doc123', 'user:user123']
}));

// Handle real-time document updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'document_edited') {
        updateDocumentContent(data);
    } else if (data.type === 'comment_added') {
        addCommentToUI(data);
    }
};
```

## 📊 Dashboard Widgets

Available dashboard widgets include:
- **Collaboration Overview** - Active collaborators and recent activity
- **Workflow Progress** - Task status and completion tracking
- **Document Analytics** - Usage statistics and metrics
- **Team Presence** - Online status and availability
- **Notification Center** - Real-time alerts and updates
- **Compliance Monitoring** - Document compliance status

## 🤖 Automation Engine

Configure intelligent automation rules:

```python
from search_engine.workflow_automation import WorkflowRule, AutomationTrigger

rule = WorkflowRule(
    name="Auto-approve After 3 Days",
    description="Auto-approve workflows after 3 days of inactivity",
    trigger=AutomationTrigger.TIME_BASED,
    conditions=[{"type": "days_in_status", "value": 3}],
    actions=[{"type": "update_status", "value": "approved"}]
)
```

## 🔗 System Integration

The collaborative system integrates seamlessly with:

- **Search Engine** - Enhanced search with collaboration context
- **Version Control** - Collaborative version management
- **Compliance Systems** - Real-time compliance monitoring
- **Summarization Engine** - Automated document summarization

## 🚀 Deployment

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

### Production Configuration

```bash
# Environment variables for production
export WEBSOCKET_HOST=0.0.0.0
export WEBSOCKET_PORT=8766
export CACHE_TTL=300
export AUTOMATION_SCHEDULER_INTERVAL=60

# Start all services
python -m search_engine.collaborative_integration
```

## 📈 Monitoring & Analytics

### Key Performance Metrics

- **Response Times** - API response time tracking
- **User Activity** - Active users and session monitoring
- **System Health** - Background process monitoring
- **Real-time Connections** - WebSocket connection statistics

### Getting System Statistics

```python
# Get comprehensive system stats
stats = collaborative_manager.get_system_statistics()
print(f"Active Users: {stats['active_users']}")
print(f"Total Documents: {stats['total_documents']}")
print(f"Workflow Completion Rate: {stats['completion_rate']}%")
```

## 🔧 API Documentation

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/collaborative/api/users` | Initialize user |
| GET | `/collaborative/api/documents/:id/collaborators` | Get collaborators |
| POST | `/collaborative/api/documents/:id/comments` | Add comment |
| GET | `/collaborative/api/users/:id/notifications` | Get notifications |
| POST | `/collaborative/api/workflows` | Create workflow |

### WebSocket Events

**Client Events:**
- `subscribe` - Subscribe to channels
- `collaborative_edit` - Send edit operations
- `cursor_position` - Update cursor location

**Server Events:**
- `document_edited` - Real-time document updates
- `comment_added` - New comment notifications
- `notification` - System notifications

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest search_engine/ -v

# Run specific component tests
python test_collaborative_system.py
python test_workflow_automation.py

# Run integration tests
python test_integration.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Write comprehensive unit tests
- Ensure all tests pass before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python async/await patterns
- WebSocket implementation using `websockets` library
- Operational Transform algorithms for conflict-free collaboration
- Comprehensive integration with existing document management systems

## 📧 Support

- 🐛 [Issues](https://github.com/achosoluto/collaborative-document-workflow-system/issues)
- 💬 [Discussions](https://github.com/achosoluto/collaborative-document-workflow-system/discussions)
- 📖 [Documentation](https://github.com/achosoluto/collaborative-document-workflow-system/wiki)

---

**Transform document management into a collaborative powerhouse 🚀**