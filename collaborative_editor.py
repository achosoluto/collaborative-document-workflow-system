"""
Real-time Collaborative Document Editor
WebSocket-based collaborative editing with operational transforms
"""

import asyncio
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import uuid
import websockets
from dataclasses import dataclass, field

# Import collaborative workflow components
from .collaborative_workflow import (
    collaborative_manager, OperationalTransform,
    CollaborationEvent, CollaborationEventType, UserRole
)


@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection"""
    websocket: Any
    user_id: str
    document_id: str
    session_id: str
    connected_at: datetime = field(default_factory=datetime.now)
    is_alive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'document_id': self.document_id,
            'session_id': self.session_id,
            'connected_at': self.connected_at.isoformat(),
            'is_alive': self.is_alive
        }


@dataclass
class DocumentState:
    """Represents the current state of a document"""
    document_id: str
    content: str = ""
    version: int = 0
    last_modified: datetime = field(default_factory=datetime.now)
    active_users: Set[str] = field(default_factory=set)
    cursors: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # user_id -> position

    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'content': self.content,
            'version': self.version,
            'last_modified': self.last_modified.isoformat(),
            'active_users': list(self.active_users),
            'cursors': self.cursors
        }


class OperationalTransformEngine:
    """Handles operational transforms for collaborative editing"""

    def __init__(self):
        self.transform_cache: Dict[str, List[OperationalTransform]] = {}

    def transform_operation(self, op: OperationalTransform,
                          concurrent_ops: List[OperationalTransform]) -> OperationalTransform:
        """Transform an operation against concurrent operations"""
        transformed_op = op

        for concurrent_op in concurrent_ops:
            if concurrent_op.operation_id != op.operation_id:
                transformed_op = self._transform_single_operation(transformed_op, concurrent_op)

        return transformed_op

    def _transform_single_operation(self, op1: OperationalTransform,
                                   op2: OperationalTransform) -> OperationalTransform:
        """Transform operation 1 against operation 2"""
        # Simple transformation logic - can be enhanced with more sophisticated algorithms
        if op1.position <= op2.position:
            if op2.operation_type == 'insert':
                # Shift position by insertion length
                new_position = op1.position + len(op2.content) if op1.position > op2.position else op1.position
            else:  # delete
                # Shift position by deletion length (backwards)
                if op1.position > op2.position:
                    new_position = max(op1.position - op2.length, op2.position)
                else:
                    new_position = op1.position
        else:
            if op2.operation_type == 'insert':
                new_position = op1.position
            else:  # delete
                new_position = op1.position - min(op2.length, op1.position - op2.position)

        # Create transformed operation
        return OperationalTransform(
            operation_id=op1.operation_id,
            document_id=op1.document_id,
            user_id=op1.user_id,
            operation_type=op1.operation_type,
            position=new_position,
            content=op1.content,
            length=op1.length,
            timestamp=op1.timestamp,
            version=op1.version
        )

    def apply_operation(self, document_state: DocumentState,
                       operation: OperationalTransform) -> DocumentState:
        """Apply an operation to document state"""
        content = document_state.content
        new_version = document_state.version + 1

        if operation.operation_type == 'insert':
            # Insert content at position
            content = content[:operation.position] + operation.content + content[operation.position:]
        elif operation.operation_type == 'delete':
            # Delete content from position
            end_pos = operation.position + operation.length
            content = content[:operation.position] + content[end_pos:]

        return DocumentState(
            document_id=document_state.document_id,
            content=content,
            version=new_version,
            last_modified=datetime.now(),
            active_users=document_state.active_users.copy(),
            cursors=document_state.cursors.copy()
        )


class CollaborativeEditingServer:
    """WebSocket server for real-time collaborative editing"""

    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.connections: Dict[str, WebSocketConnection] = {}
        self.document_states: Dict[str, DocumentState] = {}
        self.server = None
        self.loop = None

        # Operational transform engine
        self.ot_engine = OperationalTransformEngine()

        # Threading control
        self._running = False
        self._server_thread = None

    def start_server(self):
        """Start the WebSocket server"""
        if self._running:
            return

        self._running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()

        print(f"Collaborative editing server started on ws://{self.host}:{self.port}")

    def stop_server(self):
        """Stop the WebSocket server"""
        self._running = False

        if self.server:
            self.server.close()

        # Close all connections
        for connection in self.connections.values():
            if connection.is_alive:
                asyncio.run(self._close_connection(connection))

        print("Collaborative editing server stopped")

    def _run_server(self):
        """Run the WebSocket server in asyncio loop"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.server = websockets.serve(self._handle_connection, self.host, self.port)
            self.loop.run_until_complete(self.server)
            self.loop.run_forever()
        except Exception as e:
            print(f"Error running collaborative server: {e}")
        finally:
            self.loop.close()

    async def _handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        try:
            # Parse path to extract document_id and user_id
            # Expected format: /document/{document_id}/user/{user_id}
            path_parts = path.strip('/').split('/')
            if len(path_parts) >= 4 and path_parts[0] == 'document':
                document_id = path_parts[1]
                user_id = path_parts[3]

                # Create connection
                connection = WebSocketConnection(
                    websocket=websocket,
                    user_id=user_id,
                    document_id=document_id,
                    session_id=str(uuid.uuid4())
                )

                await self._register_connection(connection)

                try:
                    await self._handle_client_messages(connection)
                finally:
                    await self._unregister_connection(connection)

        except Exception as e:
            print(f"Error handling connection: {e}")

    async def _register_connection(self, connection: WebSocketConnection):
        """Register a new connection"""
        connection_id = f"{connection.user_id}_{connection.session_id}"

        with threading.Lock():
            self.connections[connection_id] = connection

            # Update document state
            if connection.document_id not in self.document_states:
                self.document_states[connection.document_id] = DocumentState(
                    document_id=connection.document_id
                )

            document_state = self.document_states[connection.document_id]
            document_state.active_users.add(connection.user_id)

            # Update user presence
            collaborative_manager.update_user_presence(connection.user_id, True, connection.document_id)

        print(f"User {connection.user_id} connected to document {connection.document_id}")

        # Send current document state
        await self._send_to_connection(connection, {
            'type': 'document_state',
            'data': document_state.to_dict()
        })

        # Send current collaborators
        collaborators = collaborative_manager.get_document_collaborators(connection.document_id)
        await self._send_to_connection(connection, {
            'type': 'collaborators',
            'data': [c.to_dict() for c in collaborators]
        })

    async def _unregister_connection(self, connection: WebSocketConnection):
        """Unregister a connection"""
        connection_id = f"{connection.user_id}_{connection.session_id}"

        with threading.Lock():
            if connection_id in self.connections:
                del self.connections[connection_id]

            # Update document state
            if connection.document_id in self.document_states:
                document_state = self.document_states[connection.document_id]
                document_state.active_users.discard(connection.user_id)

                # Clean up if no active users
                if not document_state.active_users:
                    del self.document_states[connection.document_id]

            # Update user presence
            collaborative_manager.update_user_presence(connection.user_id, False)

        print(f"User {connection.user_id} disconnected from document {connection.document_id}")

    async def _handle_client_messages(self, connection: WebSocketConnection):
        """Handle messages from a client"""
        try:
            async for message in connection.websocket:
                try:
                    data = json.loads(message)
                    await self._process_client_message(connection, data)
                except json.JSONDecodeError:
                    print(f"Invalid JSON received from {connection.user_id}")
                except Exception as e:
                    print(f"Error processing message from {connection.user_id}: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass  # Connection was closed
        except Exception as e:
            print(f"Error handling messages for {connection.user_id}: {e}")

    async def _process_client_message(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Process a message from a client"""
        message_type = data.get('type')

        if message_type == 'operation':
            await self._handle_operation(connection, data)
        elif message_type == 'cursor_position':
            await self._handle_cursor_position(connection, data)
        elif message_type == 'request_state':
            await self._handle_state_request(connection, data)
        elif message_type == 'ping':
            await self._send_to_connection(connection, {'type': 'pong'})

    async def _handle_operation(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle a collaborative editing operation"""
        operation_data = data.get('operation', {})

        # Create operation transform
        operation = OperationalTransform(
            operation_id=str(uuid.uuid4()),
            document_id=connection.document_id,
            user_id=connection.user_id,
            operation_type=operation_data.get('type'),
            position=operation_data.get('position', 0),
            content=operation_data.get('content', ''),
            length=operation_data.get('length', 0),
            timestamp=datetime.now(),
            version=operation_data.get('version', 0)
        )

        # Get concurrent operations since the client's last known version
        document_state = self.document_states.get(connection.document_id)
        if document_state:
            concurrent_ops = collaborative_manager.get_document_operations(
                connection.document_id, operation_data.get('version', 0)
            )

            # Transform the operation against concurrent operations
            transformed_op = self.ot_engine.transform_operation(operation, concurrent_ops)

            # Apply the operation to document state
            new_state = self.ot_engine.apply_operation(document_state, transformed_op)

            # Update document state
            with threading.Lock():
                self.document_states[connection.document_id] = new_state

            # Store operation in history
            collaborative_manager.add_collaborative_edit(
                connection.document_id, connection.user_id,
                transformed_op.operation_type, transformed_op.position,
                transformed_op.content, transformed_op.length
            )

            # Broadcast operation to other clients
            await self._broadcast_operation(connection.document_id, transformed_op, exclude_user=connection.user_id)

            # Send acknowledgment to client
            await self._send_to_connection(connection, {
                'type': 'operation_ack',
                'operation_id': transformed_op.operation_id,
                'new_version': new_state.version
            })

    async def _handle_cursor_position(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle cursor position updates"""
        position = data.get('position', {})

        with threading.Lock():
            if connection.document_id in self.document_states:
                document_state = self.document_states[connection.document_id]
                document_state.cursors[connection.user_id] = position

        # Broadcast cursor position to other clients
        await self._broadcast_to_document(connection.document_id, {
            'type': 'cursor_update',
            'user_id': connection.user_id,
            'position': position
        }, exclude_user=connection.user_id)

    async def _handle_state_request(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle request for current document state"""
        document_state = self.document_states.get(connection.document_id)
        if document_state:
            await self._send_to_connection(connection, {
                'type': 'document_state',
                'data': document_state.to_dict()
            })

    async def _broadcast_operation(self, document_id: str, operation: OperationalTransform, exclude_user: str = None):
        """Broadcast an operation to all connected clients for a document"""
        message = {
            'type': 'operation',
            'operation': operation.to_dict()
        }

        await self._broadcast_to_document(document_id, message, exclude_user)

    async def _broadcast_to_document(self, document_id: str, message: Dict[str, Any], exclude_user: str = None):
        """Broadcast a message to all users connected to a document"""
        disconnected = []

        for connection_id, connection in self.connections.items():
            if (connection.document_id == document_id and
                connection.user_id != exclude_user and
                connection.is_alive):

                try:
                    await self._send_to_connection(connection, message)
                except Exception:
                    disconnected.append(connection_id)

        # Clean up disconnected connections
        for connection_id in disconnected:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                await self._close_connection(connection)

    async def _send_to_connection(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Send a message to a specific connection"""
        try:
            json_message = json.dumps(message)
            await connection.websocket.send(json_message)
        except Exception as e:
            connection.is_alive = False
            raise e

    async def _close_connection(self, connection: WebSocketConnection):
        """Close a WebSocket connection"""
        try:
            await connection.websocket.close()
        except Exception:
            pass  # Connection might already be closed

        connection.is_alive = False

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with threading.Lock():
            total_connections = len(self.connections)
            active_documents = len(set(conn.document_id for conn in self.connections.values()))
            users_by_document = {}

            for connection in self.connections.values():
                if connection.document_id not in users_by_document:
                    users_by_document[connection.document_id] = []
                users_by_document[connection.document_id].append(connection.user_id)

            return {
                'total_connections': total_connections,
                'active_documents': active_documents,
                'users_by_document': users_by_document,
                'server_status': 'running' if self._running else 'stopped'
            }


class CollaborativeEditorManager:
    """Manager for collaborative editing functionality"""

    def __init__(self):
        self.server = CollaborativeEditingServer()
        self.document_locks: Dict[str, threading.Lock] = {}

    def start_collaborative_editing(self):
        """Start the collaborative editing server"""
        self.server.start_server()

    def stop_collaborative_editing(self):
        """Stop the collaborative editing server"""
        self.server.stop_server()

    def get_document_lock(self, document_id: str) -> threading.Lock:
        """Get a lock for a document (creates if doesn't exist)"""
        if document_id not in self.document_locks:
            self.document_locks[document_id] = threading.Lock()
        return self.document_locks[document_id]

    def get_document_state(self, document_id: str) -> Optional[DocumentState]:
        """Get current state of a document"""
        return self.server.document_states.get(document_id)

    def initialize_document_for_collaboration(self, document_id: str, initial_content: str = ""):
        """Initialize a document for collaborative editing"""
        with threading.Lock():
            if document_id not in self.server.document_states:
                self.server.document_states[document_id] = DocumentState(
                    document_id=document_id,
                    content=initial_content
                )

    def get_collaborative_stats(self) -> Dict[str, Any]:
        """Get comprehensive collaborative editing statistics"""
        connection_stats = self.server.get_connection_stats()
        workflow_stats = collaborative_manager.get_collaboration_stats()

        return {
            'connection_stats': connection_stats,
            'workflow_stats': workflow_stats,
            'document_states': len(self.server.document_states),
            'total_operations': sum(len(ops) for ops in collaborative_manager.operation_history.values())
        }


# Global collaborative editor manager
editor_manager = CollaborativeEditorManager()