"""
WebSocket Manager for Real-time Collaboration
Manages WebSocket connections and real-time communication for collaborative features
"""

import json
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Callable
import uuid
import websockets
from dataclasses import dataclass, field

# Import collaborative components
from .collaborative_workflow import collaborative_manager, CollaborationEvent, CollaborationEventType
from .notification_system import notification_manager
from .annotation_system import annotation_manager


@dataclass
class WebSocketClient:
    """Represents a WebSocket client connection"""
    client_id: str
    websocket: Any
    user_id: str
    subscriptions: Set[str] = field(default_factory=set)  # Channel subscriptions
    connected_at: datetime = field(default_factory=datetime.now)
    last_ping: datetime = field(default_factory=datetime.now)
    is_alive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'client_id': self.client_id,
            'user_id': self.user_id,
            'subscriptions': list(self.subscriptions),
            'connected_at': self.connected_at.isoformat(),
            'last_ping': self.last_ping.isoformat(),
            'is_alive': self.is_alive
        }


class WebSocketManager:
    """Manages WebSocket connections for real-time collaboration"""

    def __init__(self, host: str = 'localhost', port: int = 8766):
        self.host = host
        self.port = port
        self.clients: Dict[str, WebSocketClient] = {}
        self.user_clients: Dict[str, List[str]] = {}  # user_id -> client_ids
        self.channel_subscribers: Dict[str, Set[str]] = {}  # channel -> client_ids

        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}

        # Server control
        self.server = None
        self.loop = None
        self._running = False
        self._server_thread = None

        # Initialize handlers
        self._initialize_handlers()

        # Start heartbeat checker
        self._start_heartbeat_checker()

    def _initialize_handlers(self):
        """Initialize WebSocket message handlers"""
        self.message_handlers = {
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
            'ping': self._handle_ping,
            'collaborative_edit': self._handle_collaborative_edit,
            'comment': self._handle_comment,
            'workflow_update': self._handle_workflow_update,
            'notification_read': self._handle_notification_read
        }

    def _start_heartbeat_checker(self):
        """Start heartbeat checking thread"""
        def heartbeat_check():
            while self._running:
                try:
                    self._check_heartbeats()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    print(f"Error in heartbeat check: {e}")
                    time.sleep(60)

        heartbeat_thread = threading.Thread(target=heartbeat_check, daemon=True)
        heartbeat_thread.start()

    def _check_heartbeats(self):
        """Check client heartbeats and remove dead connections"""
        timeout = timedelta(seconds=90)  # 90 seconds timeout
        current_time = datetime.now()
        dead_clients = []

        for client_id, client in self.clients.items():
            if current_time - client.last_ping > timeout:
                dead_clients.append(client_id)

        for client_id in dead_clients:
            client = self.clients[client_id]
            asyncio.run(self._close_client(client))
            print(f"Removed dead client: {client_id}")

    def start_server(self):
        """Start the WebSocket server"""
        if self._running:
            return

        self._running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()

        print(f"WebSocket manager started on ws://{self.host}:{self.port}")

    def stop_server(self):
        """Stop the WebSocket server"""
        self._running = False

        if self.server:
            self.server.close()

        # Close all client connections
        for client in list(self.clients.values()):
            asyncio.run(self._close_client(client))

        print("WebSocket manager stopped")

    def _run_server(self):
        """Run the WebSocket server in asyncio loop"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.server = websockets.serve(self._handle_connection, self.host, self.port)
            self.loop.run_until_complete(self.server)
            self.loop.run_forever()
        except Exception as e:
            print(f"Error running WebSocket server: {e}")
        finally:
            self.loop.close()

    async def _handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        try:
            # Parse authentication from path or initial message
            user_id = await self._authenticate_connection(websocket, path)

            if not user_id:
                await websocket.close(1008, "Authentication failed")
                return

            # Create client
            client = WebSocketClient(
                client_id=str(uuid.uuid4()),
                websocket=websocket,
                user_id=user_id
            )

            await self._register_client(client)

            try:
                await self._handle_client_messages(client)
            finally:
                await self._unregister_client(client)

        except Exception as e:
            print(f"Error handling WebSocket connection: {e}")

    async def _authenticate_connection(self, websocket, path) -> Optional[str]:
        """Authenticate WebSocket connection"""
        try:
            # For now, use a simple token-based authentication
            # In production, this would verify JWT tokens or API keys

            # Check for token in path: /ws/{token}
            path_parts = path.strip('/').split('/')
            if len(path_parts) >= 2 and path_parts[0] == 'ws':
                token = path_parts[1]

                # Simple token validation (would integrate with auth system)
                if token and len(token) > 10:  # Basic validation
                    # Extract user_id from token (simplified)
                    # In reality, would decode JWT or query auth service
                    return f"user_{token[:8]}"  # Mock user_id

            # If no token in path, expect authentication message
            try:
                init_message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                init_data = json.loads(init_message)

                if init_data.get('type') == 'authenticate':
                    token = init_data.get('token')
                    if token:
                        return f"user_{token[:8]}"  # Mock user_id

            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass

        except Exception as e:
            print(f"Authentication error: {e}")

        return None

    async def _register_client(self, client: WebSocketClient):
        """Register a new client"""
        with threading.Lock():
            self.clients[client.client_id] = client

            if client.user_id not in self.user_clients:
                self.user_clients[client.user_id] = []
            self.user_clients[client.user_id].append(client.client_id)

        print(f"Client {client.client_id} registered for user {client.user_id}")

        # Send welcome message
        await self._send_to_client(client, {
            'type': 'welcome',
            'client_id': client.client_id,
            'user_id': client.user_id,
            'timestamp': datetime.now().isoformat()
        })

    async def _unregister_client(self, client: WebSocketClient):
        """Unregister a client"""
        with threading.Lock():
            # Remove from clients
            if client.client_id in self.clients:
                del self.clients[client.client_id]

            # Remove from user clients
            if client.user_id in self.user_clients:
                if client.client_id in self.user_clients[client.user_id]:
                    self.user_clients[client.user_id].remove(client.client_id)

                # Clean up empty user entries
                if not self.user_clients[client.user_id]:
                    del self.user_clients[client.user_id]

            # Remove from channel subscriptions
            for channel, subscribers in self.channel_subscribers.items():
                subscribers.discard(client.client_id)

        print(f"Client {client.client_id} unregistered")

    async def _handle_client_messages(self, client: WebSocketClient):
        """Handle messages from a client"""
        try:
            async for message in client.websocket:
                try:
                    data = json.loads(message)

                    # Update heartbeat
                    client.last_ping = datetime.now()

                    await self._process_client_message(client, data)

                except json.JSONDecodeError:
                    await self._send_error(client, "Invalid JSON")
                except Exception as e:
                    print(f"Error processing message from {client.client_id}: {e}")
                    await self._send_error(client, "Message processing error")

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"Error handling messages for {client.client_id}: {e}")

    async def _process_client_message(self, client: WebSocketClient, data: Dict[str, Any]):
        """Process a message from a client"""
        message_type = data.get('type')

        if message_type in self.message_handlers:
            try:
                handler = self.message_handlers[message_type]
                await handler(client, data)
            except Exception as e:
                print(f"Error in handler {message_type}: {e}")
                await self._send_error(client, f"Handler error: {message_type}")
        else:
            await self._send_error(client, f"Unknown message type: {message_type}")

    async def _handle_subscribe(self, client: WebSocketClient, data: Dict[str, Any]):
        """Handle subscription request"""
        channels = data.get('channels', [])

        for channel in channels:
            # Subscribe to channel
            if channel not in self.channel_subscribers:
                self.channel_subscribers[channel] = set()
            self.channel_subscribers[channel].add(client.client_id)

            # Add to client subscriptions
            client.subscriptions.add(channel)

        await self._send_to_client(client, {
            'type': 'subscribed',
            'channels': channels
        })

    async def _handle_unsubscribe(self, client: WebSocketClient, data: Dict[str, Any]):
        """Handle unsubscription request"""
        channels = data.get('channels', [])

        for channel in channels:
            # Unsubscribe from channel
            if channel in self.channel_subscribers:
                self.channel_subscribers[channel].discard(client.client_id)

            # Remove from client subscriptions
            client.subscriptions.discard(channel)

        await self._send_to_client(client, {
            'type': 'unsubscribed',
            'channels': channels
        })

    async def _handle_ping(self, client: WebSocketClient, data: Dict[str, Any]):
        """Handle ping message"""
        await self._send_to_client(client, {
            'type': 'pong',
            'timestamp': datetime.now().isoformat()
        })

    async def _handle_collaborative_edit(self, client: WebSocketClient, data: Dict[str, Any]):
        """Handle collaborative edit message"""
        # This would integrate with the collaborative editor
        # For now, just acknowledge
        await self._send_to_client(client, {
            'type': 'edit_ack',
            'operation_id': data.get('operation_id'),
            'timestamp': datetime.now().isoformat()
        })

    async def _handle_comment(self, client: WebSocketClient, data: Dict[str, Any]):
        """Handle comment message"""
        # This would integrate with the annotation system
        # For now, just acknowledge
        await self._send_to_client(client, {
            'type': 'comment_ack',
            'comment_id': data.get('comment_id'),
            'timestamp': datetime.now().isoformat()
        })

    async def _handle_workflow_update(self, client: WebSocketClient, data: Dict[str, Any]):
        """Handle workflow update message"""
        # This would integrate with the workflow automation
        # For now, just acknowledge
        await self._send_to_client(client, {
            'type': 'workflow_update_ack',
            'workflow_id': data.get('workflow_id'),
            'timestamp': datetime.now().isoformat()
        })

    async def _handle_notification_read(self, client: WebSocketClient, data: Dict[str, Any]):
        """Handle notification read message"""
        notification_ids = data.get('notification_ids', [])

        # Mark notifications as read
        notification_manager.mark_as_read(client.user_id, notification_ids)

        await self._send_to_client(client, {
            'type': 'notifications_marked_read',
            'notification_ids': notification_ids
        })

    async def _send_to_client(self, client: WebSocketClient, message: Dict[str, Any]):
        """Send a message to a specific client"""
        try:
            json_message = json.dumps(message)
            await client.websocket.send(json_message)
        except Exception as e:
            client.is_alive = False
            raise e

    async def _close_client(self, client: WebSocketClient):
        """Close a client connection"""
        try:
            await client.websocket.close()
        except Exception:
            pass

        client.is_alive = False

    def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """Broadcast a message to all subscribers of a channel"""
        if channel not in self.channel_subscribers:
            return

        subscribers = self.channel_subscribers[channel].copy()
        message['channel'] = channel
        message['timestamp'] = datetime.now().isoformat()

        # Send to all subscribers
        for client_id in subscribers:
            if client_id in self.clients:
                client = self.clients[client_id]
                asyncio.run(self._send_to_client(client, message))

    def broadcast_to_user(self, user_id: str, message: Dict[str, Any]):
        """Broadcast a message to all connections for a user"""
        if user_id not in self.user_clients:
            return

        client_ids = self.user_clients[user_id].copy()
        message['timestamp'] = datetime.now().isoformat()

        for client_id in client_ids:
            if client_id in self.clients:
                client = self.clients[client_id]
                asyncio.run(self._send_to_client(client, message))

    def send_collaborative_event(self, event: CollaborationEvent):
        """Send a collaborative event to relevant clients"""
        # Create message based on event type
        if event.event_type == CollaborationEventType.DOCUMENT_EDIT:
            message = {
                'type': 'document_edited',
                'document_id': event.document_id,
                'user_id': event.user_id,
                'username': event.username,
                'data': event.data
            }
        elif event.event_type == CollaborationEventType.COMMENT_ADDED:
            message = {
                'type': 'comment_added',
                'document_id': event.document_id,
                'user_id': event.user_id,
                'username': event.username,
                'data': event.data
            }
        elif event.event_type == CollaborationEventType.WORKFLOW_STATUS_CHANGED:
            message = {
                'type': 'workflow_updated',
                'document_id': event.document_id,
                'user_id': event.user_id,
                'username': event.username,
                'data': event.data
            }
        elif event.event_type == CollaborationEventType.USER_JOINED:
            message = {
                'type': 'user_joined',
                'document_id': event.document_id,
                'user_id': event.user_id,
                'username': event.username,
                'data': event.data
            }
        elif event.event_type == CollaborationEventType.USER_LEFT:
            message = {
                'type': 'user_left',
                'document_id': event.document_id,
                'user_id': event.user_id,
                'username': event.username,
                'data': event.data
            }
        else:
            return  # Unknown event type

        # Broadcast to document channel
        document_channel = f"document:{event.document_id}"
        self.broadcast_to_channel(document_channel, message)

        # Also broadcast to user channels if needed
        user_channel = f"user:{event.user_id}"
        self.broadcast_to_channel(user_channel, message)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        with threading.Lock():
            total_clients = len(self.clients)
            clients_by_user = {user_id: len(client_ids) for user_id, client_ids in self.user_clients.items()}

            return {
                'total_clients': total_clients,
                'total_users': len(self.user_clients),
                'clients_by_user': clients_by_user,
                'total_channels': len(self.channel_subscribers),
                'channel_subscribers': {
                    channel: len(subscribers)
                    for channel, subscribers in self.channel_subscribers.items()
                },
                'server_status': 'running' if self._running else 'stopped'
            }

    async def _send_error(self, client: WebSocketClient, error_message: str):
        """Send an error message to a client"""
        await self._send_to_client(client, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })


# Global WebSocket manager
websocket_manager = WebSocketManager()

# Integration functions
def setup_websocket_collaboration_integration():
    """Set up integration between WebSocket manager and collaborative systems"""

    def on_collaborative_event(event: CollaborationEvent):
        """Handle collaborative events and broadcast via WebSocket"""
        websocket_manager.send_collaborative_event(event)

    def on_notification_created(user_id: str, notification_type, title: str, message: str):
        """Handle new notifications and broadcast via WebSocket"""
        websocket_manager.broadcast_to_user(user_id, {
            'type': 'notification',
            'notification_type': notification_type,
            'title': title,
            'message': message
        })

    # Connect to notification system
    # This would be connected to notification_manager when notifications are created

    # Connect to collaborative workflow events
    # This would be connected to collaborative_manager when events occur

    print("WebSocket collaboration integration setup complete")