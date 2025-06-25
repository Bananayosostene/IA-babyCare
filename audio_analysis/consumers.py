import json
import asyncio
import tempfile
import os
import time
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.utils import timezone
from .ml_model import analyzer

logger = logging.getLogger(__name__)

class BabyMonitorConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baby_id = None
        self.room_group_name = None
        self.is_recording = False
        self.chunk_count = 0
        self.client_type = None  # 'recorder' or 'dashboard'

    async def connect(self):
        self.baby_id = self.scope['url_route']['kwargs']['baby_id']
        self.room_group_name = f'baby_monitor_{self.baby_id}'
        
        # Get client type from query parameters
        query_string = self.scope.get('query_string', b'').decode()
        if 'type=recorder' in query_string:
            self.client_type = 'recorder'
        elif 'type=dashboard' in query_string:
            self.client_type = 'dashboard'
        else:
            self.client_type = 'unknown'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()
        
        # Send connection confirmation
        await self.send(text_data=json.dumps({
            'type': 'connection_status',
            'status': 'connected',
            'baby_id': self.baby_id,
            'client_type': self.client_type,
            'message': f'WebSocket connected successfully as {self.client_type}'
        }))

        # Notify other clients about new connection
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'client_connected',
                'baby_id': self.baby_id,
                'client_type': self.client_type,
                'channel_name': self.channel_name,
                'timestamp': timezone.now().isoformat()
            }
        )

        logger.info(f"WebSocket connected for baby {self.baby_id} as {self.client_type}")

    async def disconnect(self, close_code):
        # Notify other clients about disconnection
        if self.room_group_name:
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'client_disconnected',
                    'baby_id': self.baby_id,
                    'client_type': self.client_type,
                    'timestamp': timezone.now().isoformat()
                }
            )
            
            # Leave room group
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )
        
        logger.info(f"WebSocket disconnected for baby {self.baby_id} ({self.client_type})")

    async def receive(self, text_data=None, bytes_data=None):
        try:
            if text_data:
                # Handle text messages (control commands)
                data = json.loads(text_data)
                message_type = data.get('type')
                
                if message_type == 'start_recording':
                    await self.handle_start_recording(data)
                elif message_type == 'stop_recording':
                    await self.handle_stop_recording(data)
                elif message_type == 'ping':
                    await self.send(text_data=json.dumps({
                        'type': 'pong',
                        'timestamp': timezone.now().isoformat()
                    }))
                elif message_type == 'request_status':
                    # Send current status to requesting client
                    await self.send_current_status()
                    
            elif bytes_data:
                # Handle binary audio data (only from recorder clients)
                if self.client_type == 'recorder':
                    await self.handle_audio_chunk(bytes_data)
                else:
                    logger.warning(f"Received audio data from non-recorder client: {self.client_type}")
                
        except Exception as e:
            logger.error(f"Error in WebSocket receive: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing message: {str(e)}'
            }))

    async def handle_start_recording(self, data):
        """Handle start recording command"""
        self.is_recording = True
        self.chunk_count = 0
        
        logger.info(f"Starting recording for baby {self.baby_id}")
        
        # Broadcast to ALL clients in the room (including dashboards)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'recording_status_broadcast',
                'status': 'started',
                'baby_id': self.baby_id,
                'recorder_client': self.client_type,
                'timestamp': timezone.now().isoformat()
            }
        )

    async def handle_stop_recording(self, data):
        """Handle stop recording command"""
        self.is_recording = False
        
        logger.info(f"Stopping recording for baby {self.baby_id}")
        
        # Broadcast to ALL clients in the room
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'recording_status_broadcast',
                'status': 'stopped',
                'baby_id': self.baby_id,
                'total_chunks': self.chunk_count,
                'recorder_client': self.client_type,
                'timestamp': timezone.now().isoformat()
            }
        )

    async def handle_audio_chunk(self, audio_data):
        """Process audio chunk and predict emotion"""
        if not self.is_recording:
            logger.warning("Received audio chunk but not recording")
            return

        start_time = time.time()
        
        try:
            # Save audio chunk to temporary file
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(
                temp_dir, 
                f"ws_audio_chunk_{self.baby_id}_{int(time.time())}.webm"
            )
            
            with open(temp_file_path, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"Processing audio chunk {self.chunk_count + 1} for baby {self.baby_id} ({len(audio_data)} bytes)")
            
            # Process audio and get prediction
            prediction_result = await self.process_audio_async(temp_file_path)
            
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except:
                pass
            
            processing_time = time.time() - start_time
            self.chunk_count += 1
            
            if 'error' not in prediction_result:
                # Broadcast prediction to ALL clients in the room (including dashboards)
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'prediction_update_broadcast',
                        'baby_id': self.baby_id,
                        'predicted_class': prediction_result['state'],
                        'confidence': prediction_result['confidence'],
                        'probabilities': prediction_result['probabilities'],
                        'processing_time': round(processing_time, 3),
                        'chunk_number': self.chunk_count,
                        'chunk_size': len(audio_data),
                        'timestamp': timezone.now().isoformat()
                    }
                )
                
                logger.info(f"✅ Broadcasted prediction for baby {self.baby_id}: "
                           f"{prediction_result['state']} ({prediction_result['confidence']:.2f}) "
                           f"to room {self.room_group_name}")
            else:
                # Send error to this client only
                await self.send(text_data=json.dumps({
                    'type': 'processing_error',
                    'error': prediction_result['error'],
                    'chunk_number': self.chunk_count
                }))
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            await self.send(text_data=json.dumps({
                'type': 'processing_error',
                'error': str(e),
                'chunk_number': self.chunk_count
            }))

    async def send_current_status(self):
        """Send current recording status to this client"""
        await self.send(text_data=json.dumps({
            'type': 'current_status',
            'is_recording': self.is_recording,
            'chunk_count': self.chunk_count,
            'baby_id': self.baby_id,
            'client_type': self.client_type,
            'timestamp': timezone.now().isoformat()
        }))

    @database_sync_to_async
    def process_audio_async(self, audio_path):
        """Async wrapper for audio processing"""
        try:
            return analyzer.predict(audio_path)
        except Exception as e:
            logger.error(f"ML model prediction error: {e}")
            return {'error': str(e)}

    # WebSocket message handlers for broadcasting
    async def recording_status_broadcast(self, event):
        """Send recording status to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'recording_status',
            'status': event['status'],
            'baby_id': event['baby_id'],
            'timestamp': event['timestamp'],
            'total_chunks': event.get('total_chunks', 0),
            'recorder_client': event.get('recorder_client', 'unknown')
        }))

    async def prediction_update_broadcast(self, event):
        """Send prediction update to WebSocket - THIS IS THE KEY FIX"""
        message = {
            'type': 'prediction_update',
            'baby_id': event['baby_id'],
            'predicted_class': event['predicted_class'],
            'confidence': event['confidence'],
            'probabilities': event['probabilities'],
            'processing_time': event['processing_time'],
            'chunk_number': event['chunk_number'],
            'chunk_size': event['chunk_size'],
            'timestamp': event['timestamp']
        }
        
        await self.send(text_data=json.dumps(message))
        
        # Log for debugging
        logger.info(f"📤 Sent prediction to {self.client_type} client: "
                   f"{event['predicted_class']} ({event['confidence']:.2f})")

    async def client_connected(self, event):
        """Notify about new client connection"""
        if event['channel_name'] != self.channel_name:  # Don't send to self
            await self.send(text_data=json.dumps({
                'type': 'client_status',
                'status': 'connected',
                'client_type': event['client_type'],
                'baby_id': event['baby_id'],
                'timestamp': event['timestamp']
            }))

    async def client_disconnected(self, event):
        """Notify about client disconnection"""
        await self.send(text_data=json.dumps({
            'type': 'client_status',
            'status': 'disconnected',
            'client_type': event['client_type'],
            'baby_id': event['baby_id'],
            'timestamp': event['timestamp']
        }))
