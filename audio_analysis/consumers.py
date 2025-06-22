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

    async def connect(self):
        self.baby_id = self.scope['url_route']['kwargs']['baby_id']
        self.room_group_name = f'baby_monitor_{self.baby_id}'

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
            'message': 'WebSocket connected successfully'
        }))

        logger.info(f"WebSocket connected for baby {self.baby_id}")

    async def disconnect(self, close_code):
        # Leave room group
        if self.room_group_name:
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )
        
        logger.info(f"WebSocket disconnected for baby {self.baby_id}")

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
                    
            elif bytes_data:
                # Handle binary audio data
                await self.handle_audio_chunk(bytes_data)
                
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
        
        # Broadcast to all clients in the room
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'recording_status',
                'status': 'started',
                'baby_id': self.baby_id,
                'timestamp': timezone.now().isoformat()
            }
        )

    async def handle_stop_recording(self, data):
        """Handle stop recording command"""
        self.is_recording = False
        
        # Broadcast to all clients in the room
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'recording_status',
                'status': 'stopped',
                'baby_id': self.baby_id,
                'total_chunks': self.chunk_count,
                'timestamp': timezone.now().isoformat()
            }
        )

    async def handle_audio_chunk(self, audio_data):
        """Process audio chunk and predict emotion"""
        if not self.is_recording:
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
                # Broadcast prediction to all clients in the room
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'prediction_update',
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
                
                logger.info(f"Processed audio chunk {self.chunk_count} for baby {self.baby_id}: "
                           f"{prediction_result['state']} ({prediction_result['confidence']:.2f})")
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

    @database_sync_to_async
    def process_audio_async(self, audio_path):
        """Async wrapper for audio processing"""
        return analyzer.predict(audio_path)

    # WebSocket message handlers
    async def recording_status(self, event):
        """Send recording status to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'recording_status',
            'status': event['status'],
            'baby_id': event['baby_id'],
            'timestamp': event['timestamp'],
            'total_chunks': event.get('total_chunks', 0)
        }))

    async def prediction_update(self, event):
        """Send prediction update to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'prediction_update',
            'baby_id': event['baby_id'],
            'predicted_class': event['predicted_class'],
            'confidence': event['confidence'],
            'probabilities': event['probabilities'],
            'processing_time': event['processing_time'],
            'chunk_number': event['chunk_number'],
            'chunk_size': event['chunk_size'],
            'timestamp': event['timestamp']
        }))
