from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from django.db import transaction
from .models import Baby, AudioSession, SensorData, Device
from .ml_model import analyzer
from rest_framework import generics
from .models import User
from .serializers import UserSerializer
import time
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([AllowAny])  # Changed for testing - you can add authentication later
def predict_audio(request):
    """
    Predict audio class from uploaded audio file
    """
    start_time = time.time()
    
    try:
        # Check if audio file is provided
        if 'audio_file' not in request.FILES:
            return Response(
                {'error': 'No audio file provided. Please upload an audio file.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        audio_file = request.FILES['audio_file']
        
        # Validate file extension
        if not audio_file.name.lower().endswith('.wav'):
            return Response(
                {'error': 'Only WAV format is supported'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Save uploaded file temporarily
        # temp_dir = '/tmp'  
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"temp_audio_{int(time.time())}.wav")
        
        with open(temp_file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)
        
        # Make prediction
        prediction_result = analyzer.predict(temp_file_path)
        
        # Clean up temp file
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        processing_time = time.time() - start_time
        
        if 'error' in prediction_result:
            return Response(
                {
                    'error': prediction_result['error'],
                    'processing_time': round(processing_time, 3)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Log the prediction
        logger.info(f"Audio prediction completed: "
                   f"class={prediction_result['state']}, "
                   f"confidence={prediction_result['confidence']:.2f}, "
                   f"processing_time={processing_time:.3f}s")
        
        return Response({
            'predicted_class': prediction_result['state'],
            'confidence': prediction_result['confidence'],
            'probabilities': prediction_result['probabilities'],
            'processing_time': round(processing_time, 3),
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in audio prediction: {e}")
        return Response(
            {'error': 'Internal server error', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([AllowAny])
def test_audio_prediction(request):
    """
    Test prediction with a provided audio file path or use default test files
    """
    start_time = time.time()
    
    try:
        # Get test file path from request data, or use default locations
        test_audio_path = request.data.get('test_file_path')
        
        if not test_audio_path:
            # Try common test file locations
            possible_paths = [
                os.path.join(os.getcwd(), 'test_audio', '3-152007-B.wav'),
                os.path.join(os.getcwd(), 'media', 'test_audio', '3-152007-B.wav'),
                r'C:\Users\B Sostene\Desktop\norsken\baby_care_backend\test_audio\3-152007-B.wav',
                '/test_audio/3-152007-B.wav'  # Original path
            ]
            
            test_audio_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    test_audio_path = path
                    break
        
        if not test_audio_path or not os.path.exists(test_audio_path):
            # List available test files if any
            search_dirs = [
                os.path.join(os.getcwd(), 'test_audio'),
                os.path.join(os.getcwd(), 'media', 'test_audio'),
                '/test_audio'
            ]
            
            available_files = []
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    files = [f for f in os.listdir(search_dir) if f.endswith(('.wav', '.m4a'))]
                    available_files.extend([os.path.join(search_dir, f) for f in files])
            
            error_msg = f'Test audio file not found. '
            if available_files:
                error_msg += f'Available test files: {available_files[:5]}'  # Show first 5
            else:
                error_msg += 'No test audio files found in common directories. '
                error_msg += 'Please provide test_file_path in request body or place test files in test_audio/ directory.'
            
            return Response(
                {'error': error_msg, 'searched_paths': possible_paths if not test_audio_path else [test_audio_path]},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Make prediction
        prediction_result = analyzer.predict(test_audio_path)
        
        processing_time = time.time() - start_time
        
        if 'error' in prediction_result:
            return Response(
                {
                    'error': prediction_result['error'],
                    'processing_time': round(processing_time, 3)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        logger.info(f"Test audio prediction completed: "
                   f"class={prediction_result['state']}, "
                   f"confidence={prediction_result['confidence']:.2f}")
        
        return Response({
            'test_file': test_audio_path,
            'predicted_class': prediction_result['state'],
            'confidence': prediction_result['confidence'],
            'probabilities': prediction_result['probabilities'],
            'processing_time': round(processing_time, 3),
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in test audio prediction: {e}")
        return Response(
            {'error': 'Internal server error', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Alternative: File upload version of test endpoint
@api_view(['POST'])
@permission_classes([AllowAny])
def test_audio_prediction_upload(request):
    """
    Test prediction by uploading a test audio file
    """
    start_time = time.time()
    
    try:
        # Check if audio file is provided
        if 'test_audio_file' not in request.FILES:
            return Response(
                {'error': 'No test audio file provided. Please upload a test audio file using "test_audio_file" field.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        audio_file = request.FILES['test_audio_file']
        
        # Validate file extension
        if not audio_file.name.lower().endswith(('.wav', '.m4a')):
            return Response(
                {'error': 'Only WAV and M4A formats are supported for testing'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Save uploaded file temporarily
        # temp_dir = '/tmp'
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"test_audio_{int(time.time())}_{audio_file.name}")
        
        with open(temp_file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)
        
        # Make prediction
        prediction_result = analyzer.predict(temp_file_path)
        
        # Clean up temp file
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        processing_time = time.time() - start_time
        
        if 'error' in prediction_result:
            return Response(
                {
                    'error': prediction_result['error'],
                    'processing_time': round(processing_time, 3)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        logger.info(f"Test audio prediction completed: "
                   f"file={audio_file.name}, "
                   f"class={prediction_result['state']}, "
                   f"confidence={prediction_result['confidence']:.2f}")
        
        return Response({
            'test_file_name': audio_file.name,
            'test_file_size': audio_file.size,
            'predicted_class': prediction_result['state'],
            'confidence': prediction_result['confidence'],
            'probabilities': prediction_result['probabilities'],
            'processing_time': round(processing_time, 3),
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in test audio prediction upload: {e}")
        return Response(
            {'error': 'Internal server error', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_audio(request):
    """
    Analyze audio data and return baby state prediction (original functionality)
    """
    start_time = time.time()

    try:
        data = request.data
        baby_id = data.get('baby_id')
        audio_data = data.get('audio_data')  # This should be a file path
        timestamp = data.get('timestamp')
        sample_rate = data.get('sample_rate', 16000)
        duration = data.get('duration', 2.0)
        user_id = data.get('user_id')
        
        if not all([baby_id, audio_data, timestamp]):
            return Response(
                {'error': 'Missing required fields: baby_id, audio_data, timestamp'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Verify baby exists and belongs to user
        try:
            baby = Baby.objects.get(id=baby_id, parent_id=user_id)
        except Baby.DoesNotExist:
            return Response(
                {'error': 'Baby not found or access denied'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Get or create active audio session
        audio_session, created = AudioSession.objects.get_or_create(
            baby=baby,
            status='active',
            defaults={'start_time': timezone.now()}
        )
        
        # Analyze audio using ML model
        analysis_result = analyzer.analyze(audio_data)
        
        # Get or create a default device for this baby
        device, _ = Device.objects.get_or_create(
            baby=baby,
            type='sensor',
            defaults={
                'name': f'{baby.name} Audio Sensor',
                'serial_number': f'audio-{baby_id[:8]}',
                'status': 'active'
            }
        )
        
        # Save analysis result to database
        with transaction.atomic():
            # Create sensor data record
            sensor_data = SensorData.objects.create(
                device_id=device.id,
                baby=baby,
                status=analysis_result['state'],
                confidence_score=analysis_result['confidence'],
                audio_session=audio_session,
                timestamp=timezone.now()
            )
            
            # Update baby's current state
            baby.state = analysis_result['state']
            baby.save()
            
            # Update audio session chunk count
            audio_session.total_chunks += 1
            audio_session.save()
        
        processing_time = time.time() - start_time
        
        # Log the analysis
        logger.info(f"Audio analysis completed for baby {baby_id}: "
                   f"state={analysis_result['state']}, "
                   f"confidence={analysis_result['confidence']:.2f}, "
                   f"processing_time={processing_time:.3f}s")
        
        # Return analysis result
        return Response({
            'state': analysis_result['state'],
            'confidence': analysis_result['confidence'],
            'timestamp': sensor_data.timestamp.isoformat(),
            'processing_time': round(processing_time, 3),
            'probabilities': analysis_result['probabilities'],
            'session_id': str(audio_session.id)
        })
        
    except Exception as e:
        logger.error(f"Error in audio analysis: {e}")
        return Response(
            {'error': 'Internal server error', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint"""
    try:
        # Check database connection
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        
        # Check if ML model is loaded
        model_status = "loaded" if analyzer.model is not None else "not_loaded"
        
        return Response({
            'status': 'healthy',
            'database': 'connected',
            'ml_model': model_status,
            'model_path': analyzer.model_path,
            'class_names': analyzer.class_names,
            'timestamp': timezone.now().isoformat()
        })
    except Exception as e:
        return Response({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

class UserListCreateView(generics.ListCreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny]