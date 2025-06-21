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
import subprocess
import mimetypes
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

logger = logging.getLogger(__name__)

# Supported audio formats for conversion
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.webm']

def convert_audio_to_wav(input_path, output_path=None):
    """
    Convert audio file to WAV format using pydub
    
    Args:
        input_path (str): Path to the input audio file
        output_path (str): Optional path for output WAV file. If None, generates a temp path
        
    Returns:
        str: Path to the converted WAV file
        
    Raises:
        Exception: If conversion fails
    """
    try:
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + '_converted.wav'
        
        # Load audio file using pydub
        audio = AudioSegment.from_file(input_path)
        
        # Convert to WAV with standard parameters for ML model
        audio = audio.set_frame_rate(16000)  # Standard sample rate for audio ML models
        audio = audio.set_channels(1)        # Mono channel
        audio = audio.set_sample_width(2)    # 16-bit
        
        # Export as WAV
        audio.export(output_path, format="wav")
        
        logger.info(f"Successfully converted {input_path} to {output_path}")
        return output_path
        
    except CouldntDecodeError as e:
        logger.error(f"Could not decode audio file {input_path}: {e}")
        raise Exception(f"Unsupported audio format or corrupted file: {e}")
    except Exception as e:
        logger.error(f"Error converting audio file {input_path}: {e}")
        raise Exception(f"Audio conversion failed: {e}")

def convert_audio_to_wav_ffmpeg(input_path, output_path=None):
    """
    Alternative conversion method using FFmpeg (fallback option)
    
    Args:
        input_path (str): Path to the input audio file
        output_path (str): Optional path for output WAV file
        
    Returns:
        str: Path to the converted WAV file
        
    Raises:
        Exception: If conversion fails
    """
    try:
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + '_converted.wav'
        
        # FFmpeg command to convert to WAV with ML-friendly parameters
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ar', '16000',  # Sample rate 16kHz
            '-ac', '1',      # Mono channel
            '-sample_fmt', 's16',  # 16-bit
            '-y',           # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")
        
        logger.info(f"Successfully converted {input_path} to {output_path} using FFmpeg")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Exception("Audio conversion timed out")
    except FileNotFoundError:
        raise Exception("FFmpeg not found. Please install FFmpeg or use pydub conversion method")
    except Exception as e:
        logger.error(f"FFmpeg conversion error: {e}")
        raise Exception(f"Audio conversion failed: {e}")

def get_audio_format(file_path_or_name):
    """
    Determine audio format from file path or name
    
    Args:
        file_path_or_name (str): File path or filename
        
    Returns:
        str: File extension (e.g., '.mp3', '.wav')
    """
    return os.path.splitext(file_path_or_name.lower())[1]

def process_audio_file(audio_file, temp_dir):
    """
    Process uploaded audio file - convert to WAV if necessary
    
    Args:
        audio_file: Django uploaded file object
        temp_dir (str): Directory for temporary files
        
    Returns:
        tuple: (wav_file_path, needs_cleanup)
    """
    file_extension = get_audio_format(audio_file.name)
    
    # Generate temp file path
    temp_file_path = os.path.join(temp_dir, f"temp_audio_{int(time.time())}{file_extension}")
    
    # Save uploaded file
    with open(temp_file_path, 'wb+') as destination:
        for chunk in audio_file.chunks():
            destination.write(chunk)
    
    # If already WAV, return as-is
    if file_extension == '.wav':
        return temp_file_path, True
    
    # Convert to WAV
    wav_file_path = os.path.join(temp_dir, f"converted_audio_{int(time.time())}.wav")
    
    try:
        # Try pydub first
        convert_audio_to_wav(temp_file_path, wav_file_path)
        # Clean up original file
        try:
            os.remove(temp_file_path)
        except:
            pass
        return wav_file_path, True
    except Exception as e:
        # Try FFmpeg as fallback
        try:
            convert_audio_to_wav_ffmpeg(temp_file_path, wav_file_path)
            # Clean up original file
            try:
                os.remove(temp_file_path)
            except:
                pass
            return wav_file_path, True
        except Exception as ffmpeg_error:
            # Clean up files
            try:
                os.remove(temp_file_path)
            except:
                pass
            try:
                os.remove(wav_file_path)
            except:
                pass
            raise Exception(f"Failed to convert audio: {str(e)}. FFmpeg fallback also failed: {str(ffmpeg_error)}")

@api_view(['POST'])
@permission_classes([AllowAny])
def predict_audio(request):
    """
    Predict audio class from uploaded audio file (supports multiple formats)
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
        file_extension = get_audio_format(audio_file.name)
        
        # Validate file extension
        if file_extension not in SUPPORTED_AUDIO_FORMATS:
            return Response(
                {
                    'error': f'Unsupported audio format: {file_extension}',
                    'supported_formats': SUPPORTED_AUDIO_FORMATS
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Process audio file (convert if necessary)
        temp_dir = tempfile.gettempdir()
        try:
            wav_file_path, needs_cleanup = process_audio_file(audio_file, temp_dir)
        except Exception as e:
            return Response(
                {
                    'error': f'Audio processing failed: {str(e)}',
                    'original_format': file_extension
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Make prediction
        prediction_result = analyzer.predict(wav_file_path)
        
        # Clean up temp file
        if needs_cleanup:
            try:
                os.remove(wav_file_path)
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
                   f"original_format={file_extension}, "
                   f"class={prediction_result['state']}, "
                   f"confidence={prediction_result['confidence']:.2f}, "
                   f"processing_time={processing_time:.3f}s")
        
        return Response({
            'predicted_class': prediction_result['state'],
            'confidence': prediction_result['confidence'],
            'probabilities': prediction_result['probabilities'],
            'processing_time': round(processing_time, 3),
            'original_format': file_extension,
            'converted_to_wav': file_extension != '.wav',
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
            # Try common test file locations with multiple formats
            possible_paths = []
            base_paths = [
                os.path.join(os.getcwd(), 'test_audio', '3-152007-B'),
                os.path.join(os.getcwd(), 'media', 'test_audio', '3-152007-B'),
                r'C:\Users\B Sostene\Desktop\norsken\baby_care_backend\test_audio\3-152007-B',
                '/test_audio/3-152007-B'
            ]
            
            # Add all supported formats to each base path
            for base_path in base_paths:
                for ext in SUPPORTED_AUDIO_FORMATS:
                    possible_paths.append(base_path + ext)
            
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
                    files = [f for f in os.listdir(search_dir) 
                            if get_audio_format(f) in SUPPORTED_AUDIO_FORMATS]
                    available_files.extend([os.path.join(search_dir, f) for f in files])
            
            error_msg = f'Test audio file not found. '
            if available_files:
                error_msg += f'Available test files: {available_files[:5]}'
            else:
                error_msg += 'No test audio files found in common directories. '
                error_msg += f'Supported formats: {SUPPORTED_AUDIO_FORMATS}'
            
            return Response(
                {'error': error_msg, 'searched_paths': possible_paths if not test_audio_path else [test_audio_path]},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Check if conversion is needed
        file_extension = get_audio_format(test_audio_path)
        wav_file_path = test_audio_path
        needs_cleanup = False
        
        if file_extension != '.wav':
            # Convert to WAV
            temp_dir = tempfile.gettempdir()
            wav_file_path = os.path.join(temp_dir, f"test_converted_{int(time.time())}.wav")
            try:
                convert_audio_to_wav(test_audio_path, wav_file_path)
                needs_cleanup = True
            except Exception as e:
                return Response(
                    {
                        'error': f'Failed to convert test audio file: {str(e)}',
                        'original_format': file_extension
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # Make prediction
        prediction_result = analyzer.predict(wav_file_path)
        
        # Clean up converted file if needed
        if needs_cleanup:
            try:
                os.remove(wav_file_path)
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
                   f"original_format={file_extension}, "
                   f"class={prediction_result['state']}, "
                   f"confidence={prediction_result['confidence']:.2f}")
        
        return Response({
            'test_file': test_audio_path,
            'original_format': file_extension,
            'converted_to_wav': file_extension != '.wav',
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

@api_view(['POST'])
@permission_classes([AllowAny])
def test_audio_prediction_upload(request):
    """
    Test prediction by uploading a test audio file (supports multiple formats)
    """
    start_time = time.time()
    
    try:
        # Check if audio file is provided
        if 'test_audio_file' not in request.FILES:
            return Response(
                {
                    'error': 'No test audio file provided. Please upload a test audio file using "test_audio_file" field.',
                    'supported_formats': SUPPORTED_AUDIO_FORMATS
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        audio_file = request.FILES['test_audio_file']
        file_extension = get_audio_format(audio_file.name)
        
        # Validate file extension
        if file_extension not in SUPPORTED_AUDIO_FORMATS:
            return Response(
                {
                    'error': f'Unsupported audio format: {file_extension}',
                    'supported_formats': SUPPORTED_AUDIO_FORMATS
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Process audio file (convert if necessary)
        temp_dir = tempfile.gettempdir()
        try:
            wav_file_path, needs_cleanup = process_audio_file(audio_file, temp_dir)
        except Exception as e:
            return Response(
                {
                    'error': f'Audio processing failed: {str(e)}',
                    'original_format': file_extension
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Make prediction
        prediction_result = analyzer.predict(wav_file_path)
        
        # Clean up temp file
        if needs_cleanup:
            try:
                os.remove(wav_file_path)
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
                   f"original_format={file_extension}, "
                   f"class={prediction_result['state']}, "
                   f"confidence={prediction_result['confidence']:.2f}")
        
        return Response({
            'test_file_name': audio_file.name,
            'test_file_size': audio_file.size,
            'original_format': file_extension,
            'converted_to_wav': file_extension != '.wav',
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
    Analyze audio data and return baby state prediction (original functionality with conversion support)
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
        
        # Check if audio file needs conversion
        file_extension = get_audio_format(audio_data)
        processed_audio_path = audio_data
        needs_cleanup = False
        
        if file_extension != '.wav' and os.path.exists(audio_data):
            # Convert to WAV
            temp_dir = tempfile.gettempdir()
            processed_audio_path = os.path.join(temp_dir, f"analyze_converted_{int(time.time())}.wav")
            try:
                convert_audio_to_wav(audio_data, processed_audio_path)
                needs_cleanup = True
            except Exception as e:
                return Response(
                    {
                        'error': f'Failed to convert audio file: {str(e)}',
                        'original_format': file_extension
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # Get or create active audio session
        audio_session, created = AudioSession.objects.get_or_create(
            baby=baby,
            status='active',
            defaults={'start_time': timezone.now()}
        )
        
        # Analyze audio using ML model
        analysis_result = analyzer.analyze(processed_audio_path)
        
        # Clean up converted file if needed
        if needs_cleanup:
            try:
                os.remove(processed_audio_path)
            except:
                pass
        
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
                   f"original_format={file_extension}, "
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
            'session_id': str(audio_session.id),
            'original_format': file_extension,
            'converted_to_wav': file_extension != '.wav'
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
        
        # Check audio conversion capabilities
        conversion_methods = []
        try:
            import pydub
            conversion_methods.append("pydub")
        except ImportError:
            pass
        
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            conversion_methods.append("ffmpeg")
        except:
            pass
        
        return Response({
            'status': 'healthy',
            'database': 'connected',
            'ml_model': model_status,
            'model_path': analyzer.model_path,
            'class_names': analyzer.class_names,
            'supported_audio_formats': SUPPORTED_AUDIO_FORMATS,
            'conversion_methods': conversion_methods,
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