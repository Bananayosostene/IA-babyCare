import tensorflow as tf
import numpy as np
import os
import logging
import librosa
import soundfile as sf
from scipy.io import wavfile

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self, model_path="/models/sound_model.keras"):
        self.model_path = model_path
        self.model = None
        self.class_names = ['cry', 'hungry', 'laugh', 'noise', 'silence']
        self.load_model()
    
    def load_model(self):
        """Load the TensorFlow model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.error(f"Model file not found at {self.model_path}")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def get_spectrogram(self, waveform):
        """Convert waveform to spectrogram"""
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram
    
    def convert_to_pcm_wav(self, input_path, output_path):
        """Convert audio file to PCM WAV format using librosa"""
        try:
            # Load audio using librosa (handles various formats)
            audio, sample_rate = librosa.load(input_path, sr=16000, mono=True)
            
            # Save as PCM WAV using soundfile
            sf.write(output_path, audio, sample_rate, format='WAV', subtype='PCM_16')
            return True
        except Exception as e:
            logger.error(f"Error converting audio to PCM WAV: {e}")
            return False
    
    def preprocess_audio_with_librosa(self, audio_path):
        """Preprocess audio using librosa (more robust)"""
        try:
            # Load audio using librosa
            audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            # Ensure we have exactly 16000 samples (1 second at 16kHz)
            if len(audio) > 16000:
                audio = audio[:16000]
            elif len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
            
            # Convert to TensorFlow tensor
            waveform = tf.constant(audio, dtype=tf.float32)
            
            # Convert to spectrogram
            spectrogram = self.get_spectrogram(waveform)
            spectrogram = spectrogram[tf.newaxis, ...]
            
            return spectrogram
        except Exception as e:
            logger.error(f"Error preprocessing audio with librosa: {e}")
            return None
    
    def preprocess_audio_tensorflow(self, audio_path):
        """Preprocess audio using TensorFlow (original method with PCM conversion)"""
        try:
            # First, try to convert to PCM format
            temp_pcm_path = audio_path + "_pcm.wav"
            if self.convert_to_pcm_wav(audio_path, temp_pcm_path):
                try:
                    # Read the PCM WAV file
                    x = tf.io.read_file(str(temp_pcm_path))
                    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
                    x = tf.squeeze(x, axis=-1)
                    
                    # Convert to spectrogram
                    x = self.get_spectrogram(x)
                    x = x[tf.newaxis, ...]
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_pcm_path)
                    except:
                        pass
                    
                    return x
                except Exception as e:
                    logger.error(f"Error with TensorFlow decode_wav: {e}")
                    # Clean up temp file
                    try:
                        os.remove(temp_pcm_path)
                    except:
                        pass
                    return None
            else:
                return None
        except Exception as e:
            logger.error(f"Error in TensorFlow preprocessing: {e}")
            return None
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio file for prediction with multiple fallback methods"""
        try:
            # Method 1: Try librosa approach (most robust)
            processed_audio = self.preprocess_audio_with_librosa(audio_path)
            if processed_audio is not None:
                logger.info("Successfully preprocessed audio using librosa")
                return processed_audio
            
            # Method 2: Try TensorFlow with PCM conversion
            logger.info("Librosa preprocessing failed, trying TensorFlow with PCM conversion")
            processed_audio = self.preprocess_audio_tensorflow(audio_path)
            if processed_audio is not None:
                logger.info("Successfully preprocessed audio using TensorFlow with PCM conversion")
                return processed_audio
            
            # Method 3: Try scipy.io.wavfile as last resort
            logger.info("TensorFlow preprocessing failed, trying scipy.io.wavfile")
            try:
                sample_rate, audio = wavfile.read(audio_path)
                
                # Convert to float32 and normalize
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                elif audio.dtype == np.uint8:
                    audio = (audio.astype(np.float32) - 128) / 128.0
                
                # Handle stereo to mono conversion
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                
                # Ensure we have exactly 16000 samples
                if len(audio) > 16000:
                    audio = audio[:16000]
                elif len(audio) < 16000:
                    audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
                
                # Convert to TensorFlow tensor and spectrogram
                waveform = tf.constant(audio, dtype=tf.float32)
                spectrogram = self.get_spectrogram(waveform)
                spectrogram = spectrogram[tf.newaxis, ...]
                
                logger.info("Successfully preprocessed audio using scipy.io.wavfile")
                return spectrogram
                
            except Exception as e:
                logger.error(f"Error with scipy.io.wavfile: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return None
    
    def predict(self, audio_path):
        """Predict audio class from file path"""
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'state': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
        
        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_path)
            if processed_audio is None:
                return {
                    'error': 'Failed to preprocess audio',
                    'state': 'unknown',
                    'confidence': 0.0,
                    'probabilities': {}
                }
            
            # Make prediction
            prediction = self.model(processed_audio)
            probabilities = tf.nn.softmax(prediction[0]).numpy()
            
            # Get predicted class
            predicted_index = tf.argmax(probabilities).numpy()
            predicted_class = self.class_names[predicted_index]
            confidence = float(probabilities[predicted_index])
            
            # Create probabilities dictionary
            prob_dict = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities)
            }
            
            logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.2f})")
            
            return {
                'state': predicted_class,
                'confidence': confidence,
                'probabilities': prob_dict,
                'predicted_index': int(predicted_index)
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'error': str(e),
                'state': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
    
    def analyze(self, audio_data):
        """
        Analyze audio data (for compatibility with existing code)
        This method expects audio_data to be a file path
        """
        return self.predict(audio_data)

# Create global analyzer instance
analyzer = AudioAnalyzer(r"C:\Users\B Sostene\Desktop\norsken\baby_care_backend\models\model.keras")