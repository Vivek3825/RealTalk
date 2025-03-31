import os
import wave
import json
import vosk
import pyaudio
import sys
import numpy as np
from scipy import signal
from collections import deque
from colorama import init, Fore, Style  # Add color support

# Initialize colorama for cross-platform colored terminal output
init()

class SpeechRecognizer:
    def __init__(self, lang=None):
        """Initialize the Vosk model based on the selected language."""
        self.model_paths = {
            "hi": "models/vosk-model-hi-0.22",
            "en": "models/vosk-model-en-in-0.5"
        }
        
        # If language is not provided during initialization,
        # prompt user for language selection
        if lang is None:
            lang = self.select_language()
        
        self.selected_lang = lang  # Store the selected language
            
        # Check audio quality and calibrate thresholds
        self.noise_level = self.calibrate_audio()
        
        self.setup_model(lang)
        
    def select_language(self):
        """Prompt user to select a language."""
        print(Fore.CYAN + "Please select a language:" + Style.RESET_ALL)
        print(Fore.YELLOW + "1. Hindi (hi)" + Style.RESET_ALL)
        print(Fore.YELLOW + "2. English (en)" + Style.RESET_ALL)
        
        while True:
            choice = input(Fore.GREEN + "Enter your choice (1/2 or hi/en): " + Style.RESET_ALL).strip().lower()
            
            if choice in ["1", "hi"]:
                return "hi"
            elif choice in ["2", "en"]:
                return "en"
            else:
                print(Fore.RED + "Invalid choice! Please try again." + Style.RESET_ALL)
    
    def setup_model(self, lang):
        """Set up the speech recognition model with improved configuration."""
        self.model_path = self.model_paths.get(lang)
        if not self.model_path:
            raise ValueError("Invalid language choice! Choose 'hi' for Hindi or 'en' for English.")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        language_name = 'Hindi' if lang == 'hi' else 'English'
        print(Fore.CYAN + f"Loading {language_name} recognition model..." + Style.RESET_ALL)
        
        # Create model with better configuration
        self.model = vosk.Model(self.model_path)
        
        # Configure recognizer with words list for better accuracy
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        
        # Enable punctuation if available in the model
        self.recognizer.SetWords(True)
        
        self.audio = pyaudio.PyAudio()
        print(Fore.GREEN + f"✓ {language_name} model loaded successfully!" + Style.RESET_ALL)

    def calibrate_audio(self):
        """Calibrate audio by measuring background noise levels and setting appropriate thresholds."""
        p = pyaudio.PyAudio()
        
        try:
            # Get default input device info
            device_info = p.get_default_input_device_info()
            print(Fore.CYAN + f"Using audio device: {device_info['name']}" + Style.RESET_ALL)
            
            print(Fore.CYAN + "Calibrating microphone (please stay quiet)..." + Style.RESET_ALL)
            
            # Test recording with longer duration for better calibration
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096
            )
            
            # Measure background noise for 2 seconds
            frames = []
            for i in range(0, 8):  # ~2 seconds (8 * 4096 samples at 16kHz)
                data = stream.read(4096, exception_on_overflow=False)
                frames.append(data)
            
            # Calculate noise levels
            noise_levels = []
            noise_profile = None  # For spectral subtraction
            
            for frame in frames:
                # Convert bytes to numpy array for better processing
                audio_array = np.frombuffer(frame, dtype=np.int16)
                noise_levels.append(np.abs(audio_array).mean())
                
                # Build noise profile for spectral subtraction
                if noise_profile is None:
                    noise_profile = np.abs(audio_array)
                else:
                    noise_profile = (noise_profile + np.abs(audio_array)) / 2
            
            # Calculate baseline noise level using mean and standard deviation
            avg_noise = np.mean(noise_levels)
            std_noise = np.std(noise_levels)
            
            # ENHANCEMENT 1: Smarter adaptive thresholding using standard deviation
            speech_threshold = avg_noise + (std_noise * 2.0)
            speech_threshold = max(speech_threshold, 300)  # Minimum threshold, #type: ignore
            
            # Clean up resources
            stream.stop_stream()
            stream.close()
            
            print(Fore.CYAN + f"Background noise level: {avg_noise:.1f}" + Style.RESET_ALL)
            print(Fore.CYAN + f"Noise variability: {std_noise:.1f}" + Style.RESET_ALL)
            print(Fore.GREEN + f"Speech detection threshold set to: {speech_threshold:.1f}" + Style.RESET_ALL)
            
            return {
                'ambient_level': float(avg_noise),  
                'noise_std': float(std_noise),      # Store standard deviation for dynamic adjustments
                'speech_threshold': float(speech_threshold),
                'vad_threshold': float(speech_threshold * 0.7),  # Voice Activity Detection threshold
                'noise_profile': noise_profile      # Store noise profile for spectral subtraction
            }
        finally:
            p.terminate()

    def spectral_subtraction(self, audio_array, noise_estimate):
        """
        ENHANCEMENT 3: Perform spectral subtraction for better noise reduction
        """
        # Apply spectral subtraction (simplified version)
        # For real spectral subtraction, we would use FFT, but this is a simpler approach
        result = np.maximum(audio_array - (noise_estimate * 0.5), 0)  # Subtract scaled noise estimate
        return result

    def process_audio(self, audio_data):
        """Apply advanced noise reduction and normalization to the audio data."""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Apply a simple low-pass filter to reduce high-frequency noise
        b, a = signal.butter(3, 0.05)  #type: ignore
        filtered_audio = signal.lfilter(b, a, audio_array)
        
        # ENHANCEMENT 3: Apply spectral subtraction if noise profile exists
        if hasattr(self.noise_level, 'noise_profile') and self.noise_level['noise_profile'] is not None:
            filtered_audio = self.spectral_subtraction(filtered_audio, self.noise_level['noise_profile'])
        
        # Normalize audio levels for more consistent recognition
        if np.abs(filtered_audio).max() > 0:
            norm_factor = min(32767 / np.abs(filtered_audio).max(), 3.0)  # Cap normalization
            normalized_audio = filtered_audio * norm_factor
        else:
            normalized_audio = filtered_audio
        
        # Convert back to int16
        processed_audio = np.int16(normalized_audio)  #type: ignore
        
        # Convert back to bytes
        return processed_audio.tobytes()

    def recognize_from_mic(self):
        """Capture and recognize speech with enhanced voice activity detection."""
        stream = self.audio.open(format=pyaudio.paInt16, channels=1,
                                rate=16000, input=True, frames_per_buffer=4096)
        stream.start_stream()

        language_name = 'Hindi' if self.selected_lang == 'hi' else 'English'
        print("\n" + Fore.MAGENTA + f"✓ Active language: {language_name}" + Style.RESET_ALL)
        print(Fore.CYAN + f"🎤 Speak now... (Ctrl+C to stop)" + Style.RESET_ALL)
        
        # Setup for rolling average
        audio_level_history = deque(maxlen=10)  # Keep last 10 frames
        for _ in range(10):  # Initialize with ambient noise level
            audio_level_history.append(self.noise_level['ambient_level'])
        
        # Dynamic threshold variables
        last_partial = ""
        is_speaking = False
        silence_frames = 0
        speech_frames = 0
        adaptive_threshold = self.noise_level['speech_threshold']
        
        try:
            while True:
                try:
                    raw_data = stream.read(4096, exception_on_overflow=False)
                except IOError as e:
                    print(Fore.RED + f"Audio error: {e}" + Style.RESET_ALL)
                    continue
                
                # Process audio (noise reduction and normalization)
                processed_data = self.process_audio(raw_data)
                
                # Dynamic voice activity detection with rolling average
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
                current_level = float(np.abs(audio_array).mean())  # Convert to native Python float
                audio_level_history.append(current_level)
                
                # Calculate rolling average and update adaptive threshold
                rolling_avg = sum(audio_level_history) / len(audio_level_history)
                
                # ENHANCEMENT 1: More sophisticated adaptive thresholding
                if not is_speaking:
                    # Gradually adapt the threshold while not speaking
                    adaptive_threshold = max(
                        self.noise_level['ambient_level'] + (self.noise_level['noise_std'] * 2.0),
                        rolling_avg * 1.2  # 20% above rolling average
                    )
                
                # Voice activity detection with adaptive threshold
                if current_level > adaptive_threshold:
                    speech_frames += 1
                    silence_frames = 0
                    if speech_frames > 2 and not is_speaking:
                        is_speaking = True
                        print("\n" + Fore.YELLOW + "🎤 Speech detected..." + Style.RESET_ALL)
                else:
                    silence_frames += 1
                    
                    # ENHANCEMENT 2: Dynamic silence handling
                    # Adjust silence threshold based on noise level
                    silence_limit = max(5, int((rolling_avg / self.noise_level['ambient_level']) * 10))
                    silence_limit = min(silence_limit, 15)  # Cap at reasonable value
                    
                    if silence_frames > silence_limit and is_speaking:
                        is_speaking = False
                        print("\n" + Fore.BLUE + f"⏸️  Pause detected (silence threshold: {silence_limit})" + Style.RESET_ALL)
                
                # Process speech recognition with the processed audio
                if self.recognizer.AcceptWaveform(processed_data):
                    result = json.loads(self.recognizer.Result())
                    if result["text"]:
                        print("\n" + Fore.GREEN + "✓ " + result["text"] + Style.RESET_ALL)
                        last_partial = ""
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    if partial["partial"] and partial["partial"] != last_partial:
                        print(Fore.CYAN + "🔄 " + partial["partial"] + " " * 20 + Style.RESET_ALL, end="\r")
                        last_partial = partial["partial"]
                        
        except KeyboardInterrupt:
            print("\n" + Fore.YELLOW + "Speech recognition stopped." + Style.RESET_ALL)
        except Exception as e:
            print("\n" + Fore.RED + f"Error: {e}" + Style.RESET_ALL)
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

    def __del__(self):
        """Ensure resources are cleaned up when the object is deleted."""
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()

if __name__ == "__main__":
    try:
        # Add a nice welcome message
        print(Fore.MAGENTA + "\n===== RealTalk Speech Recognition =====" + Style.RESET_ALL)
        print(Fore.CYAN + "Enhanced with advanced noise handling and dynamic thresholds" + Style.RESET_ALL)
        
        # Create recognizer with no language specified to trigger the selection menu
        sr = SpeechRecognizer()
        sr.recognize_from_mic()
    except Exception as e:
        print(f"\n" + Fore.RED + f"Error: {e}" + Style.RESET_ALL)
        sys.exit(1)