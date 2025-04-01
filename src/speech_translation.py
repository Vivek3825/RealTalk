import time
import sys
import os

# Add the parent directory to the path so Python can find your modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your modules
from src.speech_recognition.speech_recognizer import SpeechRecognizer
from src.translation.translator import Translator

class SpeechTranslation:
    def __init__(self, source_lang="hi"):
        """Initialize the Speech Recognizer & Translator
        
        Args:
            source_lang (str): Initial language to recognize (default: "hi" for Hindi)
        """
        try:
            self.sr = SpeechRecognizer(lang=source_lang)
            self.translator = Translator()
            print("✅ Speech recognition and translation systems initialized")
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            raise

    def start(self):
        """Continuously recognize speech and translate with a cleaner interface"""
        print("\n" + "=" * 60)
        print("🎤  RealTalk Speech Translation")
        print("=" * 60)
        print("• Speak in Hindi or English")
        print("• Press Ctrl+C to stop")
        print("• Your speech will be automatically translated")
        print("=" * 60 + "\n")

        try:
            while True:
                try:
                    # Recognize speech without excessive messages
                    print("\n👂 Listening...")
                    recognized_text = self.sr.recognize_from_mic()
                    
                    if recognized_text:
                        # Detect language
                        source_lang = "hi" if self.sr.selected_lang == "hi" else "en"
                        target_lang = "en" if source_lang == "hi" else "hi"
                        
                        # Translate recognized text (without debug messages)
                        translated_text = self.translator.translate_text(recognized_text, source_lang, target_lang)
                        
                        # Display result with better formatting
                        print("\n" + "─" * 60)
                        print(f"🗣️  {source_lang.upper()}: {recognized_text}")
                        print(f"🌍  {target_lang.upper()}: {translated_text}")
                        print("─" * 60)
                    
                    time.sleep(0.5)  # Small delay for better performance
                    
                except Exception as e:
                    print(f"\n⚠️ Error: {str(e)}")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("🛑 Translation stopped")
            print("=" * 60)

def check_model_files():
    """Check if required model files exist"""
    model_dirs = [
        "models/Helsinki-NLP_opus-mt-en-hi",
        "models/Helsinki-NLP_opus-mt-hi-en"
    ]
    
    all_exist = True
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            print(f"❌ Missing model directory: {model_dir}")
            all_exist = False
    
    return all_exist

def test_translator():
    """Test the translator independently"""
    try:
        translator = Translator()
        
        # Test English to Hindi
        en_text = "Hello, how are you?"
        print(f"Testing EN→HI: '{en_text}'")
        hi_translation = translator.translate_text(en_text, "en", "hi")
        print(f"Result: '{hi_translation}'")
        
        # Test Hindi to English
        hi_text = "नमस्ते, आप कैसे हैं?"
        print(f"Testing HI→EN: '{hi_text}'")
        en_translation = translator.translate_text(hi_text, "hi", "en")
        print(f"Result: '{en_translation}'")
        
        return True
    except Exception as e:
        print(f"❌ Translator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the translator first
    if test_translator():
        print("✅ Translator test passed! Starting speech translation...")
        st = SpeechTranslation()
        st.start()
    else:
        print("❌ Translator test failed. Please check your implementation.")
