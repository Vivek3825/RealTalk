from transformers import MarianMTModel, MarianTokenizer
import os
import sys
from colorama import init, Fore, Style
import gc
import time

# Initialize colorama
init()

# Define the model paths
MODEL_DIR = "models"
EN_TO_HI_MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"
HI_TO_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"

def download_model(model_name):
    """Downloads and saves the translation model if not already available."""
    model_path = os.path.join(MODEL_DIR, model_name.replace("/", "_"))
    
    if not os.path.exists(model_path):
        print(Fore.YELLOW + f"⬇️ Downloading {model_name}..." + Style.RESET_ALL)
        try:
            # Create progress indicator
            chars = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
            
            # Start download
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            
            # Show some activity during download
            print(Fore.CYAN + "Saving model locally..." + Style.RESET_ALL, end="")
            for i in range(10):
                print(Fore.CYAN + f" {chars[i % len(chars)]}" + Style.RESET_ALL, end="", flush=True)
                time.sleep(0.2)
                print("\b" * 2, end="", flush=True)
            
            # Save model locally
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(Fore.GREEN + f"\n✓ Model {model_name} saved at {model_path}" + Style.RESET_ALL)
        
        except Exception as e:
            print(Fore.RED + f"\n❌ Error downloading model: {e}" + Style.RESET_ALL)
            raise
    
    return model_path

class Translator:
    def __init__(self):
        """Initialize translation models for English-Hindi and Hindi-English"""
        try:
            # Ensure models are downloaded
            print(Fore.CYAN + "Preparing translation models..." + Style.RESET_ALL)
            en_to_hi_model_path = download_model(EN_TO_HI_MODEL_NAME)
            hi_to_en_model_path = download_model(HI_TO_EN_MODEL_NAME)
            
            # Load models with status messages
            print(Fore.CYAN + "Loading English→Hindi model..." + Style.RESET_ALL)
            self.en_to_hi_model = MarianMTModel.from_pretrained(en_to_hi_model_path)
            self.en_to_hi_tokenizer = MarianTokenizer.from_pretrained(en_to_hi_model_path)
            
            print(Fore.CYAN + "Loading Hindi→English model..." + Style.RESET_ALL)
            self.hi_to_en_model = MarianMTModel.from_pretrained(hi_to_en_model_path)
            self.hi_to_en_tokenizer = MarianTokenizer.from_pretrained(hi_to_en_model_path)
            
            print(Fore.GREEN + "✓ Translation models ready" + Style.RESET_ALL)
            
            # Force garbage collection after loading large models
            gc.collect()
            
        except Exception as e:
            print(Fore.RED + f"❌ Error initializing translator: {e}" + Style.RESET_ALL)
            raise
    
    def translate_text(self, text, src_lang="en", tgt_lang="hi"):
        """Translates text between English and Hindi.
        
        Args:
            text (str): Text to translate
            src_lang (str): Source language code ('en' or 'hi')
            tgt_lang (str): Target language code ('en' or 'hi')
            
        Returns:
            str: Translated text
        """
        if not text or text.strip() == "":
            return ""
            
        try:
            # Select the appropriate model
            if src_lang == "en" and tgt_lang == "hi":
                model, tokenizer = self.en_to_hi_model, self.en_to_hi_tokenizer
            elif src_lang == "hi" and tgt_lang == "en":
                model, tokenizer = self.hi_to_en_model, self.hi_to_en_tokenizer
            else:
                raise ValueError(f"Unsupported language pair: {src_lang}→{tgt_lang}")
            
            # Truncate very long text to avoid memory issues
            if len(text) > 500:
                text = text[:500]
                print(Fore.YELLOW + "⚠️ Text truncated to 500 characters to prevent memory issues" + Style.RESET_ALL)
            
            # Translate the text
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated_tokens = model.generate(**inputs, max_length=512)
            translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            return translated_text
            
        except Exception as e:
            print(Fore.RED + f"❌ Translation error: {e}" + Style.RESET_ALL)
            return f"[Translation error: {str(e)}]"

# For backwards compatibility, keep the standalone function
def translate(text, src_lang="en", tgt_lang="hi"):
    """Translates text between English and Hindi."""
    # Create a temporary translator and use it
    translator = Translator()
    return translator.translate_text(text, src_lang, tgt_lang)

# Test translation
if __name__ == "__main__":
    try:
        print(Fore.MAGENTA + "\n===== RealTalk Translator =====" + Style.RESET_ALL)
        translator = Translator()
        
        # Test English to Hindi
        test_en = "Hello, how are you? I hope you are doing well today."
        print(Fore.CYAN + f"\nEnglish: {test_en}" + Style.RESET_ALL)
        result_hi = translator.translate_text(test_en, "en", "hi")
        print(Fore.GREEN + f"Hindi: {result_hi}" + Style.RESET_ALL)
        
        # Test Hindi to English
        test_hi = "नमस्ते, आप कैसे हैं? आशा है आज आप अच्छा महसूस कर रहे हैं।"
        print(Fore.CYAN + f"\nHindi: {test_hi}" + Style.RESET_ALL)
        result_en = translator.translate_text(test_hi, "hi", "en")
        print(Fore.GREEN + f"English: {result_en}" + Style.RESET_ALL)
        
    except Exception as e:
        print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)
        sys.exit(1)
