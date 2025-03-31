from transformers import MarianMTModel, MarianTokenizer

class BidirectionalTranslator:
    def __init__(self):
        """Initialize MarianMT models for Hindi ↔ English translation"""
        self.models = {
            "hi-en": "Helsinki-NLP/opus-mt-hi-en",
            "en-hi": "Helsinki-NLP/opus-mt-en-hi"
        }

        # Load models and tokenizers
        self.tokenizers = {pair: MarianTokenizer.from_pretrained(model) for pair, model in self.models.items()}
        self.models = {pair: MarianMTModel.from_pretrained(model) for pair, model in self.models.items()}

    def detect_language(self, text):
        """Basic language detection (Can be improved with a proper model)"""
        hindi_chars = set("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
        if any(char in hindi_chars for char in text):
            return "hi"
        return "en"

    def translate(self, text):
        """Automatically detect the language and translate"""
        src_lang = self.detect_language(text)
        if src_lang == "hi":
            model_key = "hi-en"
        else:
            model_key = "en-hi"

        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]

        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs) #type: ignore
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        return translated_text[0]

# Example usage
translator = BidirectionalTranslator()

print("English to Hindi:", translator.translate("Hello, how are you?"))
print("Hindi to English:", translator.translate("नमस्ते, आप कैसे हैं?"))
