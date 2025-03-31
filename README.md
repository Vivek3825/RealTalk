# RealTalk: Real-Time Speech Translation System

## Project Setup & Development Progress

### Step 1: Project Initialization 
1. Created the project folder structure:  

   RealTalk/
   ├── src/
   │   ├── speech_recognition/  # Speech-to-text module
   │   ├── translation/         # Translation module
   │   ├── text_to_speech/      # Text-to-speech module
   │   └── utils/               # Helper functions
   ├── models/                  # Stores downloaded models
   ├── data/                    # Any sample test data
   ├── tests/                   # Unit & integration tests
   ├── realtalk_env/            # Virtual environment
   ├── requirements.txt         # Dependencies list
   ├── .gitignore               # Ignore unnecessary files
   └── README.md                # Documentation

2. Initialized a **virtual environment** (`realtalk_env`).  

   python -m venv realtalk_env

3. Activated the virtual environment:  

     realtalk_env\Scripts\activate

4. Installed required dependencies:  

   pip install -r requirements.txt


### Step 2: Implemented Speech Recognition Module  
1. Used Vosk for speech-to-text conversion.  
2. Implemented noise calibration**, adaptive thresholds, and spectral subtraction for better recognition.  
3. Added dynamic voice activity detection (VAD).  
4. Created `speech_recognizer.py` inside `src/speech_recognition/`.  
5. Successfully tested real-time speech recognition for **English & Hindi**.


### Step 3: Implemented Translation Module  
1. Used MarianMT (fast & efficient) and IndicTrans (for Hindi-English).  
2. Added bidirectional translation (English ↔ Hindi).  
3. Created `translator.py` inside `src/translation/`.  
4. Successfully tested translation:  
   English to Hindi: हैलो, तुम कैसे हो?
   Hindi to English: Hello, how are you?


