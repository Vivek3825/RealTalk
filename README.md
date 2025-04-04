# RealTalk: Real-Time Speech Translation System

## Project Setup & Development Progress

### Step 1: Project Initialization 
1. Created the project folder structure:  

The folder structure for the "RealTalk" project is organized as follows: The src folder contains various modules, including speech recognition for speech-to-text functionality, translation for language processing, text-to-speech for synthesizing audio from text, and utils for helper functions. The models folder stores any downloaded models, while the data folder holds sample test data. The tests folder is dedicated to unit and integration tests. The realtalk_env folder represents the virtual environment for the project. The requirements.txt file lists the project's dependencies, and the .gitignore file specifies files and directories to be ignored by Git. Finally, the README.md file serves as the project's documentation. All components are structured to ensure efficient organization and functionality.

2. Initialized a **virtual environment** (`realtalk_env`).  
   python -m venv realtalk_env

3. Activated the virtual environment:  
     realtalk_env\Scripts\activate

4. Installed required dependencies:  
   pip install -r requirements.txt


### Step 2: Implemented Speech Recognition Module  
1. Used Vosk for speech-to-text conversion.  
2. Implemented noise calibration, adaptive thresholds, and spectral subtraction for better recognition.  
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

### Step 4: Speech Recognition + Translation Integration
1. Modified `speech_recognizer.py` & `translator.py` files
2. created `speech_translation.py` where `speech_recognizer.py` & `translator.py` are integrated.









