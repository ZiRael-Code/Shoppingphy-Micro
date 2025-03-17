from flask import Flask, request, jsonify
from flask_cors import CORS
from vosk import Model, KaldiRecognizer
from textblob import TextBlob
import wave
import json
import os
import subprocess
import tempfile

app = Flask(__name__)
CORS(app)  # Allow all origins

# Load Vosk model
print("Loading Vosk model...")
model = Model("model")
print("Vosk model loaded successfully!")

def convert_webm_to_wav(input_bytes):
    """ Convert WebM audio bytes to WAV format using FFmpeg """
    print("Received WebM audio for conversion...")

    # Create temporary files for WebM and WAV
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
        temp_webm.write(input_bytes)
        temp_webm_path = temp_webm.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name

    try:
        print(f"Converting WebM to WAV using FFmpeg... ({temp_webm_path} -> {temp_wav_path})")

        command = [
            "ffmpeg", "-y",  # Force overwrite
            "-i", temp_webm_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            temp_wav_path
        ]

        # Run FFmpeg
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print("❌ FFmpeg conversion failed:", result.stderr)
            raise Exception(f"FFmpeg Error: {result.stderr}")

        print("✅ FFmpeg conversion successful!")

        # Read the WAV file as bytes
        with open(temp_wav_path, "rb") as f:
            wav_bytes = f.read()

    finally:
        # Clean up temp files
        os.remove(temp_webm_path)

    return wav_bytes, temp_wav_path  # Return path for further processing

def transcribe_audio(wav_path):
    """ Transcribe audio using Vosk """
    print("Starting transcription process...")

    try:
        with wave.open(wav_path, "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate())

            print("Processing audio with Vosk...")
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

            result = json.loads(rec.Result())  # Convert result to JSON object
            print(f"Transcription result: {result}")

    finally:
        # Close and delete the temp WAV file properly
        os.remove(wav_path)

    return result.get("text", "")  # Return only transcribed text
def extract_data(text):
    """Extract structured data from text using TextBlob."""
    print("Extracting structured data from text...")

    try:
        blob = TextBlob(text)
        words = blob.words
    except Exception as e:
        print(f"❌ Error in TextBlob processing: {e}")
        return {"error": "Text processing failed. Ensure TextBlob corpora are installed."}

    data = {}
    colors = ["black", "white", "red", "blue"]
    sizes = ["small", "medium", "large"]

    for word in words:
        if word.isdigit() and int(word) > 1900:
            data["year"] = int(word)
        elif word.lower() in colors:
            data["color"] = word.lower()
        elif word.lower() in sizes:
            data["size"] = word.lower()
        else:
            data["description"] = word.lower()

    print(f"✅ Extracted structured data: {data}")
    return data

@app.route('/process_audio', methods=['POST'])
def process_audio():
    print("Received API request to process audio...")

    if "file" not in request.files:
        print("No file uploaded!")
        return jsonify({"error": "No file uploaded"}), 400

    webm_file = request.files["file"].read()
    print(f"File received. Size: {len(webm_file)} bytes")

    try:
        # Convert WebM to WAV
        wav_bytes, wav_path = convert_webm_to_wav(webm_file)

        # Transcribe the WAV audio
        transcription_text = transcribe_audio(wav_path)

        # Extract structured data
        # structured_data = extract_data(transcription_text)

        print("Processing completed successfully!")
        return jsonify({"text": transcription_text})

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
