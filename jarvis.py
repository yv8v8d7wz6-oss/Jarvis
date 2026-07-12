import argparse
import io
import os
import uuid

import anthropic
import requests
import speech_recognition as sr
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your_claude_api_key"))

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "your_elevenlabs_api_key")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "your_chosen_voice_id")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")  # used only for Whisper speech-to-text (CLI mode)

tools = [
    {
        "name": "google_maps_search",
        "description": "Search for places, restaurants, or coordinates using Google Maps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search term, e.g., 'coffee near Central Park'"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_to_memory",
        "description": "Save a new fact learned about the user to long-term memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The specific preference or fact to remember."}
            },
            "required": ["fact"]
        }
    }
]

system_prompt = """
You are Jarvis, an AI assistant with access to Google Maps and long-term memory.
If the user tells you something personal or a preference, use 'save_to_memory' to remember it.
Always confirm with the user what you saved.
Keep spoken responses concise and natural — they will be read aloud.
"""


def call_google_maps(query: str) -> str:
    # Replace with a real Google Maps API call (requests.get(...))
    return f"Mock result: found 3 places matching '{query}'"


def save_fact(fact: str) -> str:
    # Replace with a real DB write
    print(f"[Saved to memory: {fact}]")
    return "Fact saved successfully."


def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "google_maps_search":
        return call_google_maps(tool_input["query"])
    elif tool_name == "save_to_memory":
        return save_fact(tool_input["fact"])
    return f"Unknown tool: {tool_name}"


def speak(text: str) -> str:
    """Send text to ElevenLabs TTS and save the audio to a uniquely named file.
    Returns the path to the saved file."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.75},
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"TTS failed: {response.status_code} {response.text}")

    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(AUDIO_DIR, filename)
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def play_audio_local(path: str):
    """CLI-only: play a generated file through local speakers. Requires 'playsound'."""
    try:
        from playsound import playsound
        playsound(path)
    except ImportError:
        print(f"[Audio saved to {path} — install 'playsound' to auto-play]")


def listen(timeout: int = 8, phrase_time_limit: int = 15) -> str:
    """CLI-only: record from the default microphone and transcribe with OpenAI Whisper."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("[Listening...]")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return ""

    wav_bytes = audio.get_wav_data()

    response = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={"file": ("speech.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={"model": "whisper-1"},
    )

    if response.status_code != 200:
        raise RuntimeError(f"Transcription failed: {response.status_code} {response.text}")

    text = response.json().get("text", "").strip()
    print(f"[Heard: {text}]")
    return text


def run_jarvis(user_message: str) -> str:
    """Send a message through Claude, handling any tool calls, and return the final text reply."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final_text

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Web server: serves the HUD page and a chat API the browser mic talks to
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "jarvis.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        reply_text = run_jarvis(user_message)
    except Exception as e:
        return jsonify({"error": f"Jarvis failed to respond: {e}"}), 500

    audio_url = None
    try:
        audio_path = speak(reply_text)
        audio_url = f"/static/audio/{os.path.basename(audio_path)}"
    except Exception as e:
        # Text reply still works even if TTS fails (e.g. bad ElevenLabs key)
        print(f"[TTS failed: {e}]")

    return jsonify({"reply": reply_text, "audio_url": audio_url})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cli", action="store_true",
        help="Run in terminal voice mode (mic + speakers) instead of starting the web server"
    )
    args = parser.parse_args()

    if args.cli:
        print("Jarvis is ready (CLI mode). Say 'quit' or 'exit' to stop.")
        while True:
            user_message = listen()
            if not user_message:
                continue
            if user_message.strip().lower() in ("quit", "exit"):
                break
            answer = run_jarvis(user_message)
            print(f"Jarvis: {answer}")
            audio_path = speak(answer)
            play_audio_local(audio_path)
    else:
        port = int(os.environ.get("PORT", 5000))
        print(f"Starting Jarvis web server on port {port}")
        app.run(host="0.0.0.0", port=port)
