import os
import json
import websocket
import soundfile as sf
import base64
import time
import threading
import openai
import numpy as np
import sounddevice as sd
from qdrant_client import QdrantClient
import librosa

os.environ["HTTP_PROXY"] = "http://172.16.56.101:1100"
os.environ["HTTPS_PROXY"] = "http://172.16.56.101:1100"

RATE = 16000 
CHUNK_SIZE = 4096

OPENAI_API_KEY = "..."

WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta": "realtime=v1"
}

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def search(text):
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""

    qdrant_client = QdrantClient(url="http://chatbot-qdrant.dev:80")
    query_vector = get_embedding(text=text)

    answer = qdrant_client.search(
        collection_name="english",
        query_vector=("question", query_vector),
        limit=10,
    )

    qa_pairs = [
        {"question": point.payload["question"][0], "answer": point.payload["answer"]}
        for point in answer
    ]

    os.environ["HTTP_PROXY"] = "http://172.16.56.101:1100"
    os.environ["HTTPS_PROXY"] = "http://172.16.56.101:1100"

    return qa_pairs


tools = [{
    "type": "function",
    "name": "search",
    "description": "Retrieve relevant extra information using Qdrant.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    }
}]

instructions = """You are a helpful assistant. You must help the user with their queries.
This is a chat between a user and an assistant.
User asks you some questions this company. You must answer them in English, using relevant information from the search function.
"""

ws_audio_buffer = []  
AUDIO_OUTPUT_PATH = "final_output.wav"  

def on_open(ws):
    print("WebSocket connection opened.")

    session_update_event = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": instructions,
            "voice": "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200
            },
            "temperature": 0.6,
            "tools": tools,
        }
    }
    ws.send(json.dumps(session_update_event))

    audio_file_path = "audio_english.ogg"  
    threading.Thread(target=send_audio_file, args=(ws, audio_file_path)).start()

def on_message(ws, message):
    global ws_audio_buffer
    event = json.loads(message)
    print("Received event:", json.dumps(event, indent=2))

    event_type = event.get("type")

    if event_type == "response.done":
        response_data = event.get("response", {}).get("output", [])
        function_call_event = None
        for item in response_data:
            if item.get("type") == "function_call":
                function_call_event = item

        if function_call_event:
            function_name = function_call_event.get("name")
            call_id = function_call_event.get("call_id")
            arguments = json.loads(function_call_event.get("arguments", "{}"))

            if function_name == "search":
                result = search(arguments["text"])

                function_response_event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"result": result})
                    }
                }
                ws.send(json.dumps(function_response_event))
                ws.send(json.dumps({"type": "response.create"}))
        else:
            if ws_audio_buffer:
                save_audio_file(ws_audio_buffer)
                print(f"Final audio response saved at: {AUDIO_OUTPUT_PATH}")
                ws_audio_buffer = []  
            else:
                print("Response sequence completed (text/audio). No function calls detected.")

    elif event_type == "response.audio.delta":
        audio_delta = event.get("delta")
        if audio_delta:
            audio_data = base64.b64decode(audio_delta)  
            ws_audio_buffer.append(audio_data)

    elif event_type == "error":
        error_message = event.get("error", {}).get("message", "Unknown error")
        print(f"Error: {error_message}")

def save_audio_file(audio_chunks, sample_rate=RATE):
    all_audio = b"".join(audio_chunks)
    # Convert bytes to NumPy array of int16
    audio_array = np.frombuffer(all_audio, dtype=np.int16)
    sf.write(AUDIO_OUTPUT_PATH, audio_array, sample_rate, subtype="PCM_16")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed.")

def convert_audio_to_16k_mono(input_file, output_file):
    audio, sr = librosa.load(input_file, sr=16000, mono=True)  
    sf.write(output_file, audio, 16000, subtype="PCM_16")  
    return output_file 

def send_audio_file(ws, file_path, sample_rate=RATE, chunk_size=CHUNK_SIZE):
    converted_file = "converted_audio.wav"

    # Convert the file if it isn't a WAV or has a different sample rate
    try:
        current_sr = librosa.get_samplerate(file_path)
    except Exception as e:
        print(f"Error reading samplerate: {e}")
        current_sr = None

    if not file_path.endswith(".wav") or (current_sr and current_sr != sample_rate):
        print(f"Converting {file_path} to 16kHz PCM16...")
        file_path = convert_audio_to_16k_mono(file_path, converted_file)

    print(f"Sending audio file: {file_path}")

    with sf.SoundFile(file_path, 'r') as audio_file:
        if audio_file.samplerate != sample_rate:
            raise ValueError(f"Sample rate mismatch! Expected {sample_rate}, got {audio_file.samplerate}")

        while True:
            data = audio_file.read(frames=chunk_size, dtype='int16')
            if len(data) == 0:
                break

            audio_bytes = data.tobytes()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            ws.send(json.dumps(audio_event))
            # Sleep based on the duration of the chunk
            time.sleep(len(data) / sample_rate)

    print("Audio file transmission completed.")

def main():
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        WEBSOCKET_URL,
        header=HEADERS,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

if __name__ == "__main__":
    main()

