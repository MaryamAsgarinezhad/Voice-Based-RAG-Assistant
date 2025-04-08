from openai import OpenAI
import json
import requests
import sounddevice as sd
import numpy as np
import wave
import tempfile
from qdrant_client import QdrantClient
import os 

os.environ["HTTP_PROXY"] = "http://172.16.56.101:1100"
os.environ["HTTPS_PROXY"] = "http://172.16.56.101:1100"

api_key = '...'

client = OpenAI(api_key=api_key)

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def search(text, history=[]):
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""

    qdreantClient = QdrantClient(url="http://chatbot-qdrant.dev:80")
    query_vector = get_embedding(text=text)

    answer = qdreantClient.search(
        collection_name="english",
        query_vector=("question", query_vector),
        limit= 10,
    )

    qa_pairs = [
        {"question": point.payload["question"][0], "answer": point.payload["answer"]}
        for point in answer
    ]
  
    os.environ["HTTP_PROXY"] = "http://172.16.56.101:1100"
    os.environ["HTTPS_PROXY"] = "http://172.16.56.101:1100"

    print(qa_pairs)
    return qa_pairs

tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Always look for relative extra information by calling search function, to incorporate its output as a context to generate response (RAG).",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

def chat_with_gpt(user_query):
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages= messages,
    tools=tools,
    )

    tool_call = completion.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    print(args["text"])

    history = []
    result = search(args["text"], history)

    messages.append(completion.choices[0].message)  # append model's function call message
    messages.append({                               # append result message
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    })

    completion_2 = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    return completion_2.choices[0].message.content

import openai
openai.api_key = '...'

def transcribe_audio(audio_file):
    with open(audio_file, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

import openai

def text_to_speech(text):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    with open("response.mp3", "wb") as f:
        f.write(response.content)
    print("Response saved as 'response.mp3'. Play it using an audio player.")

audio_file = input("Enter the path to your audio file: ").strip()

user_query = transcribe_audio(audio_file)
print("Transcribed text:", user_query)

response_text = chat_with_gpt(user_query)
print("ChatGPT Response:", response_text)

text_to_speech(response_text)

