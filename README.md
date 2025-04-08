# 🧠 Voice-Based RAG Assistant using GPT-4o (Real-Time + Offline)

#### This project enhances OpenAI’s GPT-4o capabilities by adding a voice-driven Retrieval-Augmented Generation (RAG) mechanism. It allows users to ask questions via audio, retrieve relevant knowledge using semantic search, and receive rich, context-aware responses in both real-time and offline modes.:

  - Real-Time Mode: Uses WebSockets with OpenAI's GPT-4o for live conversations.
  - Offline Mode: Fully sequential pipeline that processes recorded audio and provides a spoken response.

## Purpose

Goal: Combine voice interaction with GPT-4o and external knowledge retrieval using Qdrant.

This system supports:

- Audio input (speech-based questions)
- Retrieval of relevant information via vector search
- Augmented text generation using GPT-4o + tools
- Audio output using OpenAI's TTS API

## Features

✅ Adds voice-based Retrieval-Augmented Generation (RAG) to both real-time and offline GPT-4o APIs
✅ Audio input via Whisper (speech-to-text)
✅ Contextual answers using Qdrant vector database
✅ Audio responses using TTS-1
✅ Supports OpenAI function calling


---

## Retrieval-Augmented Generation (RAG) with Qdrant

This assistant includes a search() tool powered by Qdrant vector search. It uses:

- text-embedding-3-large model to convert text into vectors
- Semantic search over a custom dataset
- Retrieved data is passed back into GPT-4o via tool calling for richer answers


---

## Real-Time Mode (WebSocket-Based)

What it does:

- Streams audio input (e.g., audio_english.ogg) to OpenAI’s GPT-4o real-time WebSocket API
- GPT-4o automatically transcribes the speech server-side using Whisper, applies turn detection, and parses out structured data
- When needed, GPT-4o generates a function call, such as search(text="..."), based on the transcribed user speech
- The client then executes the function (e.g., semantic search using Qdrant), sends back the result, and the assistant responds
- Final response is received as audio (streamed in chunks), which is saved to final_output.wav


---

## Offline Mode (Scripted Pipeline)

What it does:

- Accepts pre-recorded audio input
- Transcribes with Whisper
- GPT-4o processes input and dynamically calls the search() function if needed
- Context from Qdrant is used to enhance GPT’s response
- Final response is synthesized into speech (MP3 format)


