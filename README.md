# AiVoiceAssistant
AI Voice Assistant: Records, transcribes, processes text with a language model, and generates speech using Parler TTS. Real-time interaction with minimal latency. Built with Python and Transformers.
AI Voice Assistant - Speech-to-Speech Pipeline
This project implements an End-to-End AI Voice Assistant pipeline capable of processing user voice queries and responding with synthesized speech. The pipeline integrates various components, including Automatic Speech Recognition (ASR), a Language Model (LLM) for natural language processing, and Text-to-Speech (TTS) for generating vocal responses.

Features:
Voice Input: Records audio using the user's microphone.
Automatic Speech Recognition (ASR): Transcribes recorded audio into text using Google Speech Recognition.
Language Model (LLM): Processes the transcribed text and generates a coherent and relevant response.
Text-to-Speech (TTS): Converts the generated response text into speech using a pre-trained TTS model.
Real-time Processing: The entire pipeline operates in real-time, enabling interactive conversations with the AI assistant.
Technologies Used:
SpeechRecognition: Used for converting recorded audio into text (ASR).
Transformers: Provides the language model (LLM) used to generate text-based responses.
Parler TTS: A text-to-speech model that converts the generated response into spoken language.
Sounddevice & Scipy: Used for audio recording and processing.
Torch: Utilized for running models on GPU (if available) for faster processing.
How it Works:
Audio Recording: The assistant listens to the user’s voice and records a 5-second audio clip.
Speech-to-Text Conversion: The recorded audio is transcribed into text using Google’s Speech Recognition API.
Text Generation: The transcribed text is processed by a language model to generate a relevant response.
Text-to-Speech Conversion: The generated text response is converted into speech using the Parler TTS model.
Speech Output: The AI assistant responds with synthesized speech, completing the interaction.





