import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
import torch
from transformers import pipeline, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf

# Recording parameters
DURATION = 5  # seconds
SAMPLE_RATE = 16000

# Function to record audio
def record_audio(filename):
    print("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    wav.write(filename, SAMPLE_RATE, audio)
    print(f"Audio saved as {filename}")

# Function to transcribe audio to text using Google Speech Recognition
def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
        print("Transcribing...")
        try:
            text = recognizer.recognize_google(audio)
            print("Transcription complete.")
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

# Function to generate a response using a language model (LLM)
def generate_response(transcription):
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    messages = [{"role": "user", "content": transcription}]
    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response

# Function to convert text to speech using Parler TTS
def text_to_speech(assistant_response):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

    description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch."
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(assistant_response, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
    print("Speech synthesis complete. Output saved as 'parler_tts_out.wav'")

# Main function to run the full pipeline
def main():
    audio_file = 'recorded_audio.wav'
    # Record audio
    record_audio(audio_file)
    # Transcribe the audio
    transcription = transcribe_audio(audio_file)
    print(f"Transcribed text: {transcription}")
    # Generate a response based on the transcription
    assistant_response = generate_response(transcription)
    print(f"Generated response: {assistant_response}")
    # Convert the response text to speech
    text_to_speech(assistant_response)

if __name__ == "__main__":
    main()
