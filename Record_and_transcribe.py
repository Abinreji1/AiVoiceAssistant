import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav

# Recording parameters
DURATION = 5  # seconds
SAMPLE_RATE = 16000

def record_audio(filename):
    print("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    wav.write(filename, SAMPLE_RATE, audio)
    print(f"Audio saved as {filename}")

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

def main():
    audio_file = 'recorded_audio.wav'
    record_audio(audio_file)
    transcription = transcribe_audio(audio_file)
    print(f"Transcribed text: {transcription}")

if __name__ == "__main__":
    main()
