import pyaudio
import wave
import threading

class AudioHandler:
    def __init__(self):
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.fs = 16000  # Changed to 16kHz for wav2vec2 compatibility
        self.frames = []
        self.recording = False
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.stream = self.p.open(format=self.sample_format,
                                  channels=self.channels,
                                  rate=self.fs,
                                  frames_per_buffer=self.chunk,
                                  input=True)
        print("Recording started...")
        self.record_thread = threading.Thread(target=self._record_loop)
        self.record_thread.start()

    def _record_loop(self):
        while self.recording:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            self.frames.append(data)
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        print("Recording stopped.")
        self.save_audio()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.record_thread.join()
            print("Recording fully stopped.")

    def save_audio(self, filename="input.wav"):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"Saved as {filename}")

    def __del__(self):
        self.p.terminate()

if __name__ == "__main__":
    audio = AudioHandler()
    audio.start_recording()
    input("Press Enter to stop recording...")
    audio.stop_recording()