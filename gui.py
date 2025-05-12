import tkinter as tk
from tkinter import messagebox
import threading
from audio_handler import AudioHandler
from stt import SpeechToText
from llm import LLMHandler
from tts import TTSHandler

class KindredGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kindred - AI Therapist")
        self.root.geometry("400x300")

        self.audio = AudioHandler()
        self.stt = None
        self.llm = None
        self.tts = None
        self.models_loaded = False
        self.processing = False
        self.conversation_history = []

        self.start_btn = tk.Button(root, text="Start Session", command=self.start_session)
        self.start_btn.pack(pady=10)

        self.record_btn = tk.Button(root, text="Record", command=self.record, state="disabled")
        self.record_btn.pack(pady=10)

        self.stop_btn = tk.Button(root, text="Stop Recording", command=self.stop_recording, state="disabled")
        self.stop_btn.pack(pady=10)

        self.end_btn = tk.Button(root, text="End Session", command=self.end_session, state="disabled")
        self.end_btn.pack(pady=10)

        self.clear_history_btn = tk.Button(root, text="Clear History", command=self.clear_history, state="disabled")
        self.clear_history_btn.pack(pady=10)

        self.text_area = tk.Text(root, height=20, width=60)
        self.text_area.pack(pady=10)

        threading.Thread(target=self._load_models, daemon=True).start()

    def start_session(self):
        self.text_area.insert(tk.END, "Session started.\n")
        self.start_btn.config(state="disabled")
        self.record_btn.config(state="normal")
        self.end_btn.config(state="normal")
        self.clear_history_btn.config(state="normal")
        self.conversation_history.clear()
        self._update_text(f"History reset. Current history: {self.conversation_history}\n")
        # Reinitialize LLM to ensure fresh state
        if self.llm:
            self.llm = LLMHandler()
            self._update_text("LLM reinitialized for new session.\n")

    def _load_models(self):
        try:
            if not self.stt:
                self.stt = SpeechToText()
                self._update_text("Speech-to-text model loaded.\n")
            if not self.llm:
                self.llm = LLMHandler()
                self._update_text("Language model loaded.\n")
            if not self.tts:
                self.tts = TTSHandler()
                self._update_text("Text-to-speech model loaded.\n")
            self.models_loaded = True
            self._update_text("All models loaded. Ready to record.\n")
        except Exception as e:
            self._update_text(f"Error loading models: {e}\n")
            self.models_loaded = False

    def _update_text(self, message):
        self.text_area.insert(tk.END, message)
        self.text_area.see(tk.END)
        self.root.update_idletasks()

    def record(self):
        if self.processing:
            return
        self._update_text("Recording...\n")
        self.record_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.record_thread = threading.Thread(target=self.audio.start_recording)
        self.record_thread.start()

    def stop_recording(self):
        if self.processing:
            return
        self._update_text("Recording stopped.\n")
        self.stop_btn.config(state="disabled")
        self.record_btn.config(state="normal")
        self.audio.stop_recording()
        self.record_thread.join()

        if not self.models_loaded:
            self._update_text("Waiting for models to load...\n")
            return

        self.processing = True
        self._update_text("Processing your input...\n")
        threading.Thread(target=self._process_input, daemon=True).start()

    def _process_input(self):
        try:
            self._update_text("Transcribing audio...\n")
            text = self.stt.audio_to_text("input.wav")
            self._update_text(f"You said: {text}\n")

            self._update_text("Generating response...\n")
            self._update_text(f"Current history before response: {self.conversation_history}\n")  # Debug
            response = self.llm.generate_response(text, self.conversation_history)
            self._update_text(f"Kindred: {response}\n")

            if len(self.conversation_history) >= 4:
                self.conversation_history = self.conversation_history[-4:]
            self.conversation_history.append(f"User: {text}")
            self.conversation_history.append(f"Kindred: {response}")

            self._update_text("Converting to speech...\n")
            saved_file = self.tts.text_to_speech(response)
            self._update_text(f"Response spoken successfully. Saved as: {saved_file}\n")
        except Exception as e:
            self._update_text(f"Error: {e}\n")
        finally:
            self.processing = False
            self.record_btn.config(state="normal")

    def end_session(self):
        if self.processing:
            self._update_text("Please wait for processing to finish.\n")
            return
        self._update_text("Session ended. Thank you for sharing.\n")
        if self.tts and self.models_loaded:
            self.tts.play_cached_end()
        self.start_btn.config(state="normal")
        self.record_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.end_btn.config(state="disabled")
        self.clear_history_btn.config(state="disabled")
        if self.audio.recording:
            self.audio.stop_recording()

    def clear_history(self):
        self.conversation_history.clear()
        self._update_text("Conversation history cleared.\n")

    def __del__(self):
        if self.audio.recording:
            self.audio.stop_recording()

if __name__ == "__main__":
    root = tk.Tk()
    app = KindredGUI(root)
    root.mainloop()