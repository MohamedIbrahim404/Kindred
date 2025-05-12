import torch
import torchaudio
from pygame import mixer
import os
import sys
import time
from datetime import datetime

# Add CSM directory to path
CSM_DIR = os.path.join(os.path.dirname(__file__), "csm")
if CSM_DIR not in sys.path:
    sys.path.append(CSM_DIR)

try:
    from generator import load_csm_1b
except ImportError:
    raise ImportError("Could not import load_csm_1b. Ensure the 'csm' repo is cloned and dependencies are installed.")

class TTSHandler:
    def __init__(self, model_name="sesame/csm-1b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = load_csm_1b(device=self.device)
        self.sample_rate = self.generator.sample_rate  # Typically 24kHz
        # Initialize mixer once with fixed settings
        mixer.init(frequency=self.sample_rate, size=-16, channels=1)
        self.speaker_id = 1  # Fixed speaker ID for consistency
        self.context = []    # Fixed context (empty for now)
        self.cached_end = "end_session.wav"
        self._cache_end_message()
        self.responses_dir = "responses"
        os.makedirs(self.responses_dir, exist_ok=True)

    def _cache_end_message(self):
        text = "Thank you for sharing today. I’m here whenever you need me."
        audio = self.generator.generate(
            text=text,
            speaker=self.speaker_id,  # Use fixed speaker
            context=self.context,     # Use fixed context
            max_audio_length_ms=15000
        )
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save(self.cached_end, audio.cpu(), self.sample_rate, bits_per_sample=16)
        print(f"Cached end-session message as {self.cached_end}")

    def text_to_speech(self, text, output_file="response.wav"):
        word_count = len(text.split())
        estimated_seconds = max(10, word_count / 2.5)
        max_audio_length_ms = int(estimated_seconds * 1000)
        print(f"Text: '{text}'")
        print(f"Word count: {word_count}, Estimated duration: {estimated_seconds}s, Max length: {max_audio_length_ms}ms")

        # Generate audio with consistent settings
        audio = self.generator.generate(
            text=text,
            speaker=self.speaker_id,  # Same speaker as cached message
            context=self.context,     # Same context
            max_audio_length_ms=max_audio_length_ms
        )
        audio = audio.cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        torchaudio.save(output_file, audio, self.sample_rate, bits_per_sample=16)

        info = torchaudio.info(output_file)
        print(f"Generated WAV: sample_rate={info.sample_rate}, num_channels={info.num_channels}, bits_per_sample={info.bits_per_sample}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"response_{timestamp}.wav"
        new_filepath = os.path.join(self.responses_dir, new_filename)

        os.rename(output_file, new_filepath)
        print(f"Moved {output_file} to {new_filepath}")

        time.sleep(0.1)
        mixer.music.load(new_filepath)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)

        return new_filepath

    def play_cached_end(self):
        mixer.music.load(self.cached_end)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)

    def __del__(self):
        mixer.quit()