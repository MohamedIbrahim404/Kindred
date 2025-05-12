from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

class SpeechToText:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        # Specify tokenizer_class to suppress FutureWarning
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, tokenizer_class="Wav2Vec2CTCTokenizer")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def audio_to_text(self, audio_file="input.wav"):
        audio_input, sample_rate = sf.read(audio_file)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16kHz, got {sample_rate}Hz.")
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.lower()

if __name__ == "__main__":
    stt = SpeechToText()
    text = stt.audio_to_text("input.wav")
    print(f"Transcribed text: {text}")