# Kindred Therapist AI

**Kindred** is an AI-powered emotional support tool that combines speech-to-text, large-language-model dialogue, and speech generation into a simple Flask web UI.

---

##  Requirements

- **Hardware:**
  - CUDA-compatible GPU (≥4 GB VRAM recommended)
- **OS & Language:**
  - Python 3.10+
  - Linux, macOS, or Windows 10+
- **Libraries & Tools:**
  - `torch` & `torchaudio`
  - `transformers`
  - `flask`
  - `pygame`
  - `soundfile`
- **Other:**
  - `ffmpeg` on PATH (for audio operations)
  - Access to Hugging Face models:
    - `sesame/csm-1b` (CSM speech generator)
    - `meta-llama/Llama-2-7b-chat-hf` (chat LLM)
    - `facebook/wav2vec2-base-960h` (speech-to-text)

---

## Setup

1. **Clone & enter repo**
   ```bash
   git clone https://github.com/MohamedIbrahim404/Kindred.git
   cd kindred-therapist-ai
   ```

2. **Create & activate virtual env**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
   ```

3. **Install Python deps**
   ```bash
   pip install -r requirements.txt
   ```

4. **Login to Hugging Face & pull models**
   ```bash
   huggingface-cli login
   ```

