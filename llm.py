from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
import emoji

class LLMHandler:
    def __init__(self, model_name="google/gemma-3-27b-it"):  # Updated to 27b
        # Quantization config for 4-bit to manage memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._warmup()

    def _warmup(self):
        prompt = "Warmup: Respond with 'Hello'."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model.generate(inputs["input_ids"], max_new_tokens=5)

    def generate_response(self, user_input, conversation_history=None):
        if conversation_history is None:
            conversation_history = []

        # Prompt optimized for Gemma
        prompt = (
            "You are Kindred, an AI therapist. Respond with empathy, understanding, and a supportive tone. "
            "Under no circumstances should you include emojis (e.g., 😊, ❤️, 👍) or informal symbols. "
            "Use only plain text. Provide a complete, concise response after 'Kindred:' and do not repeat the user's input or history. "
            "Encourage sharing more if appropriate. Here’s the conversation so far:\n\n"
        )
        if conversation_history:
            prompt += "\n".join(conversation_history) + "\n\n"
        prompt += f"User: {user_input}\nKindred:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Full model output: '{full_response}'")  # Debug

        # Extract only Kindred’s response
        if "Kindred:" in full_response:
            response = full_response.split("Kindred:")[-1].strip()
        else:
            response = full_response[len(prompt):].strip()

        response = re.sub(r"(User:.*|\n.*)", "", response).strip()
        if not response.endswith(('.', '?', '!')):
            response += "."

        response = emoji.replace_emoji(response, replace="")
        response = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF]', '', response)
        
        print(f"Processed response: '{response}'")  # Debug
        return response

if __name__ == "__main__":
    llm = LLMHandler()
    history = []
    response1 = llm.generate_response("Hi how are you doing", history)
    print(f"Kindred: {response1}")
    history.append("User: Hi how are you doing")
    history.append(f"Kindred: {response1}")
    response2 = llm.generate_response("that’s lovely to know! What can I do for you today?", history)
    print(f"Kindred: {response2}")