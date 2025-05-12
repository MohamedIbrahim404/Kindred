from llm import LLMHandler

llm = LLMHandler()
response = llm.generate_response("i’m feeling stressed")
print(f"Kindred: {response}")