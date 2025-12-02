import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("=" * 70)
print("Available Gemini Models:")
print("=" * 70)

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"✓ {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description[:100]}...")
        print()

print("=" * 70)
print("\nFor LangChain, use the model name WITHOUT 'models/' prefix")
print("Example: 'gemini-pro' instead of 'models/gemini-pro'")
print("=" * 70)
