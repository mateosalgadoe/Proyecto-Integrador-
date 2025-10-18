# test_models.py
import os
from dotenv import load_dotenv

load_dotenv()

# Test Gemini
print("=== TESTING GEMINI ===")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    models = genai.list_models()
    print("Modelos Gemini disponibles:")
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")
except Exception as e:
    print(f"Error Gemini: {e}")

# Test Claude
print("\n=== TESTING CLAUDE ===")
try:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # Intentar con diferentes modelos
    test_models = [
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229"
    ]
    for model in test_models:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            print(f"  ✓ {model} funciona")
            break
        except Exception as e:
            print(f"  ✗ {model}: {str(e)[:50]}")
except Exception as e:
    print(f"Error Claude general: {e}")
