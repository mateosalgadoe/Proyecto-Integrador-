import os
from dotenv import load_dotenv

load_dotenv()

def get_model():
    """
    Devuelve el cliente LLM según MODEL_DEFAULT.
    Soporta: GPT-5 (OpenAI), Claude 4 Sonnet (Anthropic), Gemini 2.5 Pro (Google).
    """
    model_name = os.getenv("MODEL_DEFAULT", "gpt-5").lower().strip()
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.3"))

    # === GPT-5 ===
    if "gpt" in model_name:
        from langchain_openai import ChatOpenAI
        temp = max(temperature, 1.0)  # GPT-5 requiere ≥1
        return ChatOpenAI(model="gpt-5", temperature=temp)

    # === CLAUDE 4 SONNET ===
    elif "claude" in model_name or "anthropic" in model_name:
        from langchain_anthropic import ChatAnthropic
        
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",  # Claude 4 Sonnet
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=8192,
        )

    # === GEMINI 2.5 PRO ===
    elif "gemini" in model_name or "google" in model_name:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    else:
        raise ValueError(f"Modelo desconocido o no soportado: {model_name}")
