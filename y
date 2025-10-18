import os
from dotenv import load_dotenv

load_dotenv()

def get_model():
    """
    Devuelve el cliente LLM correcto según la variable MODEL_DEFAULT.
    Soporta GPT-5 (OpenAI), Claude 3.5 (Anthropic) y Gemini 1.5 Ultra (Google).
    """
    model_name = os.getenv("MODEL_DEFAULT", "gpt-5").lower()
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.3"))

    if "gpt" in model_name:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature="1")

    elif "claude" in model_name or "anthropic" in model_name:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=temperature)

    elif "gemini" in model_name:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-ultra", temperature=temperature)

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


