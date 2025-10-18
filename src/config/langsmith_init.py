import os
from langsmith import Client

def init_langsmith():
    """Inicializa el cliente LangSmith y verifica conexión."""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    if not api_key or not project:
        print("⚠️Falta configurar LANGCHAIN_API_KEY o LANGCHAIN_PROJECT en el .env")
        return None

    try:
        client = Client(api_key=api_key, api_url=endpoint)
        print(f" LangSmith inicializado correctamente. Proyecto activo: {project}")
        return client
    except Exception as e:
        print(f"XX Error inicializando LangSmith: {e}")
        return None
