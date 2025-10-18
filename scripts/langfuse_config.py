"""
Configuración global de Langfuse para observability
"""
import os
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe

load_dotenv()

# Inicializar Langfuse
def init_langfuse():
    """Inicializa Langfuse con las credenciales del .env"""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if not public_key or not secret_key:
        print("⚠️  ADVERTENCIA: Credenciales de Langfuse no encontradas")
        print("   Langfuse estará deshabilitado para esta sesión")
        return False
    
    print(f"✅ Langfuse inicializado: {host}")
    return True

# Exportar decoradores para uso fácil
__all__ = ['init_langfuse', 'observe', 'langfuse_context']
