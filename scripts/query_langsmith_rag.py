#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consulta el vector store con embeddings de LangSmith (RAG básico).
Recupera los k documentos más similares y genera una respuesta
combinando ese contexto con el modelo actual definido en MODEL_DEFAULT.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import HumanMessage, SystemMessage
from src.agents.utils.model_factory import get_model

# -----------------------------
# Configuración base
# -----------------------------
load_dotenv()

VECTOR_DIR = "data/vectorstore/langsmith_chroma"
K = 3  # cantidad de fragmentos recuperados

# -----------------------------
# Función principal
# -----------------------------
def query_rag(question: str):
    # 1. Cargar embeddings y vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)

    # 2. Recuperar fragmentos relevantes
    docs = db.similarity_search(question, k=K)
    if not docs:
        print("No se encontraron resultados relevantes.")
        return

    print(f"Fragmentos recuperados: {len(docs)}")
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    # 3. Construir prompt con contexto
    system_prompt = (
        "Eres un analista de datos senior de RocknBlock. "
        "Usa el contexto de ejecuciones previas para responder de forma clara, "
        "razonada y ejecutiva. Si el contexto no contiene la respuesta, indícalo."
    )
    user_prompt = f"Pregunta: {question}\n\nContexto recuperado:\n{context}"

    # 4. Llamar al modelo seleccionado en model_factory
    llm = get_model()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)

    print("\nRespuesta generada:\n")
    print(response.content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 scripts/query_langsmith_rag.py 'tu pregunta'")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    query_rag(query)
