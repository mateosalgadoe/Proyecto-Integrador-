#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crea un vector store básico (Chroma) con los runs de LangSmith exportados.
Usa los campos: question, plan.rationale y final_answer.
"""

import os, json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

load_dotenv()

# Ruta al JSON exportado
INPUT_FILE = "data/processed/langsmith_runs.json"
DB_DIR = "data/vectorstore/langsmith_chroma"

def load_runs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_docs(runs):
    docs = []
    for trace in runs:
        steps = trace.get("steps", [])
        for s in steps:
            inputs = s.get("inputs") or {}
            outputs = s.get("outputs") or {}
            plan = outputs.get("plan") or inputs.get("plan") or {}

            question = inputs.get("question", "")
            answer = outputs.get("final_answer") or outputs.get("draft_answer") or outputs.get("tool_result") or ""
            rationale = plan.get("rationale", "")

            if not question and not answer:
                continue

            content = f"Pregunta: {question}\nRazonamiento: {rationale}\nRespuesta: {answer}"
            meta = {
                "trace_id": s.get("trace_id"),
                "name": s.get("name"),
                "run_type": s.get("run_type"),
                "start_time": s.get("start_time"),
                "end_time": s.get("end_time"),
            }
            docs.append(Document(page_content=content, metadata=meta))
    return docs

def main():
    runs = load_runs(INPUT_FILE)
    docs = build_docs(runs)
    print(f"Total documentos: {len(docs)}")

    if not docs:
        print("XX No hay datos para indexar.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=DB_DIR)
    db.persist()
    print(f"Vector store creado en {DB_DIR}")

if __name__ == "__main__":
    main()
