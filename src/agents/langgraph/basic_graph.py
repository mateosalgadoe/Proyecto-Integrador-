from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from typing import TypedDict
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Estado que viaja por el grafo
class AgentState(TypedDict):
    question: str
    context: dict
    answer: str

@traceable(run_type="tool", name="Load Context")
def load_context_node(state: AgentState) -> AgentState:
    from src.semantic.load_context import load_context
    state["context"] = load_context()
    return state

@traceable(run_type="llm", name="LLM Node")
# Nodo que responde usando el LLM
def llm_node(state: AgentState) -> AgentState:
    from src.agents.utils.model_factory import get_model
    llm = get_model()
    # Prompt sencillo con contexto
    prompt = f"""
    Eres un analista experto en datos de RocknBlock.
    Pregunta: {state['question']}
    Contexto de negocio (tablas y columnas): {state['context']}
    Responde de forma clara y en español.
    """

    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state

# Construcción del grafo
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_context", load_context_node)
    graph.add_node("llm", llm_node)

    # flujo: primero contexto → luego LLM → fin
    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "llm")
    graph.add_edge("llm", END)

    return graph.compile()

if __name__ == "__main__":
    workflow = build_graph()

    # Ejemplo: correr una pregunta
    state = {"question": "¿Cuántas ventas hay registradas en fct_sales?", "context": {}, "answer": ""}
    result = workflow.invoke(state)
    print(result["answer"])
