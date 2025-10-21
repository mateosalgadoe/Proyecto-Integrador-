import os, json, re
from typing import TypedDict, Optional, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from src.agents.utils.model_factory import get_model

# Cargar .env
load_dotenv()

# Importamos el grafo SQL como "herramienta"
from src.agents.langgraph.sql_graph import build_graph as build_sql_graph
from src.config.langsmith_init import init_langsmith
from langsmith import traceable
# Inicializar LangSmith
init_langsmith()


# ----------------------------
# Estado del grafo ReAct
# ----------------------------
class ReActState(TypedDict):
    question: str
    business_context: str
    plan: Optional[dict]
    tool_result: Optional[str]
    draft_answer: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]
    retrieved_context: Optional[str]
    kpi_numeric_value: Optional[float]

# ----------------------------
# Config cierre estándar (sin firma)
# ----------------------------
STANDARD_CLOSING = (
    "Espero que esta información sea útil para la toma de decisiones. "
    "Quedo a disposición para profundizar o apoyar en el diseño de acciones específicas."
)

SIGNATURE_PATTERNS = [
    r"^\s*saludos( cordiales)?\s*,?\s*$",
    r"^\s*atentamente\s*,?\s*$",
    r"^\s*cordialmente\s*,?\s*$",
    r"^\s*\[?tu nombre\]?\s*$",
    r"^\s*analista de datos( senior)?\s*$",
    r"^\s*rocknblock\s*$",
    r"^\s*atte\.?\s*$",
    r"^\s*firma:.*$",
]
SIGNATURE_REGEX = re.compile("|".join(SIGNATURE_PATTERNS), flags=re.IGNORECASE | re.MULTILINE)

def strip_signature(text: str) -> str:
    clean = SIGNATURE_REGEX.sub("", text).strip()
    clean = re.sub(r"\n{2,}$", "\n", clean).strip()
    return clean

def ensure_closing(text: str) -> str:
    if STANDARD_CLOSING.lower() not in text.lower():
        if not text.endswith((".", "!", "?")):
            text += "."
        text += "\n\n" + STANDARD_CLOSING
    return text


# ----------------------------
# Modelo
# ----------------------------
def get_llm():
    from src.agents.utils.model_factory import get_model
    return get_model()


# ----------------------------
# Nodo previo: Industry Context
# ----------------------------
@traceable(run_type="tool", name="Industry Context")
def industry_context_node(state: ReActState) -> ReActState:
    from src.agents.langgraph.industry_context import get_industry_context
    try:
        industry_info = get_industry_context()
        prev_context = state.get("business_context", "")
        state["business_context"] = f"{prev_context}\n\n[Contexto de industria actualizado]\n{industry_info}"
        return state
    except Exception as e:
        state["business_context"] += f"\nxx Error obteniendo contexto de industria: {e}"
        return state


# ----------------------------
# Nodo nuevo: Retrieval (RAG LangSmith)
# ----------------------------
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

VECTOR_DIR = "data/vectorstore/langsmith_chroma"

@traceable(run_type="tool", name="Retrieval Node")
def retrieval_node(state: ReActState) -> ReActState:
    """Busca fragmentos relevantes desde el vector store LangSmith y los agrega al contexto."""
    try:
        question = state["question"]
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
        docs = db.similarity_search(question, k=3)
        if not docs:
            state["retrieved_context"] = "No se encontraron fragmentos relevantes en el vector store."
        else:
            joined = "\n\n---\n\n".join([d.page_content for d in docs])
            state["retrieved_context"] = joined
            print(f"Fragmentos recuperados: {len(docs)}")
        return state
    except Exception as e:
        state["retrieved_context"] = f"Error en retrieval: {e}"
        return state


# ----------------------------
# Nodo 1: Plan (Reason)
# ----------------------------

@traceable(run_type="chain", name="Plan Node")
def plan_node(state: ReActState) -> ReActState:
    llm = get_llm()
    system = (
        "Eres un analista senior de RocknBlock (landscaping). "
        "Debes planificar la mejor acción para responder la pregunta con datos reales, "
        "usando las fuentes clean.* (ventas, leads, jobs) o ai.* (KPIs semánticos)."
    )
    context = (
        state.get("business_context", "")
        + "\n\n[Contexto recuperado]\n"
        + state.get("retrieved_context", "")
    )

    prompt = f"""
[Sistema]
{system}

[Contexto combinado]
{context}

[Instrucciones de planificación]
Analiza la pregunta del usuario y devuelve un JSON con:
- action: "sql_query", "kpi_query" o "direct_answer".
- rationale: breve justificación.
- reformulated_question: versión optimizada de la pregunta para ejecutar si aplica.

[Guías de decisión MEJORADAS]
- Usa "kpi_query" si la pregunta menciona:
  * Palabras como "KPI", "indicador", "métrica", "porcentaje", "ratio", "tasa", "volumen promedio", "eficiencia", "promedio de ventas"
  * "benchmark", "benchmarks", "industria", "sector", "comparativa", "mercado", "industria de landscaping", "PIB", "crecimiento"  ← 🆕 AÑADIR "PIB", "crecimiento"
  * "tasa de cierre", "closing rate", "conversión", "win rate"
  * "tendencia", "evolución", "serie temporal" (SOLO si menciona KPI o métricas específicas)
- Usa "sql_query" si la pregunta pide:
  * Datos crudos, conteos, listados o exploraciones directas
  * Tendencias temporales sin mencionar KPIs (ej: "ventas por mes")
- Usa "direct_answer" si es conceptual o fuera del dominio de datos.
- **Si menciona "benchmark", "industria", "sector", "PIB" o "crecimiento", SIEMPRE usa "kpi_query".**  ← 🆕 REGLA EXPLÍCITA
- Si hay duda entre SQL y KPI, prefiere "kpi_query".

[Pregunta del usuario]
{state['question']}

[Formato de salida esperado]
{{"action":"...","rationale":"...","reformulated_question":"..."}}
"""
    resp = llm.invoke(prompt)
    text = resp.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        plan = json.loads(text)
        print(f" Plan generado: {plan}")
        state["plan"] = plan
        return state
    except Exception as e:
        state["error"] = f"Error parseando plan JSON: {e}\nTexto: {text}"
        return state



# ----------------------------
# Nodo 2: Act (Act/Observe)
# ----------------------------


@traceable(run_type="tool", name="Act Node")
def act_node(state: ReActState) -> ReActState:
    if state.get("error"):
        return state

    plan = state.get("plan") or {}
    action = (plan.get("action") or "").lower()
    question = plan.get("reformulated_question") or state["question"]

    print(f" Ejecutando acción detectada: {action}")

    if action in ["sql_query", "kpi_query"]:
        try:
            sql_workflow = build_sql_graph().compile()
            result = sql_workflow.invoke({"input": question})

            # Extrae kpi_numeric_value si existe
            kpi_numeric_value = result.get("kpi_numeric_value")

            # CRÍTICO: Incluir el valor numérico en tool_result
            summary_text = result.get("summary", "Sin resultados desde SQL/KPI Graph.")
            
            if kpi_numeric_value is not None:
                # Añadir el valor real al contexto del LLM
                summary_text += f"\n\n[VALOR KPI VERIFICADO: {kpi_numeric_value}]"
                print(f"KPI numérico añadido al contexto: {kpi_numeric_value}")

            state["tool_result"] = summary_text
            state["kpi_numeric_value"] = kpi_numeric_value

        except Exception as e:
            state["tool_result"] = f"Error ejecutando grafo SQL/KPI: {e}"
        return state

    elif action == "direct_answer":
        state["tool_result"] = (
            "No se requirió consulta a datos; se responderá con conocimiento existente + contexto."
        )
        return state

    else:
        state["error"] = f"Acción desconocida: {action}"
        return state

# ----------------------------
# Nodo 3: Draft (síntesis)
# ----------------------------

@traceable(run_type="llm", name="Draft Node")
def draft_node(state: ReActState) -> ReActState:
    if state.get("error"):
        return state
    
    llm = get_llm()
    context = (
        state.get("business_context", "")
        + "\n\n[Contexto recuperado]\n"
        + state.get("retrieved_context", "")
    )
    tool_result = state.get("tool_result", "")
    kpi_value = state.get("kpi_numeric_value")
    
    # CRÍTICO: Instrucción explícita para usar datos reales
    value_instruction = ""
    if kpi_value is not None:
        value_instruction = f"\n\nVALOR KPI REAL A USAR: {kpi_value}\nEste es el valor exacto que debes citar y analizar."
    
    prompt = f"""
Eres un analista de datos senior de RocknBlock. Redacta una respuesta clara y ejecutiva con base en el siguiente material:

[Contexto de negocio]
{context}

[Resultados obtenidos]
{tool_result}{value_instruction}

INSTRUCCIONES CRÍTICAS:
- USA ÚNICAMENTE los datos y valores proporcionados arriba
- Si hay un valor numérico específico, cítalo exactamente
- NO inventes cifras ni interpretaciones sin datos de soporte
- En español y orientado a decisiones
- Interpreta hallazgos, destaca tendencias, outliers y próximos pasos
- Sugiere al final una pregunta siguiente útil y una acción táctica
- No incluyas firmas ni placeholders
"""
    
    resp = llm.invoke(prompt)
    draft = strip_signature(resp.content.strip())
    state["draft_answer"] = draft
    return state

# ----------------------------
# Nodo 4: Reflection (auditoría)
# ----------------------------

@traceable(run_type="llm", name="Reflection Node")
def reflection_node(state: ReActState) -> ReActState:
    if state.get("error"):
        return state
    
    llm = get_llm()
    draft = state.get("draft_answer", "")
    rules = (
        "- Verifica que la conclusión se apoye en hechos o resultados.\n"
        "- Sugiere mejoras de precisión y claridad ejecutiva.\n"
        "- Añade 1–2 recomendaciones accionables.\n"
        "- No agregues firmas ni placeholders."
    )
    
    prompt = f"""
Eres un auditor de calidad de análisis. Mejora el texto cumpliendo las siguientes reglas:

[Reglas]
{rules}

[Borrador]
{draft}

[Salida]
Texto mejorado, final y listo para entregar.
"""
    
    resp = llm.invoke(prompt)
    improved = resp.content.strip()
    improved = strip_signature(improved)
    improved = ensure_closing(improved)
    
    #  CRÍTICO: Preservar kpi_numeric_value en TODOS los casos
    return {
        "final_answer": improved,
        "kpi_numeric_value": state.get("kpi_numeric_value"),  # 
        "error": state.get("error"),
        "question": state.get("question"),  # 
        "plan": state.get("plan"),
        "tool_result": state.get("tool_result"),
        "draft_answer": state.get("draft_answer"),
    }


# ----------------------------
# Build & Run
# ----------------------------
def build_graph():
    g = StateGraph(ReActState)

    g.add_node("industry", industry_context_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("planner", plan_node)
    g.add_node("actor", act_node)
    g.add_node("drafter", draft_node)
    g.add_node("reflector", reflection_node)

    g.set_entry_point("industry")
    g.add_edge("industry", "retrieval")
    g.add_edge("retrieval", "planner")
    g.add_edge("planner", "actor")
    g.add_edge("actor", "drafter")
    g.add_edge("drafter", "reflector")
    g.add_edge("reflector", END)
    return g.compile()


if __name__ == "__main__":
    import sys
    business_context = (
        "RocknBlock es una empresa de landscaping. Métricas clave: ventas (fct_sales), "
        "leads (dim_leads), trabajos (fct_housecallpro_jobs) y empleados (dim_housecallpro_employees). "
        "Objetivo: responder preguntas con datos reales del esquema 'clean' y entregar recomendaciones accionables."
    )
    question = " ".join(sys.argv[1:]).strip() or "¿Cuál es el top 5 de ciudades por ventas?"
    workflow = build_graph()
    state: ReActState = {
        "question": question,
        "business_context": business_context,
        "plan": None,
        "tool_result": None,
        "draft_answer": None,
        "final_answer": None,
        "error": None,
        "retrieved_context": None
    }
    result = workflow.invoke(state)
    print(result.get("final_answer") or result.get("draft_answer") or result.get("tool_result") or result.get("error"))
