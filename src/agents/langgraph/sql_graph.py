import os
import psycopg2
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.agents.utils.model_factory import get_model
from src.config.langsmith_init import init_langsmith
from langsmith import traceable

# ======================
# Inicialización global
# ======================

load_dotenv()
init_langsmith()

# ======================
# Diccionario de KPIs
# ======================

KPI_ALIASES = {
    # --- Ventas Globales ---
    "promedio global de ventas": "kpi_avg_sales_volume_global",
    "promedio general de ventas": "kpi_avg_sales_volume_global",
    "volumen de ventas global": "kpi_avg_sales_volume_global",
    "ventas promedio": "kpi_avg_sales_volume_global",
    "volumen de ventas": "kpi_avg_sales_volume_global",
    "promedio de ventas": "kpi_avg_sales_volume_global",
    
    # --- Ventas por Ciudad ---
    "ventas promedio por ciudad": "kpi_avg_sales_volume",
    "promedio por ciudad": "kpi_avg_sales_volume",
    "ventas por ciudad": "kpi_sales_volume_by_city",
    "volumen total por ciudad": "kpi_sales_volume_by_city",
    "total de ventas por ciudad": "kpi_sales_volume_by_city",
    
    # --- Closing Rate ---
    "closing rate": "kpi_closing_rate",
    "tasa de cierre": "kpi_closing_rate",
    "tasa de cierre por ciudad": "kpi_closing_rate",
    "porcentaje de cierre": "kpi_closing_rate",
    
    # --- Tendencias ---
    "tendencia mensual": "Sales Trend Monthly",
    "tendencia de ventas mensual": "Sales Trend Monthly",
    "ventas por mes": "Sales Trend Monthly",
    "evolución mensual": "Sales Trend Monthly",
    
    # --- Benchmarks ---
    "industria": "industry_benchmarks",
    "benchmark": "industry_benchmarks",
    "benchmarks de industria": "industry_benchmarks",
    "comparativa de industria": "industry_benchmarks",

    # --- MÁS ESPECÍFICO PARA by_city ---
    "ventas por ciudad": "kpi_sales_volume_by_city",  # ANTES de "ventas"
    "total de ventas por ciudad": "kpi_sales_volume_by_city",
    "volumen total por ciudad": "kpi_sales_volume_by_city",
    
    # --- TENDENCIA ---
    "tendencia de ventas mensual": "Sales Trend Monthly",
    "tendencia mensual de ventas": "Sales Trend Monthly",
    "evolución mensual": "Sales Trend Monthly",



}

# ======================
# Funciones auxiliares
# ======================

def fetch_kpi_definition(kpi_name: str):
    """Busca el KPI por nombre en la capa semántica (ai.table_context)."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="require"
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, description
        FROM ai.table_context
        WHERE table_type = 'kpi'
        AND table_name ILIKE %s
        LIMIT 1;
    """, (f"%{kpi_name}%",))
    result = cur.fetchone()
    conn.close()
    if not result:
        return None
    return {"kpi_name": result[0], "description": result[1]}


def execute_kpi_query(kpi_name: str):
    """Ejecuta la consulta del KPI correspondiente (nombre flexible y seguro)."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require"
        )
        cur = conn.cursor()

        safe_name = kpi_name.strip().lower().replace(" ", "_")
        if safe_name.startswith("kpi_"):
            full_table = f"ai.{safe_name}"
        else:
            cur.execute("""
                SELECT table_name FROM ai.table_context
                WHERE table_name ILIKE %s LIMIT 1;
            """, (f"%{safe_name}%",))
            found = cur.fetchone()
            if found:
                table_name = found[0]
                # NUEVO: Escapar nombres con espacios
                if " " in table_name:
                    full_table = f'ai."{table_name}"'
                else:
                    full_table = f"ai.{table_name}"
            else:
                full_table = f"ai.kpi_{safe_name}"

        print(f"Ejecutando KPI: {full_table}")
        
        # Usar con cautela para nombres con caracteres especiales
        cur.execute(f"SELECT * FROM {full_table} LIMIT 10;")
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        conn.close()

        if not rows:
            print("KPI sin resultados.")
            return {"columns": colnames, "rows": [], "message": "KPI sin filas."}

        print(f"{len(rows)} filas devueltas.")
        return {"columns": colnames, "rows": rows}

    except Exception as e:
        print(f"Error en execute_kpi_query: {e}")
        return {"error": str(e), "columns": [], "rows": []}

# ======================
# Nodos del grafo
# ======================


@traceable(run_type="tool", name="KPI Node")
def kpi_node(state):
    """Ejecuta la lógica KPI con detección semántica robusta (prioridad: global > ciudad > benchmark > fallback)."""

    print("Entrando a kpi_node. Keys de entrada:", list(state.keys()))
    user_input = (state.get("input") or "").lower()
    kpi_info = None

    # --- 1️⃣ Detección explícita por alias ---
    for key, val in KPI_ALIASES.items():
        if key in user_input:
            kpi_info = {"kpi_name": val, "description": f"KPI detectado por alias: {val}"}
            print(f"KPI detectado por alias: {val}")
            break

    # --- 2️⃣ Detección semántica global (tiene prioridad absoluta) ---
    if not kpi_info and any(word in user_input for word in ["global", "total", "en general", "todas las ciudades"]):
        kpi_info = {
            "kpi_name": "kpi_avg_sales_volume_global",
            "description": "Promedio global de volumen de ventas (detección semántica prioritaria)"
        }
        print("KPI detectado por semántica: kpi_avg_sales_volume_global")

    # --- 3️⃣ Detección contextual por ciudad (solo si no hay global) ---
    if not kpi_info:
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                sslmode="require"
            )
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT LOWER(city) FROM ai.kpi_avg_sales_volume;")
            city_list = [r[0] for r in cur.fetchall()]
            conn.close()
        except Exception as e:
            print(f"No se pudo cargar lista de ciudades: {e}")
            city_list = []

        context_words = ["ciudad", "mercado", "oportunidad", "crecimiento", "ventas", "desempeño"]

        if any(word in user_input for word in city_list + context_words):
            print("Contexto detectado (ciudad u oportunidad). Usando KPI por defecto: kpi_avg_sales_volume")
            kpi_info = {
                "kpi_name": "kpi_avg_sales_volume",
                "description": "Volumen promedio de ventas por ciudad (detección automática)"
            }

    # --- 4️⃣ Búsqueda semántica general si no hay coincidencia ---
    if not kpi_info:
        kpi_info = fetch_kpi_definition(user_input)

    # --- 5️⃣ Error controlado si no se encontró nada ---
    if not kpi_info:
        state["error"] = f"No se encontró un KPI relacionado con '{user_input}'."
        print("No se encontró KPI asociado, devolviendo error controlado.")
        return state

    # --- 6️⃣ Ejecución del KPI ---
    result = execute_kpi_query(kpi_info["kpi_name"])
    print(f"KPI filas devueltas: {len(result.get('rows', []))}")

    new_state = dict(state)
    new_state.update({
        "result_type": "kpi",
        "kpi_name": kpi_info["kpi_name"],
        "description": kpi_info["description"],
        "data": result
    })
    print("Saliendo de kpi_node. Keys de salida:", list(new_state.keys()))
    return new_state


def run_sql_query(prompt: str):
    """Genera y ejecuta una consulta SQL válida o KPI reconocida."""
    llm = get_model()
    sql_prompt = f"""
    Eres un analista experto en datos de RocknBlock.
    Devuelve solo una instrucción SQL ejecutable para PostgreSQL.
    - Si el usuario menciona "KPI", consulta directamente la vista en el esquema ai.
    - Si menciona ventas, leads o jobs, usa el esquema clean.
    - No incluyas explicaciones ni texto adicional.
    ---
    Pregunta del usuario:
    {prompt}
    ---
    """

    sql_response = llm.invoke(sql_prompt)
    sql = sql_response.content.strip().replace("```sql", "").replace("```", "")
    print(f"\nSQL generado por el modelo:\n{sql}\n")

    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require"
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        conn.close()
        return {"sql": sql, "columns": colnames, "rows": rows}
    except Exception as e:
        print(f"Error al ejecutar SQL: {e}")
        return {"error": str(e), "sql": sql, "columns": [], "rows": []}


@traceable(run_type="tool", name="Intent Router")
def intent_router(state):
    """Determina si el usuario pide un KPI o una consulta SQL.
    Respeta la acción detectada previamente (si existe en el estado)."""
    action = state.get("action", "").lower()
    if "sql" in action:
        path = "sql"
    elif "kpi" in action:
        path = "kpi"
    else:
        user_input = state.get("input", "")
        kpi_keywords = [
            "tasa", "promedio", "ratio", "porcentaje",
            "métrica", "indicador", "volumen", "ventas", "kpi", "closing"
        ]
        path = "kpi" if any(word in user_input.lower() for word in kpi_keywords) else "sql"
    print(f"Ruta seleccionada: {path.upper()}")
    new_state = dict(state)
    new_state["path"] = path
    return new_state


@traceable(run_type="tool", name="SQL Node")
def sql_node(state):
    """Ejecuta una consulta SQL generada por LLM."""
    user_input = state["input"]
    result = run_sql_query(user_input)
    new_state = dict(state)
    new_state.update({"result_type": "sql", "data": result})
    return new_state
@traceable(run_type="llm", name="Summary Node")
def summary_node(state):
    """Genera interpretación ejecutiva y devuelve summary + valor numérico puro."""
    
    print("Entrando a summary_node con keys:", list(state.keys()))

    llm = get_model()
    result_type = state.get("result_type", "unknown")
    data = state.get("data", {})

    # ✅ NUEVO: Extrae el valor numérico si es un KPI global
    kpi_numeric_value = None
    if result_type == "kpi" and data.get("rows"):
        rows = data.get("rows", [])
        if len(rows) == 1 and len(rows[0]) == 1:
            # Es un KPI global (una sola fila, una sola columna)
            try:
                kpi_numeric_value = float(rows[0][0])
                print(f"✅ KPI global extraído: {kpi_numeric_value}")
            except (ValueError, TypeError):
                pass

    # Validación: si no hay datos, retorna error
    if not data or not data.get("rows"):
        print("⚠️ Advertencia: no se encontraron datos en el estado.")
        summary_text = "No se encontraron resultados o el dataset está vacío."
        return {
            "result_type": result_type,
            "data": data if isinstance(data, dict) else {},
            "summary": summary_text,
            "kpi_numeric_value": kpi_numeric_value,
        }

    print("📊 Datos enviados al LLM:", {
        "columns": data.get("columns"),
        "rows_count": len(data.get("rows", []))
    })

    # Construir el prompt basado en el tipo de resultado
    if result_type == "kpi":
        prompt = f"""
Eres un analista senior de RocknBlock. Genera una interpretación ejecutiva clara y CONCISA del siguiente resultado de KPI:

[Datos del KPI]
Columnas: {data.get('columns', [])}
Filas (primeras 20): {data.get('rows', [])[:20]}

Requisitos:
- En español, orientado a decisiones de negocio.
- Máximo 150 palabras.
- Destaca hallazgos clave, tendencias y próximos pasos.
- SIN firmas ni placeholders.
"""
    else:
        prompt = f"""
Eres un analista senior de RocknBlock. Interpreta estos resultados de SQL y genera un resumen ejecutivo:

[Datos]
Columnas: {data.get('columns', [])}
Filas (primeras 20): {data.get('rows', [])[:20]}

Requisitos:
- En español.
- Máximo 200 palabras.
- Destaca patrones, anomalías y recomendaciones.
- SIN firmas.
"""

    try:
        response = llm.invoke(prompt)
        summary = response.content.strip()
        print("✅ Interpretación generada correctamente.")
    except Exception as e:
        print(f"❌ Error generando summary: {e}")
        summary = f"Error al generar resumen: {e}"

    # Retornar estado completo
    return {
        "result_type": result_type,
        "data": data,
        "summary": summary,
        "kpi_numeric_value": kpi_numeric_value,  # ← Valor numérico puro para evaluación
    }

# ======================
# Construcción del grafo
# ======================

def build_graph():
    """Define la estructura principal del flujo LangGraph."""
    from typing import TypedDict, Optional

    class AgentState(TypedDict, total=False):
        input: str
        path: Optional[str]
        result_type: Optional[str]
        kpi_name: Optional[str]
        description: Optional[str]
        data: Optional[dict]
        summary: Optional[str]
        error: Optional[str]

    g = StateGraph(AgentState)

    # Nodos del flujo
    g.add_node("router", intent_router)
    g.add_node("sql_node", sql_node)
    g.add_node("kpi_node", kpi_node)
    g.add_node("summary_node", summary_node)

    # Aseguramos que router siempre propague el path
    def route_selector(state):
        path = state.get("path", "")
        if path not in ("sql", "kpi"):
            print(f"uta no definida en el estado. Default = 'sql'")
            return "sql"
        return path

    g.add_conditional_edges(
        "router",
        route_selector,
        {"sql": "sql_node", "kpi": "kpi_node"}
    )

    # Asegurar propagación completa del estado
    g.add_edge("sql_node", "summary_node")
    g.add_edge("kpi_node", "summary_node")
    g.add_edge("summary_node", END)
    g.set_entry_point("router")

    return g

# ======================
# Ejecución directa
# ======================

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Total de ventas por ciudad"
    graph = build_graph().compile()

    print(f"\nEjecutando consulta: {query}\n")

    try:
        result = graph.invoke({"input": query})
        print("\n=== Interpretación ejecutiva ===\n")
        print(result.get("summary") or "No se generó resumen.")
        print("\n=== Resultado bruto ===\n")
        print(result.get("data") or "Sin datos.")
    except Exception as e:
        print(f"Error al ejecutar el flujo: {e}")
