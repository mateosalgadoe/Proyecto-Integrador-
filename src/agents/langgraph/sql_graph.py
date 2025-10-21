import os
import psycopg2
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.agents.utils.model_factory import get_model
from src.config.langsmith_init import init_langsmith
from langsmith import traceable

load_dotenv()
init_langsmith()

# DICCIONARIO DE KPIs MEJORADO


KPI_ALIASES = {
    # --- Ventas Globales (PRIORIDAD MÁXIMA) ---
    "promedio global de ventas": "kpi_avg_sales_volume_global",
    "promedio general de ventas": "kpi_avg_sales_volume_global",
    "volumen de ventas global": "kpi_avg_sales_volume_global",
    "ventas promedio global": "kpi_avg_sales_volume_global",
    "volumen global": "kpi_avg_sales_volume_global",
    "promedio de ventas": "kpi_avg_sales_volume_global",
    "kpi_avg_sales_volume_global": "kpi_avg_sales_volume_global",

    # --- Closing Rate GLOBAL (DEBE IR ANTES QUE POR CIUDAD) ---
    "closing rate global": "kpi_closing_rate_global",
    "tasa de cierre global": "kpi_closing_rate_global",
    "closing rate promedio": "kpi_closing_rate_global",
    "tasa de cierre promedio": "kpi_closing_rate_global",
    "closing rate de rocknblock": "kpi_closing_rate_global",  
    "tasa de cierre de rocknblock": "kpi_closing_rate_global",  
    
    # --- Closing Rate por ciudad (DESPUÉS DEL GLOBAL) ---
    "closing rate por ciudad": "kpi_closing_rate",
    "tasa de cierre por ciudad": "kpi_closing_rate",
    "closing rate de": "kpi_closing_rate",  # Para "closing rate de Las Vegas"
    "tasa de cierre de": "kpi_closing_rate",
    # ❌ REMOVER ESTOS DOS (causan conflicto):
    # "closing rate": "kpi_closing_rate",
    # "tasa de cierre": "kpi_closing_rate",
    "porcentaje de cierre": "kpi_closing_rate",
    "closing ratio": "kpi_closing_rate",
    "conversión": "kpi_closing_rate",
    "win rate": "kpi_closing_rate",

    # --- Ventas por Ciudad ---
    "ventas promedio por ciudad": "kpi_avg_sales_volume",
    "promedio por ciudad": "kpi_avg_sales_volume",
    "ventas por ciudad": "kpi_sales_volume_by_city",
    "volumen total por ciudad": "kpi_sales_volume_by_city",
    "total de ventas por ciudad": "kpi_sales_volume_by_city",
    "kpi_sales_volume_by_city": "kpi_sales_volume_by_city",

    # --- Benchmarks ---
    "benchmark": "industry_benchmarks",
    "benchmarks": "industry_benchmarks",
    "industria": "industry_benchmarks",
    "benchmarks de industria": "industry_benchmarks",
    "benchmark de industria": "industry_benchmarks",
    "comparativa de industria": "industry_benchmarks",
    "industria de landscaping": "industry_benchmarks",
    "comparación con industria": "industry_benchmarks",
    "sector landscaping": "industry_benchmarks",
    "mercado": "industry_benchmarks",
    "industry_benchmarks": "industry_benchmarks",
        # --- Benchmarks (AMPLIAR COBERTURA) ---
    "benchmarks de crecimiento": "industry_benchmarks",
    "crecimiento de pib": "industry_benchmarks",
    "pib en la industria": "industry_benchmarks",
    "benchmarks de industria": "industry_benchmarks",
    "benchmark de industria": "industry_benchmarks",
    "comparativa de industria": "industry_benchmarks",
    "industria de landscaping": "industry_benchmarks",
    "comparación con industria": "industry_benchmarks",
    "sector landscaping": "industry_benchmarks",
    "benchmarks": "industry_benchmarks",
    "benchmark": "industry_benchmarks",
    "industria": "industry_benchmarks",
    "mercado": "industry_benchmarks",
    "industry_benchmarks": "industry_benchmarks",
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
                if " " in table_name:
                    full_table = f'ai."{table_name}"'
                else:
                    full_table = f"ai.{table_name}"
            else:
                full_table = f"ai.kpi_{safe_name}"

        print(f"Ejecutando KPI: {full_table}")
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
    """Ejecuta la lógica KPI con detección semántica mejorada."""

    print("Entrando a kpi_node. Keys de entrada:", list(state.keys()))
    raw_input = (state.get("input") or "").lower()
    
    # 🆕 NORMALIZACIÓN: limpiar caracteres especiales
    user_input = raw_input.replace("_", " ")  # closing_rate → closing rate
    user_input = user_input.replace("'", "")   # 'closing rate' → closing rate
    user_input = user_input.replace('"', "")   # "closing rate" → closing rate
    
    kpi_info = None

    # --- 1 Detección EXACTA por alias (PRIORIDAD MÁXIMA) ---
    # Ordenar por longitud descendente para match más específico primero
    sorted_aliases = sorted(KPI_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
    
    for key, val in sorted_aliases:
        if key in user_input:
            kpi_info = {"kpi_name": val, "description": f"KPI detectado por alias: {val}"}
            print(f" KPI detectado por alias '{key}': {val}")
            break


    # --- 3 Búsqueda semántica general si no hay coincidencia ---
    if not kpi_info:
        kpi_info = fetch_kpi_definition(user_input)

    # --- 4 Error controlado si no se encontró nada ---
    if not kpi_info:
        state["error"] = f"No se encontró un KPI relacionado con '{user_input}'."
        print("❌ No se encontró KPI asociado.")
        return state

    # --- 5️Ejecución del KPI ---
    result = execute_kpi_query(kpi_info["kpi_name"])
    print(f" KPI filas devueltas: {len(result.get('rows', []))}")

    new_state = dict(state)
    new_state.update({
        "result_type": "kpi",
        "kpi_name": kpi_info["kpi_name"],
        "description": kpi_info["description"],
        "data": result
    })
    print("Saliendo de kpi_node. Keys de salida:", list(new_state.keys()))
    return new_state



def run_sql_query_temporal(prompt: str):
    """Genera SQL específico para consultas temporales (tendencias)."""
    llm = get_model()
    sql_prompt = f"""
Eres un analista experto en PostgreSQL para RocknBlock (landscaping).
Genera SOLO una consulta SQL para obtener tendencias temporales.

ESQUEMA DISPONIBLE:
- clean.fct_sales: ventas con id_fecha, total_invoice, city
- clean.dim_leads: leads con lead_date, status  
- clean.fct_housecallpro_jobs: trabajos con job_date, status

REQUISITOS:
- Usa DATE_TRUNC('month', fecha_campo) para agrupar por mes
- Incluye año-mes como YYYY-MM
- Ordena cronológicamente
- Limita a últimos 12-24 meses

PREGUNTA: {prompt}

SQL (sin explicaciones):
"""
    
    sql_response = llm.invoke(sql_prompt)
    sql = sql_response.content.strip().replace("```sql", "").replace("```", "")
    print(f"\nSQL temporal generado:\n{sql}\n")
    
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
        print(f"Error en SQL temporal: {e}")
        return {"error": str(e), "sql": sql, "columns": [], "rows": []}




@traceable(run_type="tool", name="Intent Router")
def intent_router(state):
    """Determina si el usuario pide un KPI o una consulta SQL."""
    action = state.get("action", "").lower()
    user_input = state.get("input", "").lower()
    
    if "sql" in action:
        path = "sql"
    elif "kpi" in action:
        path = "kpi"
    else:
        # NUEVO: Detección específica de tendencias temporales
        if any(word in user_input for word in ["tendencia mensual", "evolución mensual", "ventas por mes", "serie temporal"]) and "kpi" not in user_input:
            path = "sql"
            print(" Detectada consulta temporal → SQL directo")
        else:
            # Detección de KPIs mejorada
            kpi_keywords = [
                "tasa", "promedio", "ratio", "porcentaje", "métrica", "indicador", 
                "volumen", "ventas", "kpi", "closing", "benchmark", "industria",
                "sector", "comparativa", "mercado"
            ]
            path = "kpi" if any(word in user_input for word in kpi_keywords) else "sql"
    
    print(f" Ruta seleccionada: {path.upper()}")
    new_state = dict(state)
    new_state["path"] = path
    return new_state



@traceable(run_type="tool", name="SQL Node")
def sql_node(state):
    """Ejecuta una consulta SQL, con lógica especial para temporales."""
    user_input = state["input"]
    
    # Detectar si es consulta temporal
    if any(word in user_input.lower() for word in ["tendencia", "evolución", "mensual", "serie temporal"]):
        result = run_sql_query_temporal(user_input)
        print("Ejecutando consulta temporal específica")
    else:
        result = run_sql_query(user_input)
    
    new_state = dict(state)
    new_state.update({"result_type": "sql", "data": result})
    return new_state




@traceable(run_type="llm", name="Summary Node")
def summary_node(state):
    """Genera interpretación ejecutiva CON DATOS REALES."""
    
    print("Entrando a summary_node con keys:", list(state.keys()))
    
    llm = get_model()
    result_type = state.get("result_type", "unknown")
    data = state.get("data", {})
    
    # Extrae el valor numérico si es un KPI global
    kpi_numeric_value = None
    if result_type == "kpi" and data.get("rows"):
        rows = data.get("rows", [])
        if len(rows) == 1 and len(rows[0]) == 1:
            try:
                kpi_numeric_value = float(rows[0][0])
                print(f" KPI global extraído: {kpi_numeric_value}")
            except (ValueError, TypeError):
                pass
    elif len(rows) > 1:
        # Para KPIs con múltiples filas, extraer el valor máximo
        try:
            max_val = max(float(r[1]) for r in rows if len(r) > 1 and r[1] is not None)
            kpi_numeric_value = max_val
            print(f" KPI múltiple filas, extraído valor máximo: {kpi_numeric_value}")
        except (ValueError, TypeError):
            pass
    
    if not data or not data.get("rows"):
        summary_text = "No se encontraron resultados o el dataset está vacío."
        return {
            "result_type": result_type,
            "data": data if isinstance(data, dict) else {},
            "summary": summary_text,
            "kpi_numeric_value": kpi_numeric_value,
        }
    
    # CRÍTICO: Incluir datos reales en el prompt
    rows_sample = data.get("rows", [])[:5]  # Primeras 5 filas
    columns = data.get("columns", [])
    
    if result_type == "kpi":
        prompt = f"""
Eres un analista senior de RocknBlock. Interpreta este resultado de KPI CON LOS DATOS REALES:

[KPI EJECUTADO]
Nombre: {state.get("kpi_name", "unknown")}
Descripción: {state.get("description", "")}

[DATOS REALES]
Columnas: {columns}
Valores obtenidos: {rows_sample}
Valor numérico principal: {kpi_numeric_value if kpi_numeric_value else "N/A"}

REQUISITOS CRÍTICOS:
- Usa ÚNICAMENTE los valores reales proporcionados arriba
- Cita el valor exacto: {kpi_numeric_value} si aplica
- No inventes interpretaciones sin datos
- Máximo 150 palabras, enfoque ejecutivo
- En español, sin firmas
"""
    else:
        prompt = f"""
Eres un analista senior de RocknBlock. Interpreta estos datos SQL:

[DATOS REALES]
Columnas: {columns}
Resultados: {rows_sample}
Total de filas: {len(data.get("rows", []))}

Requisitos:
- Usa solo los datos proporcionados arriba
- Identifica patrones en los valores reales
- Máximo 200 palabras
- En español, sin firmas
"""
    
    try:
        response = llm.invoke(prompt)
        summary = response.content.strip()
        print(" Interpretación generada con datos reales.")
    except Exception as e:
        print(f"Error generando summary: {e}")
        summary = f"Error al generar resumen: {e}"
    
    return {
        "result_type": result_type,
        "data": data,
        "summary": summary,
        "kpi_numeric_value": kpi_numeric_value,
    }

def run_sql_query(prompt: str):
    """Genera y ejecuta una consulta SQL válida o KPI reconocida."""
    llm = get_model()
    sql_prompt = f"""
    Eres un analista experto en datos de RocknBlock.
    Devuelve solo una instrucción SQL ejecutable para PostgreSQL.
    
    ESQUEMA CORRECTO:

    - clean.fct_sales: ventas con id_fecha, total_invoice, city
    - clean.dim_leads: leads con lead_date, status
    - clean.fct_housecallpro_jobs: trabajos con job_date, status

    - Si el usuario menciona "KPI", consulta directamente la vista en el esquema ai.
    - Si menciona ventas, leads o jobs, usa el esquema clean.
    - No incluyas explicaciones ni texto adicional.
    
    Pregunta del usuario: {prompt}
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

    g.add_node("router", intent_router)
    g.add_node("sql_node", sql_node)
    g.add_node("kpi_node", kpi_node)
    g.add_node("summary_node", summary_node)

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

    g.add_edge("sql_node", "summary_node")
    g.add_edge("kpi_node", "summary_node")
    g.add_edge("summary_node", END)
    g.set_entry_point("router")

    return g

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
