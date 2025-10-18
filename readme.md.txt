# RocknBlock AI Pipeline

Este proyecto implementa un **pipeline de agentes inteligentes** para análisis de datos empresariales en **RocknBlock Landscaping**, utilizando **PostgreSQL**, **LangGraph** y **modelos LLM** (GPT-5, Claude).  

El objetivo es **simular el razonamiento de un analista de datos humano**, integrando:
- Una **capa semántica** en la base de datos (tablas de contexto y columnas).
- Grafos de razonamiento con **LangGraph** para generación y ejecución de SQL determinista.
- Patrones agentivos (**ReAct + Reflection**) para que el modelo planifique, actúe y mejore sus respuestas.
- Contexto de negocio específico de RocknBlock, con posibilidad de extender a **datos macro/microeconómicos** (Banco Mundial, Yahoo Finance).

## 📂 Arquitectura del Proyecto

```
rocknblock-ai/
├── data/                    # Datos locales (raw, clean, processed, external)
├── db/                      # Scripts SQL (migraciones, seeds, semantic layer, golden queries)
├── deploy/                  # Infraestructura (Docker, Kubernetes)
├── docs/                    # Documentación (arquitectura, decisiones, runbooks)
├── notebooks/               # EDA, pruebas de agentes, prototipos
├── src/
│   ├── agents/langgraph/    # Grafos de agentes
│   │   ├── basic_graph.py   # Primer prototipo simple
│   │   ├── sql_graph.py     # Grafo SQL determinista
│   │   └── react_graph.py   # Grafo ReAct + Reflection
│   ├── semantic/            # Capa semántica
│   │   └── load_context.py
│   └── ...                  # (kpi, eval, ingestion, utils, etc. listos para siguientes sprints)
└── tests/                   # Unit, integration, end-to-end tests
```

##  Componentes Principales

### 1. **Capa Semántica en DB**
- `db/sql/semantic/01_create_semantic_layer.sql`  
- Define dos tablas en el esquema `ai`:
  - `table_context`: metadatos de tablas (fact/dimension, dominio, descripción).
  - `column_context`: metadatos de columnas (tipo de dato, significado, ejemplo).
- Permite que el LLM tenga **contexto de negocio explícito**.

### 2. **Graphs de LangGraph**
- **`basic_graph.py`** → carga contexto y responde con LLM (prototipo inicial).
- **`sql_graph.py`** → genera SQL válido (solo SELECT sobre `clean.*`), lo valida y ejecuta en Postgres. Devuelve preview y explicación de negocio.
- **`react_graph.py`** → implementa el patrón **ReAct + Reflection**:
  - **Reason (plan)**: decide acción (SQL, respuesta directa, KPI).
  - **Act**: ejecuta herramienta (`sql_graph`) o responde.
  - **Observe**: analiza resultados.
  - **Reflect**: revisa y mejora la respuesta antes de entregarla.

### 3. **Requisitos**
- Python 3.10+ (se recomienda entorno virtual `venv`).
- PostgreSQL con schema `clean` y schema semántico `ai`.
- API key de OpenAI (y en futuro Anthropic Claude).

## ⚙️ Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <url_repo>
   cd rocknblock-ai
   ```

2. **Crear entorno virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables en `.env`:**
   ```env
   # Postgres
   DB_HOST=datalysis-analytics-db.postgres.database.azure.com
   DB_PORT=5432
   DB_NAME=rocknblock
   DB_USER=datalysis_analytics_db_admin
   DB_PASSWORD=********

   # LLM
   OPENAI_API_KEY=sk-********
   ```

## 🚀 Uso

### 1. Probar el grafo SQL determinista
Genera y ejecuta queries SQL con validación:

```bash
python3 -m src.agents.langgraph.sql_graph "Dame el total de ventas por ciudad usando fct_sales"
```

**Ejemplo de salida:**

```
Consulta ejecutada correctamente sobre el schema `clean`.

SQL usado:
SELECT city, SUM(total_invoice) AS total_sales
FROM clean.fct_sales
GROUP BY city
LIMIT 100;

Vista previa (primeras filas):
{
  "columns": ["city", "total_sales"],
  "rows": [["Las Vegas", 11601469.77], ["Henderson", 2330875.60], ...]
}

Interpretación de negocio: ...
```

### 2. Probar el grafo ReAct
Razonamiento paso a paso con reflexión final:

```bash
python3 -m src.agents.langgraph.react_graph "Top 5 ciudades por ventas en clean.fct_sales"
python3 -m src.agents.langgraph.react_graph "Muestra 5 leads recientes"
python3 -m src.agents.langgraph.react_graph "¿Qué métricas debo monitorear semanalmente?"
```

**El agente entrega:**
- Hallazgos clave (con datos).
- Interpretación ejecutiva.
- Recomendaciones y próximos pasos.

## 📌 Estado del Proyecto

### ✅ **Hecho**
- Capa semántica (Sprint 3).
- Grafo SQL determinista (Sprint 4).
- Grafo ReAct + Reflection (Sprint 4).
- Integración con OpenAI GPT.

### 🔜 **Próximos pasos**
- Integrar KPI Layer con "consultas doradas".
- Conectar macro/microeconómicos (Banco Mundial, Yahoo Finance).
- Añadir tests unitarios y e2e.
- Documentación en `docs/architecture/`.
- Integración con LangSmith para evaluación:
  - Execution Accuracy
  - KPI Correctness
  - Consistency Score
  - RMSE, costo/respuesta