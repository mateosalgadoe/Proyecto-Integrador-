import psycopg2, os
from dotenv import load_dotenv

load_dotenv()

def fetch_kpi_definition(kpi_name: str):
    """Busca la definición SQL del KPI en la capa semántica"""
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
    """Ejecuta la query asociada al KPI desde la tabla semantic ai.table_context"""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="require"
    )
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM ai.kpi_{kpi_name.replace(' ', '_').lower()} LIMIT 10;")
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    conn.close()
    return {"columns": colnames, "data": rows}
