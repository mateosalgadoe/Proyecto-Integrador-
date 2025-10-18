import psycopg2
import json
import os
from dotenv import load_dotenv

# Cargar variables desde .env
load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "datalysis-analytics-db.postgres.database.azure.com"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "rocknblock"),
        user=os.getenv("DB_USER", "datalysis_analytics_db_admin"),
        password=os.getenv("DB_PASSWORD")  # ⚠️ define esta variable en tu entorno
    )

def load_context():
    conn = get_connection()
    cur = conn.cursor()

    # Cargar tablas con contexto
    cur.execute("SELECT table_name, table_type, business_domain, description FROM ai.table_context;")
    tables = cur.fetchall()

    # Cargar columnas con contexto
    cur.execute("SELECT table_name, column_name, data_type, business_meaning, example_value FROM ai.column_context;")
    columns = cur.fetchall()

    conn.close()

    # Estructurar diccionario
    context = {}
    for t_name, t_type, domain, desc in tables:
        context[t_name] = {
            "type": t_type,
            "business_domain": domain,
            "description": desc,
            "columns": []
        }

    for t_name, col, dtype, meaning, example in columns:
        if t_name in context:
            context[t_name]["columns"].append({
                "name": col,
                "data_type": dtype,
                "business_meaning": meaning,
                "example": example
            })

    return context

if __name__ == "__main__":
    context = load_context()
    print(json.dumps(context, indent=2, ensure_ascii=False))
