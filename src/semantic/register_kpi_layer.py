import psycopg2, os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

kpi_queries = [
    ("Sales Volume by City", "Ventas", "db/sql/golden_queries/sales_volume_by_city.sql"),
    ("Closing Ratio", "Ventas", "db/sql/golden_queries/closing_ratio.sql"),
    ("Sales Trend Monthly", "Ventas", "db/sql/golden_queries/sales_trend_monthly.sql")
]

for name, domain, path in kpi_queries:
    with open(path, "r") as f:
        sql_code = f.read()
    cur.execute("""
        INSERT INTO ai.table_context (table_name, table_type, business_domain, description)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (table_name) DO NOTHING;
    """, (name, "kpi", domain, sql_code.split("\n")[1].replace("-- ", "")))

conn.commit()
cur.close()
conn.close()
print(" KPI Layer registrada correctamente!! ")
