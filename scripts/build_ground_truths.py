#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exporta automáticamente ground truths para TODAS las vistas KPI del esquema ai.
Ahora soporta KPI globales (una sola métrica) y KPI por dimensión (por ciudad, cliente, etc.)
Guarda los resultados en data/ground_truths/<table>.json
"""

import os
import json
import psycopg2
import statistics
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
OUTPUT_DIR = Path("data/ground_truths")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="require"
    )


def summarize_table(conn, table_name: str, limit: int = 20):
    """Obtiene resumen general de cualquier vista o tabla KPI."""
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM ai.{table_name} LIMIT {limit};")
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        if not rows:
            return None

        summary = {"kpi_name": table_name, "columns": cols}

        # Caso KPI Global (1 sola columna numérica)
        if len(cols) == 1 or all("global" in c for c in cols):
            val = float(rows[0][0])
            summary.update({
                "type": "single_value",
                "reference_metric": "single",
                "value": round(val, 4),
            })
            print(f"🌐 {table_name}: KPI global único = {val}")
            return summary

        # Caso general (por dimensión)
        import decimal
        numeric_indices = []
        for i, name in enumerate(cols):
            sample_vals = [r[i] for r in rows if r[i] is not None]
            if not sample_vals:
                continue
            if all(isinstance(v, (int, float, decimal.Decimal)) or str(v).replace('.', '', 1).isdigit() for v in sample_vals[:10]):
                numeric_indices.append(i)

        summary["type"] = "multi_value"
        summary["stats"] = {}
        for idx in numeric_indices:
            vals = [float(r[idx]) for r in rows if r[idx] is not None]
            if not vals:
                continue
            col = cols[idx]
            stats_dict = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
                "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            }
            if "avg" in col.lower() or "mean" in col.lower():
                stats_dict["type"] = "average"
            elif "total" in col.lower():
                stats_dict["type"] = "sum_total"
            else:
                stats_dict["type"] = "generic"
            summary["stats"][col] = stats_dict

        summary["reference_metric"] = "median"
        return summary

    except Exception as e:
        print(f"❌ Error procesando {table_name}: {e}")
        return None


def get_all_ai_views(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'ai'
        AND table_type IN ('VIEW','BASE TABLE')
        AND (table_name ILIKE 'kpi_%' OR table_name ILIKE 'industry_%');
    """)
    return [r[0] for r in cur.fetchall()]


def save_json(name: str, data: dict):
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    conn = get_conn()
    tables = get_all_ai_views(conn)
    print(f"\nProcesando {len(tables)} vistas del esquema ai...\n")
    for t in tables:
        data = summarize_table(conn, t)
        if data:
            save_json(t, data)
    conn.close()
    print("\nGround truths actualizados en data/ground_truths/")






