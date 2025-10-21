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

                # Caso KPI Global (1 sola columna numérica O columnas con "_global")
        if len(cols) == 1:
            val = float(rows[0][0])
            summary.update({
                "type": "single_value",
                "reference_metric": "single",
                "value": round(val, 4),
            })
            print(f"🌐 {table_name}: KPI global único = {val}")
            return summary
        elif any("_pct_global" in c or "_global" in c for c in cols):
            # 🆕 BUSCAR LA COLUMNA CON "pct_global" O "rate_global"
            for i, col in enumerate(cols):
                if "_pct_global" in col.lower() or "rate_global" in col.lower():
                    val = float(rows[0][i])
                    summary.update({
                        "type": "single_value",
                        "reference_metric": "single",
                        "value": round(val, 4),
                        "source_column": col  # 🆕 Guardar nombre de columna
                    })
                    print(f"🌐 {table_name}: KPI global único = {val} (columna: {col})")
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
    try:
        print("🔄 Conectando a la base de datos...")
        conn = get_conn()
        print("✅ Conexión exitosa")
        
        print("🔍 Buscando vistas en esquema 'ai'...")
        tables = get_all_ai_views(conn)
        
        # 🆕 DEBUG: Mostrar todas las vistas encontradas
        print(f"\n📊 Vistas encontradas ({len(tables)}):")
        for t in sorted(tables):
            print(f"  ✓ {t}")
        print()
        
        print(f"⚙️  Procesando {len(tables)} vistas...\n")
        
        success_count = 0
        error_count = 0
        
        for t in tables:
            try:
                print(f"🔄 Procesando: {t}...", end=" ")
                data = summarize_table(conn, t)
                if data:
                    save_json(t, data)
                    print(f"✅ OK")
                    success_count += 1
                else:
                    print(f"⚠️  Sin datos")
            except Exception as e:
                print(f"❌ ERROR: {e}")
                error_count += 1
        
        conn.close()
        
        print(f"\n{'='*50}")
        print(f"✅ Procesamiento completado:")
        print(f"  - Exitosos: {success_count}")
        print(f"  - Sin datos: {len(tables) - success_count - error_count}")
        print(f"  - Errores: {error_count}")
        print(f"  - Directorio: data/ground_truths/")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()