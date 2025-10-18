"""
Script: update_industry_context.py
Autor: Mateo Salgado
Descripción:
  - Actualiza los datos macro/microeconómicos desde APIs públicas (Banco Mundial y Yahoo Finance)
  - Guarda el contexto actualizado en data/external/industry_context.json
  - Inserta/actualiza la tabla ai.industry_benchmarks en PostgreSQL
"""

import os
import json
import psycopg2
import datetime
from dotenv import load_dotenv

# Librerías opcionales: yahooquery (Yahoo Finance) y requests (World Bank)
import requests
from yahooquery import Ticker

# Cargar entorno
load_dotenv()

OUTPUT_PATH = "data/external/industry_context.json"

# ===============================
# 1. Banco Mundial (macro global)
# ===============================
def fetch_worldbank_macro(country="US"):
    """Obtiene datos macroeconómicos del Banco Mundial (PIB, inflación, desempleo)."""
    indicators = {
        "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",          # Crecimiento del PIB (%)
        "FP.CPI.TOTL.ZG": "inflation_pct",              # Inflación (%)
        "SL.UEM.TOTL.ZS": "unemployment_pct"            # Desempleo (%)
    }

    macro = {}
    for code, name in indicators.items():
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{code}?format=json&per_page=1"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            if isinstance(data, list) and len(data) > 1 and data[1]:
                macro[name] = round(data[1][0]["value"], 2)
            else:
                macro[name] = None
        except Exception as e:
            macro[name] = None
            print(f"Error al obtener {name} ({code}): {e}")

    return macro


# ====================================
# 2. Yahoo Finance (sector landscaping)
# ====================================
def fetch_yahoo_sector(symbols=["SITE", "TBLD", "TREX"]):
    """Obtiene retornos YTD promedio del sector Landscaping (empresas comparables)."""
    try:
        tickers = Ticker(symbols)
        quotes = tickers.price
        sector_perf = []
        for s in symbols:
            perf = quotes[s]["regularMarketChangePercent"]
            sector_perf.append(perf)
        avg_growth = round(sum(sector_perf) / len(sector_perf), 2)
        return {"sales_growth_sector_pct": avg_growth}
    except Exception as e:
        print(f"Error al obtener datos de Yahoo Finance: {e}")
        return {"sales_growth_sector_pct": None}


# ======================================
# 3. Integración y actualización en DB
# ======================================
def update_postgres_benchmark(data):
    """Actualiza la tabla ai.industry_benchmarks en PostgreSQL."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="require"
    )
    cur = conn.cursor()

    # Crear tabla si no existe
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai.industry_benchmarks (
        sector TEXT PRIMARY KEY,
        gdp_growth_pct NUMERIC,
        sales_growth_sector_pct NUMERIC,
        avg_ticket_growth_sector_pct NUMERIC,
        inflation_pct NUMERIC,
        unemployment_pct NUMERIC,
        reference_date DATE
    );
    """)

    # Insertar/actualizar fila
    cur.execute("""
        INSERT INTO ai.industry_benchmarks (
            sector, gdp_growth_pct, sales_growth_sector_pct,
            avg_ticket_growth_sector_pct, inflation_pct,
            unemployment_pct, reference_date
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (sector)
        DO UPDATE SET
            gdp_growth_pct = EXCLUDED.gdp_growth_pct,
            sales_growth_sector_pct = EXCLUDED.sales_growth_sector_pct,
            avg_ticket_growth_sector_pct = EXCLUDED.avg_ticket_growth_sector_pct,
            inflation_pct = EXCLUDED.inflation_pct,
            unemployment_pct = EXCLUDED.unemployment_pct,
            reference_date = EXCLUDED.reference_date;
    """, (
        "Landscaping Sector",
        data.get("gdp_growth_pct"),
        data.get("sales_growth_sector_pct"),
        data.get("avg_ticket_growth_sector_pct", 2.4),
        data.get("inflation_pct"),
        data.get("unemployment_pct"),
        datetime.date.today()
    ))

    conn.commit()
    conn.close()
    print(" Tabla ai.industry_benchmarks actualizada correctamente.")


# ===============================
# 4. Ejecución principal
# ===============================
def main():
    print("=== Actualizando contexto macro y sectorial ===")

    wb_data = fetch_worldbank_macro()
    yf_data = fetch_yahoo_sector()

    merged = {**wb_data, **yf_data, "avg_ticket_growth_sector_pct": 2.4}
    merged["reference_date"] = str(datetime.date.today())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    print(f" Archivo actualizado: {OUTPUT_PATH}")

    update_postgres_benchmark(merged)

    print("=== Actualización completa ===")
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
