import os, json, datetime
import yfinance as yf
import wbdata
import pandas as pd

CACHE_PATH = "data/external/industry_context.json"
CACHE_DAYS = 7  # tiempo de validez (en días)


def fetch_from_worldbank():
    """Obtiene datos macroeconómicos recientes del Banco Mundial."""
    try:
        indicators = {
            "NY.GDP.MKTP.KD.ZG": "GDP Growth (%)",
            "FP.CPI.TOTL.ZG": "Inflation (%)",
            "SL.UEM.TOTL.ZS": "Unemployment (%)"
        }
        df = wbdata.get_dataframe(indicators, country="US", convert_date=True)
        df = df.sort_index().tail(1).iloc[0]
        return {
            "GDP Growth (%)": round(float(df["GDP Growth (%)"]), 2),
            "Inflation (%)": round(float(df["Inflation (%)"]), 2),
            "Unemployment (%)": round(float(df["Unemployment (%)"]), 2)
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_from_yfinance():
    """Obtiene retornos de las principales empresas/ETFs del sector landscaping."""
    tickers = ["SITE", "TBLD", "TREX"]
    sector_data = []
    for t in tickers:
        try:
            data = yf.Ticker(t).history(period="6mo")
            if not data.empty:
                change = ((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1) * 100
                sector_data.append({"symbol": t, "6m_return_%": round(change, 2)})
        except Exception as e:
            sector_data.append({"symbol": t, "error": str(e)})
    return sector_data


def get_industry_context(force_refresh=False):
    """
    Devuelve el contexto macro/microeconómico.
    Si existe cache reciente, lo carga.
    """
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    # Intentar usar cache si no se fuerza refresh
    if not force_refresh and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                cache = json.load(f)
            last_update = datetime.datetime.fromisoformat(cache.get("last_update"))
            if (datetime.datetime.now() - last_update).days < CACHE_DAYS:
                return format_industry_summary(cache)
        except Exception:
            pass  # si el cache está dañado, continúa con fetch

    # Si no hay cache o está vencido, obtener datos nuevos
    macro = fetch_from_worldbank()
    sector = fetch_from_yfinance()
    data = {
        "last_update": datetime.datetime.now().isoformat(),
        "macro": macro,
        "sector": sector
    }

    # Guardar cache
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)

    return format_industry_summary(data)


def format_industry_summary(data: dict):
    """Formatea el contexto de industria para integrarlo al business_context."""
    macro = data.get("macro", {})
    sector = data.get("sector", [])
    gdp = macro.get("GDP Growth (%)", "N/A")
    infl = macro.get("Inflation (%)", "N/A")
    unemp = macro.get("Unemployment (%)", "N/A")

    summary = f"""
Contexto macroeconómico (EE.UU.):
- Crecimiento PIB: {gdp}%
- Inflación: {infl}%
- Desempleo: {unemp}%

Sector Landscaping (Yahoo Finance):
"""
    for s in sector:
        if "6m_return_%" in s:
            summary += f"- {s['symbol']}: {s['6m_return_%']}% (6m)\n"
        else:
            summary += f"- {s['symbol']}: error ({s.get('error')})\n"

    summary += f"\nÚltima actualización: {data.get('last_update', 'desconocida')}\n"
    summary += "Fuente: Banco Mundial, Yahoo Finance."
    return summary.strip()

