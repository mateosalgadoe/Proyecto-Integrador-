#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluación automática con Cohere Command A (LLM-as-a-Judge) sobre trazas de LangSmith.
Lee data/processed/langsmith_runs.json (agrupado por trace_id) y genera:
  - results/evals.csv   : fila por evaluación
  - results/summary.json: métricas agregadas
Usa ground truths almacenadas en data/ground_truths/<metric>.json.
Mide Execution Accuracy, KPI Correctness, Consistency Score, RMSE, Latencia y TTFT.
"""

import os
import re
import csv
import json
import time
import math
import argparse
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.append(str(ROOT_DIR))
load_dotenv()

# ===============================================================
# Funciones auxiliares
# ===============================================================

def normalize_numeric_tokens(text: str) -> Optional[float]:
    """
    Convierte expresiones numéricas con formato correcto.
    PRIORIZA primeras líneas del texto.
    """
    if not text:
        return None
    
    # Solo procesar primeras 2 líneas
    lines = text.strip().split('\n')
    t = '\n'.join(lines[:2])
    
    #  Detectar porcentajes SOLO si están al inicio
    pct_match = re.search(r'^.*?(\d+\.?\d*)\s*%', t, re.MULTILINE)
    if pct_match:
        return float(pct_match.group(1))
    
    # Quitar caracteres no numéricos comunes
    t = re.sub(r"[^\d,.\-kKmMbB]", "", t)

    # Detectar formato latino (punto miles, coma decimal)
    if re.search(r"\d{1,3}(\.\d{3})+,\d+", t):
        t = t.replace(".", "").replace(",", ".")
    # Detectar formato internacional (coma miles, punto decimal)
    elif re.search(r"\d{1,3}(,\d{3})+\.\d+", t):
        t = t.replace(",", "")
    # Solo comas sin puntos = decimales
    elif "," in t and "." not in t:
        t = t.replace(",", ".")

    # Unidades abreviadas
    match = re.search(r"([-+]?[0-9]*\.?[0-9]+)\s*([kKmMbB]?)", t)
    if not match:
        return None
    
    num = float(match.group(1))
    unit = match.group(2).lower()
    
    if unit == "k":
        num *= 1_000
    elif unit == "m":
        num *= 1_000_000
    elif unit == "b":
        num *= 1_000_000_000
    
    #  VALIDACIÓN: Rechazar si > 100M (probablemente error de parsing)
    if num > 100_000_000:
        print(f" Valor sospechoso: {num} (unidad: {unit})")
        return None
    
    return num
def try_parse_number(text: str) -> Optional[float]:
    """
    Extrae SOLO el primer número del KPI, ignorando todo lo demás.
    Prioriza: línea 1, formato "VALOR — descripción" o "KPI: VALOR"
    """
    if not text:
        return None

    # ESTRATEGIA ULTRA-ESPECÍFICA: Solo primera línea con formato KPI
    lines = text.strip().split('\n')
    first_line = lines[0] if lines else ""
    
    # PATRÓN 1: "12.0% — descripción" o "9231.61 Interpretación"
    match = re.search(r'^(\d+\.?\d*)\s*[%—\-\s]', first_line)
    if match:
        return float(match.group(1))
    
    # PATRÓN 2: "Resultado: 12.0" o "KPI: 9231.61"
    match = re.search(r'(?:resultado|kpi|valor|tasa|rate)[\s:=-]+(\d+\.?\d*)', first_line, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # PATRÓN 3: Número puro al inicio (como "9231.61")
    match = re.search(r'^(\d+\.\d+)(?!\d)', first_line)
    if match:
        return float(match.group(1))
    
    # PATRÓN 4: Porcentaje sin punto (como "23%")
    match = re.search(r'^(\d+)%', first_line)
    if match:
        return float(match.group(1))
    
    #  NO ENCONTRÓ NADA VÁLIDO
    return None

def parse_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().lower()
    if s == "now":
        return datetime.now(timezone.utc)
    if s.startswith("now-"):
        num = "".join(ch for ch in s[4:] if ch.isdigit())
        unit = "".join(ch for ch in s[4:] if ch.isalpha())
        if not num or not unit:
            return None
        num = int(num)
        if unit.startswith("d"):
            return datetime.now(timezone.utc) - timedelta(days=num)
        if unit.startswith("h"):
            return datetime.now(timezone.utc) - timedelta(hours=num)
        if unit.startswith("m"):
            return datetime.now(timezone.utc) - timedelta(minutes=num)
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def dt_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def is_in_range(dt_str: Optional[str], start: Optional[datetime], end: Optional[datetime]) -> bool:
    if not dt_str:
        return True
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return True
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True


# ===============================================================
# Carga y extracción de LangSmith
# ===============================================================

def load_grouped_runs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_question(doc: Dict[str, Any]) -> Optional[str]:
    root = doc.get("root") or {}
    q = (root.get("inputs") or {}).get("question")
    if q:
        return q
    for s in doc.get("steps", []):
        q = (s.get("inputs") or {}).get("question")
        if q:
            return q
    return None


def extract_model_output(doc: Dict[str, Any]) -> Optional[str]:
    """Busca el output del modelo priorizando kpi_numeric_value."""
    
    # 1. PRIORIDAD ABSOLUTA: buscar kpi_numeric_value primero
    steps = doc.get("steps", [])
    if steps:  # Verificar que steps no sea None o vacío
        for step in steps:
            if step is None:  # Skip None steps
                continue
            outputs = step.get("outputs", {})
            if outputs and outputs.get("kpi_numeric_value") is not None:
                val = outputs.get("kpi_numeric_value")
                print(f"Encontrado kpi_numeric_value: {val}")
                return str(val)
    
    # 2. Buscar en root
    root = doc.get("root", {})
    if root:
        outputs = root.get("outputs", {})
        if outputs and outputs.get("kpi_numeric_value") is not None:
            val = outputs.get("kpi_numeric_value")
            print(f"Encontrado kpi_numeric_value en root: {val}")
            return str(val)
    
    # 3. Buscar final_answer con patrones
    if steps:
        for step in steps:
            if step is None:  # Skip None steps
                continue
            outputs = step.get("outputs", {})
            if outputs and outputs.get("final_answer"):
                answer = outputs.get("final_answer")
                
                # Buscar patrones específicos de KPIs
                kpi_patterns = [
                    r'(?:KPI|valor|resultado).*?(\d{6,}(?:\.\d+)?)',  # Números grandes
                    r'las vegas nv:\s*(\d+\.?\d*)',  # Closing rate específico
                    r'(?:PIB|GDP).*?(\d+\.\d+)',  # PIB como benchmark
                ]
                
                for pattern in kpi_patterns:
                    match = re.search(pattern, answer, re.IGNORECASE)
                    if match:
                        found_val = match.group(1)
                        print(f"Encontrado KPI con patrón: {found_val}")
                        return found_val
                
                return answer
    
    # 4. Fallback a otros campos
    if root:
        outputs = root.get("outputs", {})
        if outputs:
            for k in ["draft_answer", "tool_result", "summary"]:
                if outputs.get(k):
                    print(f"Encontrado '{k}' en root.outputs")
                    return outputs.get(k)
    
    print("No se encontró model_output en ningún nivel")
    return None

def extract_ground_truth(doc: Dict[str, Any]) -> Optional[str]:
    """Busca ground truth (datos crudos) en: steps → data → rows."""

    # 1️⃣ Buscar en steps - nivel más probable para data / KPI results
    steps = doc.get("steps", [])
    print(f"Buscando ground_truth en {len(steps)} steps...")

    for idx, step in enumerate(steps):
        if step is None:  # Skip None steps
            continue
        step_name = step.get("name", f"step_{idx}")
        step_outputs = step.get("outputs") or {}

        # Buscar "data" dict (típicamente contiene columnas + rows)
        if step_outputs.get("data"):
            data = step_outputs.get("data")
            if isinstance(data, dict):
                rows = data.get("rows")
                cols = data.get("columns")

                if rows:
                    print(f"✅ Encontrado 'data.rows' en step '{step_name}' ({len(rows)} filas)")
                    # Formatea como tabla legible para el juez
                    lines = []
                    if cols:
                        lines.append(" | ".join(map(str, cols)))
                    for r in rows[:20]:  # Primeras 20 filas
                        if isinstance(r, (list, tuple)):
                            lines.append(" | ".join(map(str, r)))
                        else:
                            lines.append(str(r))
                    return "\n".join(lines)

        # Buscar "summary" (a veces es el ground truth narrativo)
        if step_outputs.get("summary"):
            print(f"✅ Encontrado 'summary' en step '{step_name}'")
            return step_outputs.get("summary")

        # Buscar "tool_result" (puede contener datos formateados)
        if step_outputs.get("tool_result"):
            print(f"✅ Encontrado 'tool_result' en step '{step_name}'")
            return step_outputs.get("tool_result")

    # 2️⃣ Buscar en root.outputs como fallback
    root = doc.get("root") or {}
    outputs = root.get("outputs") or {}

    for k in ["data", "summary", "tool_result", "draft_answer", "final_answer"]:
        if outputs.get(k):
            val = outputs.get(k)
            if isinstance(val, dict) and val.get("rows"):
                # Es un data dict
                print(f"✅ Encontrado '{k}' con datos en root.outputs")
                cols = val.get("columns", [])
                rows = val.get("rows", [])
                lines = []
                if cols:
                    lines.append(" | ".join(map(str, cols)))
                for r in rows[:20]:
                    if isinstance(r, (list, tuple)):
                        lines.append(" | ".join(map(str, r)))
                    else:
                        lines.append(str(r))
                return "\n".join(lines)
            elif isinstance(val, str):
                print(f"✅ Encontrado '{k}' (texto) en root.outputs")
                return val

    # 3️⃣ Buscar recursivamente en sub-steps
    for idx, step in enumerate(steps):
        step_name = step.get("name", f"step_{idx}")
        sub_steps = step.get("steps", [])

        if sub_steps:
            print(f"🔍 Explorando sub-steps de '{step_name}' ({len(sub_steps)} anidados)...")
            for sub_idx, sub_step in enumerate(sub_steps):
                sub_outputs = sub_step.get("outputs") or {}
                sub_name = sub_step.get("name", f"substep_{sub_idx}")

                if sub_outputs.get("data"):
                    data = sub_outputs.get("data")
                    if isinstance(data, dict) and data.get("rows"):
                        print(f"✅ Encontrado 'data' en sub-step '{sub_name}'")
                        cols = data.get("columns", [])
                        rows = data.get("rows", [])
                        lines = []
                        if cols:
                            lines.append(" | ".join(map(str, cols)))
                        for r in rows[:20]:
                            if isinstance(r, (list, tuple)):
                                lines.append(" | ".join(map(str, r)))
                            else:
                                lines.append(str(r))
                        return "\n".join(lines)

    print("❌ No se encontró ground_truth en ningún nivel")
    return None

# ===============================================================
# Carga de ground truths dinámicos
# ===============================================================

def load_ground_truth_for(metric: str) -> str:
    path = Path("data/ground_truths") / f"{metric}.json"
    if path.exists():
        return path.read_text(encoding="utf-8")
    for alt in [
        Path("data/ground_truths") / f"kpi_{metric}.json",
        Path("data/ground_truths") / f"{metric}_groundtruth.json",
    ]:
        if alt.exists():
            return alt.read_text(encoding="utf-8")
    return ""


# ===============================================================
# COHERE JUDGE - FIX KPI COLUMN PRIORITY
# ===============================================================


def cohere_judge(metric: str, question: str, model_output: str, ground_truth: str) -> Dict[str, Any]:
    import cohere

    api_key = os.getenv("COHERE_API_KEY")
    model = os.getenv("COHERE_JUDGE_MODEL", "command-a")
    co = cohere.Client(api_key)

    # Manejo de KPI numérico puro
    mo_num = None
    try:
        mo_num = float(model_output.strip())
        print(f"✅ Model output es número puro: {mo_num}")
    except ValueError:
        mo_norm = normalize_numeric_tokens(model_output)
        mo_num = mo_norm if mo_norm is not None else try_parse_number(model_output)
        if mo_num is not None:
            print(f"✅ Número extraído del texto: {mo_num}")

    gt_num = None

    # Caso especial: KPI global único (single_value)
    if '"type": "single_value"' in ground_truth:
        try:
            gt_json = json.loads(ground_truth)
            gt_num = gt_json.get("value")
            print(f"🌐 Ground truth global detectado: {gt_num}")
        except Exception as e:
            print(f"⚠️ Error leyendo KPI global: {e}")


    # Lógica estándar para KPIs tipo tabla con 'stats'
    if gt_num is None:
        try:
            gt_data = json.loads(ground_truth)
            if isinstance(gt_data, dict) and "stats" in gt_data:

                # 🆕 DETECCIÓN DE CIUDAD ESPECÍFICA EN LA PREGUNTA
                cities_in_question = []
            
                common_cities = ["las vegas", "san diego", "phoenix", "denver", "santa clara",
                                 "henderson", "riverside", "north las vegas", "temecula",
                                 "anaheim", "atlanta", "aliso viejo", "apache junction", "arizona"]
                
                for city in common_cities:
                    if city in question.lower():
                        cities_in_question.append(city)
                        print(f"🎯 Ciudad detectada en pregunta: {city}")


                        
    
                # Si hay ciudad específica mencionada Y el ground_truth tiene stats multi-ciudad
                if cities_in_question and gt_data.get("type") == "multi_value":
                    print(f"🔍 Intentando buscar valor de ciudad en ground_truth de tipo multi_value...")
                    
                    # 🆕 CARGAR EL GROUND TRUTH ORIGINAL CON ROWS
                    # El ground truth JSON no tiene rows, necesitamos cargarlas desde la BD
                    try:
                        import psycopg2
                        conn = psycopg2.connect(
                            host=os.getenv("DB_HOST"),
                            port=os.getenv("DB_PORT"),
                            dbname=os.getenv("DB_NAME"),
                            user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASSWORD"),
                            sslmode="require"
                        )
                        cur = conn.cursor()
                        
                        # Obtener el nombre de la vista KPI desde el ground truth
                        kpi_table = gt_data.get("kpi_name", "")
                        if kpi_table:
                            cur.execute(f"SELECT * FROM ai.{kpi_table} LIMIT 20;")
                            rows = cur.fetchall()
                            cols = [desc[0] for desc in cur.description]
                            
                            # Buscar la ciudad en las rows
                            for row in rows:
                                if row and len(row) > 0:
                                    city_in_row = str(row[0]).lower().strip()
                                    for city_q in cities_in_question:
                                        if city_q in city_in_row:
                                            # Tomar último valor numérico de la fila
                                            gt_num = float(row[-1])
                                            print(f"✅ Encontrado valor específico para '{city_q}' en BD: {gt_num}")
                                            conn.close()
                                            break
                                    if gt_num is not None:
                                        break
                            
                            conn.close()
                    except Exception as e:
                        print(f"⚠️ No se pudo consultar BD para ciudad específica: {e}")






                # Si NO se encontró valor de ciudad específica, usar columnas prioritarias
                if gt_num is None:
                    # ✅ LISTA DE PRIORIDAD DE COLUMNAS
                    priority_columns = [
                        "closing_rate_pct_global",   # Para closing rate global
                        "closing_rate_pct",          # Para closing rate por ciudad
                        "avg_sales_volume_global",   # Para volumen global
                        "gdp_growth_pct",            # Para benchmarks macro
                        "sales_growth_sector_pct",   # Para benchmarks sector
                        "total_sales",               # Para totales
                        "avg_sale_volume",           # Para promedios
                    ]

                    # Buscar por prioridad
                    for priority_col in priority_columns:
                        if priority_col in gt_data["stats"]:
                            stats = gt_data["stats"][priority_col]
                            gt_num = stats.get("median") or stats.get("mean")
                            print(f"✅ Usando columna prioritaria '{priority_col}': {gt_num}")
                            break

                    # Si no encontró columna prioritaria, buscar avg/mean genérico
                    if gt_num is None:
                        for col, stats in gt_data["stats"].items():
                            col_lower = col.lower()
                            if any(kw in col_lower for kw in ["avg", "mean", "average", "promedio"]):
                                gt_num = stats.get("median") or stats.get("mean")
                                print(f"✅ Usando columna promedio '{col}': {gt_num}")
                                break

                    # Último recurso
                    if gt_num is None and gt_data["stats"]:
                        first_col = next(iter(gt_data["stats"].keys()))
                        first_stat = gt_data["stats"][first_col]
                        gt_num = first_stat.get("median") or first_stat.get("mean")
                        print(f"⚠️ Último recurso: {first_col} → {gt_num}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ No se pudo parsear ground_truth como JSON: {e}")

    # Fallback si no se encontró valor numérico
    if gt_num is None:
        try:
            gt_num = float(ground_truth.strip())
            print(f"✅ Ground truth es número puro: {gt_num}")
        except ValueError:
            gt_norm = normalize_numeric_tokens(ground_truth)
            gt_num = gt_norm if gt_norm is not None else try_parse_number(ground_truth)
            if gt_num is not None:
                print(f"✅ Número extraído del ground truth: {gt_num}")

    # Comparación numérica
    print(f"🔍 DEBUG FINAL: mo_num={mo_num}, gt_num={gt_num}")

    if mo_num is not None and gt_num is not None:
        diff_pct = abs(mo_num - gt_num) / gt_num * 100 if gt_num != 0 else float('inf')
        print(f"🔍 Diferencia: {diff_pct:.2f}%")
        comparison_hint = (
            f"Compara estos valores numéricos: salida={mo_num:.4f} y referencia={gt_num:.4f}. "
            f"La diferencia es {diff_pct:.2f}%. "
            "Considera 'correct' si la diferencia es menor al 20%. "
            "Si la diferencia es 20-40%, considera 'partial'. Si >40%, considera 'incorrect'."
        )
    else:
        comparison_hint = (
            "Compara el contenido textual general. Considera 'correct' si la respuesta es "
            "coherente con el ground truth."
        )

    # Envío a Cohere Judge
    system = (
        "Eres un juez experto en validación de KPIs. Devuelve solo un JSON con los campos "
        "{'verdict':'correct'|'partial'|'incorrect','score':float,'justification':str}. "
        "Score debe ser: 1.0 para 'correct', 0.5 para 'partial', 0.0 para 'incorrect'."
    )

    user = (
        f"[Métrica] {metric}\n[Pregunta]\n{question}\n\n"
        f"[Respuesta del modelo]\n{model_output[:500]}...\n\n"
        f"[Ground truth reference]\n{ground_truth[:500]}...\n\n"
        f"Instrucción: {comparison_hint}\nDevuelve solo el JSON."
    )

    # Streaming con medición TTFT
    t0 = time.time()
    t_first = None
    chunks = []

    print("\n" + "="*50)
    print(f"EVALUANDO: {metric.upper()}")
    print("="*50)

    try:
        stream = co.chat_stream(
            model=model,
            message=user,
            preamble=system,
            temperature=0.0,
        )
        for ev in stream:
            if hasattr(ev, 'event_type'):
                if ev.event_type == "text-generation":
                    if t_first is None:
                        t_first = time.time()
                    chunks.append(ev.text)
                elif ev.event_type == "stream-end":
                    break
            elif hasattr(ev, 'text'):
                if t_first is None:
                    t_first = time.time()
                chunks.append(ev.text)
    except Exception as stream_err:
        print(f"❌ Error en streaming: {stream_err}")
        try:
            response = co.chat(
                model=model,
                message=user,
                preamble=system,
                temperature=0.0,
            )
            t_first = time.time()
            chunks.append(response.text)
        except Exception as chat_err:
            print(f"❌ Error en chat: {chat_err}")
            raise

    t1 = time.time()
    content = "".join(chunks).strip()

    # Limpieza de JSON
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content).rstrip("`").strip()

    # Buscar objeto JSON dentro del texto
    json_match = re.search(r'\{[^{}]*(?:"verdict"[^{}]*"score"[^{}]*)?[^{}]*\}', content, re.DOTALL)
    if json_match:
        content = json_match.group(0)

    try:
        verdict = json.loads(content)
        print(f"✅ Veredicto: {verdict.get('verdict')} (score={verdict.get('score')})")
    except json.JSONDecodeError:
        # Conversión silenciosa de comillas simples a dobles
        try:
            content_fixed = content.replace("'", '"')
            verdict = json.loads(content_fixed)
            print(f"✅ Veredicto: {verdict.get('verdict')} (score={verdict.get('score')})")
        except json.JSONDecodeError:
            # Fallback manual silencioso
            try:
                verdict_match = re.search(r'"?verdict"?\s*:\s*"?([^",}]+)"?', content)
                score_match = re.search(r'"?score"?\s*:\s*([\d.]+)', content)

                if verdict_match and score_match:
                    extracted_verdict = verdict_match.group(1).strip().strip('"\'')
                    extracted_score = float(score_match.group(1))
                    print(f"✅ Veredicto: {extracted_verdict} (score={extracted_score})")
                    verdict = {
                        "verdict": extracted_verdict,
                        "score": extracted_score,
                        "justification": "Extraído correctamente"
                    }
                else:
                    raise ValueError("No se pudieron extraer verdict/score")
            except Exception:
                print(f"⚠️ No se pudo parsear respuesta del juez")
                verdict = {"verdict": "incorrect", "score": 0.0, "justification": "parse_error"}

    return {
        "raw": content,
        "verdict": verdict.get("verdict", "incorrect"),
        "score": float(verdict.get("score", 0.0)),
        "justification": verdict.get("justification", ""),
        "latency_ms": int((t1 - t0) * 1000),
        "ttft_ms": int((t_first - t0) * 1000) if t_first else None,
    }

# ===============================================================
# Métricas auxiliares
# ===============================================================

def compute_consistency(rows: List[Dict[str, Any]]) -> Optional[float]:
    from collections import defaultdict, Counter
    groups = defaultdict(list)
    for r in rows:
        if r.get("metric") == "execution_accuracy" and r.get("question"):
            groups[r["question"].strip().lower()].append(r["verdict"])
    scores = []
    for q, vs in groups.items():
        if len(vs) < 2:
            continue
        c = Counter(vs)
        scores.append(max(c.values()) / len(vs))
    if not scores:
        return None
    return sum(scores) / len(scores)

def compute_rmse(rows: List[Dict[str, Any]]) -> Optional[float]:
    diffs = []
    for r in rows:
        if r.get("metric") != "execution_accuracy":
            continue
        mo = try_parse_number(r.get("model_output"))
        gt = try_parse_number(r.get("ground_truth"))
        if mo is not None and gt is not None:
            diffs.append((mo - gt) ** 2)
    if not diffs:
        return None
    return math.sqrt(sum(diffs) / len(diffs))


# ===============================================================
# Main
# ===============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", default="data/processed/langsmith_runs.json")
    p.add_argument("--out", dest="output_csv", default="results/evals.csv")
    p.add_argument("--summary", dest="summary_json", default="results/summary.json")
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    p.add_argument("--metrics", default="execution_accuracy,kpi_correctness")
    args = p.parse_args()

    since_dt = parse_time(args.since)
    until_dt = parse_time(args.until)
    docs = load_grouped_runs(args.input_path)

    eval_inputs = []
    for d in docs:
        started_at = d.get("started_at")
        if not is_in_range(started_at, since_dt, until_dt):
            continue
        q = extract_question(d)
        mo = extract_model_output(d)
        gt = extract_ground_truth(d)
        if not (q and mo and gt):
            continue
        eval_inputs.append({
            "trace_id": d.get("trace_id"),
            "question": q,
            "model_output": mo,
            "ground_truth": gt,
            "started_at": started_at,
            "ended_at": d.get("ended_at"),
        })

    if not eval_inputs:
        print("No hay trazas evaluables.")
        return

    selected_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    os.makedirs(Path(args.output_csv).parent, exist_ok=True)

    out_rows = []
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_id","started_at","ended_at","metric","verdict","score",
                "justification","latency_ms","ttft_ms","question","model_output","ground_truth"
            ],
        )
        writer.writeheader()

        for row in eval_inputs:
            q, mo, gt_text = row["question"], row["model_output"], row["ground_truth"]
            
            kpi_candidates = {
                # GLOBALES (más largos = mayor prioridad)
                "promedio global de ventas": "kpi_avg_sales_volume_global",
                "volumen de ventas global": "kpi_avg_sales_volume_global",
                "closing rate global de rocknblock": "kpi_closing_rate_global",
                "tasa de cierre global de rocknblock": "kpi_closing_rate_global",
                "closing rate global": "kpi_closing_rate_global",
                "tasa de cierre global": "kpi_closing_rate_global",
                "volumen global": "kpi_avg_sales_volume_global",
                "promedio global": "kpi_avg_sales_volume_global",
                
                # POR CIUDAD (frases específicas - MÁS LARGAS PRIMERO)
                "promedio de ventas por ciudad": "kpi_avg_sales_volume",
                "total de ventas por ciudad": "kpi_sales_volume_by_city",
                "ventas por ciudad": "kpi_sales_volume_by_city",
                "promedio por ciudad": "kpi_avg_sales_volume",
                "closing rate de las vegas": "kpi_closing_rate",
                "closing rate en las vegas": "kpi_closing_rate",
                "tasa de cierre de las vegas": "kpi_closing_rate",
                "closing rate de": "kpi_closing_rate",
                "closing rate en": "kpi_closing_rate",
                
                # BENCHMARKS
                "benchmarks de crecimiento de pib": "industry_benchmarks",
                "crecimiento de pib en la industria": "industry_benchmarks",
                "benchmarks de crecimiento": "industry_benchmarks",
                "crecimiento de pib": "industry_benchmarks",
                "benchmarks de industria": "industry_benchmarks",
                "industry": "industry_benchmarks",
                "benchmark": "industry_benchmarks",
            }

            kpi_match = None
            
            # Buscar match más largo primero (más específico)
            for key in sorted(kpi_candidates.keys(), key=len, reverse=True):
                if key in q.lower():
                    kpi_match = kpi_candidates[key]
                    print(f" KPI detectado: {kpi_match}")
                    break


            for metric in selected_metrics:
                gt_json = load_ground_truth_for(kpi_match or metric)
                gt_combined = gt_json or gt_text

                try:
                    res = cohere_judge(metric, q, mo, gt_combined)
                except Exception as e:
                    res = {
                        "verdict": "incorrect", "score": 0.0,
                        "justification": f"judge_error:{e}",
                        "latency_ms": None, "ttft_ms": None,
                    }

                writer.writerow({
                    "trace_id": row["trace_id"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "metric": metric,
                    "verdict": res["verdict"],
                    "score": res["score"],
                    "justification": res["justification"],
                    "latency_ms": res["latency_ms"],
                    "ttft_ms": res["ttft_ms"],
                    "question": q,
                    "model_output": mo,
                    "ground_truth": gt_combined,
                })
                out_rows.append({
                    "metric": metric,
                    "question": q,
                    "verdict": res["verdict"],
                    "score": res["score"],
                    "model_output": mo,
                    "ground_truth": gt_combined,
                })

    ea_rows = [r for r in out_rows if r["metric"] == "execution_accuracy"]
    kpi_rows = [r for r in out_rows if r["metric"] == "kpi_correctness"]

    def pct_correct(rows):
        if not rows:
            return None
        return sum(1 for r in rows if r["verdict"] == "correct") / len(rows)

    summary = {
        "since": dt_iso(since_dt),
        "until": dt_iso(until_dt),
        "counts": {
            "total_traces": len(eval_inputs),
            "total_rows": len(out_rows),
            "execution_accuracy_rows": len(ea_rows),
            "kpi_correctness_rows": len(kpi_rows),
        },
        "metrics": {
            "execution_accuracy": pct_correct(ea_rows),
            "kpi_correctness": pct_correct(kpi_rows),
            "consistency_score": compute_consistency(out_rows),
            "rmse": compute_rmse(out_rows),
        },
    }

    os.makedirs(Path(args.summary_json).parent, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Evaluaciones escritas en: {args.output_csv}")
    print(f"📊 Resumen agregado en: {args.summary_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
