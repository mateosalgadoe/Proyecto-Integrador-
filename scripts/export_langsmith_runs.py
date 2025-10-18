#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exporta ejecuciones (runs) desde LangSmith a un JSON listo para RAG.
- Agrupa por trace_id (una "conversación/ejecución" completa).
- Incluye el run raíz + todos los sub-steps (tool/llm/chain) ordenados por tiempo.
- Filtra por proyecto y rango de fechas.
Uso:
  python3 scripts/export_langsmith_runs.py --since 2025-10-01 --until now --out data/processed/langsmith_runs.json
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Iterable

from dotenv import load_dotenv
from langsmith import Client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export LangSmith runs to JSON")
    p.add_argument("--project", default=os.getenv("LANGCHAIN_PROJECT"), help="LangSmith project name")
    p.add_argument("--since", default=None, help="Start date (YYYY-MM-DD) or 'now-7d'/'now-24h'/None")
    p.add_argument("--until", default=None, help="End date (YYYY-MM-DD) or 'now'")
    p.add_argument("--out", default="data/processed/langsmith_runs.json", help="Output JSON file")
    p.add_argument("--include-all", action="store_true",
                   help="If set, include all runs; default exports grouped traces (root + children).")
    return p.parse_args()


def parse_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().lower()
    if s == "now":
        return datetime.now(timezone.utc)
    # soportes simples: now-7d, now-24h
    if s.startswith("now-"):
        num = "".join(ch for ch in s[4:] if ch.isdigit())
        unit = "".join(ch for ch in s[4:] if ch.isalpha())
        if not num or not unit:
            return None
        num = int(num)
        from datetime import timedelta
        if unit in ("d", "day", "days"):
            return datetime.now(timezone.utc) - timedelta(days=num)
        if unit in ("h", "hour", "hours"):
            return datetime.now(timezone.utc) - timedelta(hours=num)
        if unit in ("m", "min", "mins", "minutes"):
            return datetime.now(timezone.utc) - timedelta(minutes=num)
        return None
    # formato YYYY-MM-DD
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def dt_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def serialize_run(run) -> Dict[str, Any]:
    """Convierte un run de LangSmith en un dict limpio y compacto."""
    # Atributos habituales disponibles en el SDK
    return {
        "id": str(getattr(run, "id", None)),
        "trace_id": str(getattr(run, "trace_id", None)),
        "parent_run_id": str(getattr(run, "parent_run_id", None)),
        "name": getattr(run, "name", None),
        "run_type": getattr(run, "run_type", None),
        "start_time": dt_iso(getattr(run, "start_time", None)),
        "end_time": dt_iso(getattr(run, "end_time", None)),
        "extra": getattr(run, "extra", None),            # metadatos adicionales
        "tags": getattr(run, "tags", None),
        "inputs": getattr(run, "inputs", None),
        "outputs": getattr(run, "outputs", None),
        "error": getattr(run, "error", None),
        "status": getattr(run, "status", None),
    }


def order_by_time(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        runs,
        key=lambda r: (r.get("start_time") or "", r.get("end_time") or "")
    )


def main():
    load_dotenv()
    args = parse_args()

    if not os.getenv("LANGCHAIN_API_KEY"):
        print(" Falta LANGCHAIN_API_KEY en el entorno/.env", file=sys.stderr)
        sys.exit(1)

    if not args.project:
        print(" Debes indicar --project o definir LANGCHAIN_PROJECT en el .env", file=sys.stderr)
        sys.exit(1)

    since_dt = parse_time(args.since)
    until_dt = parse_time(args.until)

    client = Client(
        api_key=os.getenv("LANGCHAIN_API_KEY"),
        api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    )

    print(f" Proyecto: {args.project}")
    print(f" Rango:   desde={dt_iso(since_dt) or 'None'}  hasta={dt_iso(until_dt) or 'None'}")
    print("onsultando runs en LangSmith...")

    # Trae runs del proyecto (paginado interno manejado por SDK).
    # Nota: el SDK permite filtrar por project_name y rango temporal.
    runs = list(client.list_runs(
        project_name=args.project,
        start_time=since_dt,
        end_time=until_dt,
        # Puedes añadir run_type=["chain","tool","llm"] si quieres filtrar
    ))

    if not runs:
        print("Xx No se recuperaron runs para ese rango/proyecto.")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(f" Archivo vacío escrito en {args.out}")
        return

    print(f"runs totales descargados: {len(runs)}")

    # Modo 1: exportar todo tal cual cada run como registro
    if args.include_all:
        all_serialized = [serialize_run(r) for r in runs]
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(order_by_time(all_serialized), f, ensure_ascii=False, indent=2)
        print(f" Exportar todos los   runs → {args.out}  (registros={len(all_serialized)})")
        return

    # Modo 2 (por defecto): agrupar por trace_id que es un root + children
    # 1) Identificar todos los trace_ids
    by_trace: Dict[str, List[Any]] = {}
    for r in runs:
        tid = str(getattr(r, "trace_id", None))
        if not tid:
            # si no hay trace_id, lo agrupamos por id individual
            tid = f"no-trace-{getattr(r, 'id', 'unknown')}"
        by_trace.setdefault(tid, []).append(r)

    grouped_docs: List[Dict[str, Any]] = []

    for trace_id, group in by_trace.items():
        # 2) Serializar todos los runs de este trace
        ser = [serialize_run(x) for x in group]

        # 3) Encontrar el "root" (sin parent_run_id)
        roots = [x for x in ser if not x.get("parent_run_id")]
        root = roots[0] if roots else None

        # 4) Orden cronológico
        ser_sorted = order_by_time(ser)

        # 5) Armar documento agregado para RAG
        doc = {
            "trace_id": trace_id,
            "project": args.project,
            "root": root,                 # puede ser None si no hay raíz clara
            "steps": ser_sorted,          # todos los sub-pasos y root ordenados
            "started_at": ser_sorted[0]["start_time"] if ser_sorted else None,
            "ended_at": ser_sorted[-1]["end_time"] if ser_sorted else None,
        }

        # 6) Campos útiles para RAG (si hay root con IO)
        if root:
            doc["root_input"] = root.get("inputs")
            doc["root_output"] = root.get("outputs")

        grouped_docs.append(doc)

    grouped_docs = sorted(grouped_docs, key=lambda d: d.get("started_at") or "")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(grouped_docs, f, ensure_ascii=False, indent=2)

    print(f"Export grouped traces → {args.out}  (trazas={len(grouped_docs)})")
    # Pequeño resumen
    roots_count = sum(1 for d in grouped_docs if d.get("root"))
    print(f"   • Con root: {roots_count} | Sin root: {len(grouped_docs)-roots_count}")


if __name__ == "__main__":
    main()
