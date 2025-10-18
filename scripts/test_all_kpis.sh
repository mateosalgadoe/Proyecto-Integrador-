#!/bin/bash

# Script para probar todos los KPIs

echo "Testing todos los KPIs de RocknBlock"
echo "=========================================="

# Activa el entorno
source .venv/bin/activate

# Arrays de preguntas para cada KPI
declare -A TESTS

# KPI 1: Global Sales Volume
TESTS["global"]="¿Cuál es el volumen global de ventas promedio?"

# KPI 2: Sales by City
TESTS["by_city"]="¿Cuál es el total de ventas por ciudad?"

# KPI 3: Closing Rate
TESTS["closing"]="¿Cuál es la tasa de cierre?"

# KPI 4: Tendencia Mensual
TESTS["trend"]="¿Cuál es la tendencia de ventas mensual?"

# KPI 5: Benchmarks
TESTS["benchmark"]="¿Cómo estamos vs la industria?"

# Contador
count=0

# Itera sobre cada test
for key in "${!TESTS[@]}"; do
    count=$((count + 1))
    question="${TESTS[$key]}"
    
    echo ""
    echo "Test $count: $key"
    echo "Pregunta: $question"
    echo "---"
    
    # Ejecuta el grafo
    python3 -m src.agents.langgraph.react_graph "$question"
    
    echo ""
done

echo ""
echo "Todos los tests completados"
echo ""
echo "Exportando resultados..."

# Exporta desde LangSmith
python3 scripts/export_langsmith_runs.py \
    --project rocknblock-ai \
    --since now-1h \
    --out data/processed/langsmith_runs_kpis.json

# Evalúa
python3 scripts/eval_llm_judge.py \
    --since now-1h \
    --out results/evals_kpis_all.csv \
    --summary results/summary_kpis_all.json

echo ""
echo " Resumen de evaluación:"
cat results/summary_kpis_all.json
