# Catálogo de KPIs - RocknBlock

## KPIs Implementados (7)

### 1. kpi_avg_sales_volume_global
- **Descripción:** Promedio global del volumen de ventas
- **Tipo:** Métrica única
- **Alias:** "promedio global", "volumen global"
- **Estado:** ✅ PROBADO (100% accuracy)
- **Ejemplo:** "¿Cuál es el volumen global de ventas promedio?"

### 2. kpi_avg_sales_volume
- **Descripción:** Promedio de volumen de ventas por ciudad
- **Tipo:** Multi-valor (por ciudad)
- **Alias:** "promedio por ciudad"
- **Estado:** ⚠️ NO PROBADO
- **Ejemplo:** "¿Cuál es el promedio de ventas por ciudad?"

### 3. kpi_closing_rate
- **Descripción:** Tasa de cierre (porcentaje de leads convertidos en trabajos)
- **Tipo:** Multi-valor (por ciudad)
- **Alias:** "closing rate", "tasa de cierre"
- **Estado:** ⚠️ NO PROBADO
- **Ejemplo:** "¿Cuál es la tasa de cierre?"

### 4. kpi_sales_volume_by_city
- **Descripción:** Total de ventas agrupadas por ciudad (últimos 30 días)
- **Tipo:** Multi-valor
- **Alias:** "ventas por ciudad", "total por ciudad"
- **Estado:** ⚠️ NO PROBADO
- **Ejemplo:** "¿Cuál es el total de ventas por ciudad?"

### 5. Sales Trend Monthly
- **Descripción:** Ventas totales por mes
- **Tipo:** Serie temporal
- **Alias:** "tendencia mensual", "evolución mensual"
- **Estado:** ⚠️ NO PROBADO
- **Ejemplo:** "¿Cuál es la tendencia de ventas mensual?"

### 6. Sales Volume by City
- **Descripción:** Total de ventas agrupadas por ciudad (últimos 30 días)
- **Tipo:** Multi-valor
- **Alias:** "sales por ciudad"
- **Estado:** ⚠️ NO PROBADO
- **Ejemplo:** "¿Cuál es el volumen de ventas por ciudad?"

### 7. industry_benchmarks
- **Descripción:** Benchmarks macro y sectoriales del sector Landscaping
- **Tipo:** Contexto
- **Alias:** "industria", "benchmark"
- **Estado:** ⚠️ NO PROBADO
- **Ejemplo:** "¿Cómo estamos vs la industria?"

## KPIs Propuestos pero NO Implementados (11)

- Average Sales per Rep
- Average Profit per Rep
- Lead Cost (global, por fuente)
- Total Investment per source
- Sold rate
- Appointments Scheduled
- Conversion % lead → scheduler
- Cuentas por cobrar (aging, DSO)
- Rentabilidad por cliente
- Utilidad neta consolidada
- Cash flow operativo mensual

## Plan de Testing

### Fase 1: Validar los 7 KPIs existentes
- Ejecutar preguntas para cada KPI
- Comparar salida vs ground truth
- Documentar accuracy

### Fase 2: Crear los 11 KPIs faltantes
- Definir SQL/vistas
- Agregar a ai.table_context
- Registrar en KPI_ALIASES

### Fase 3: Evaluar conjunto completo
- Ejecutar suite de preguntas
- Medir consistency_score
- Generar reporte final
