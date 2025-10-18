-- KPI: Closing Ratio Global
-- Description: Tasa de cierre global (Jobs / Estimates)
-- Fuente: Adaptado de getClosingRatio()
-- Dominio: Ventas

WITH jobs AS (
    SELECT COUNT(DISTINCT job_id) AS total_jobs
    FROM clean.fct_housecallpro_jobs
    WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
      AND work_status IN ('in progress', 'complete rated', 'complete unrated', 'scheduled', 'needs scheduling')
),
estimates AS (
    SELECT COUNT(DISTINCT estimate_group_id) AS total_estimates
    FROM clean.fct_housecallpro_estimates
    WHERE schedule_start >= CURRENT_DATE - INTERVAL '30 days'
      AND work_status NOT IN ('user canceled', 'pro canceled')
)
SELECT
    total_jobs,
    total_estimates,
    ROUND((total_jobs::decimal / NULLIF(total_estimates, 0)) * 100, 2) AS closing_ratio
FROM jobs, estimates;
