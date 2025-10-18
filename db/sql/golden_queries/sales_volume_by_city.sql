-- KPI: Sales Volume by City
-- Description: Total de ventas agrupadas por ciudad (últimos 30 días)
-- Fuente: Adaptado de getSaleVolume()
-- Dominio: Ventas

WITH jobs_in_date_range AS (
    SELECT 
        city,
        SUM(total_amount) AS total_sales,
        COUNT(DISTINCT job_id) AS num_jobs,
        ROUND(SUM(total_amount) / NULLIF(COUNT(DISTINCT job_id), 0), 2) AS avg_sales
    FROM clean.fct_housecallpro_jobs
    WHERE work_status IN ('in progress', 'complete rated', 'complete unrated', 'scheduled', 'needs scheduling')
      AND created_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY city
)
SELECT city, total_sales, num_jobs, avg_sales
FROM jobs_in_date_range
ORDER BY total_sales DESC;
