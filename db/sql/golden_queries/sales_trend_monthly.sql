-- KPI: Sales Trend Monthly
-- Description: Ventas totales por mes en el último año
-- Dominio: Ventas

SELECT 
    TO_CHAR(DATE_TRUNC('month', created_at), 'YYYY-MM') AS month,
    SUM(total_amount) AS total_sales
FROM clean.fct_housecallpro_jobs
WHERE work_status IN ('in progress', 'complete rated', 'complete unrated', 'scheduled', 'needs scheduling')
  AND created_at >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '12 months'
GROUP BY 1
ORDER BY 1 ASC;
