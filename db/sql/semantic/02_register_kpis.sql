-- =========================================================
--  RocknBlock KPI Layer Registration
-- Author: Mateo Salgado Espinosa
-- Date: 2025-10-10
-- =========================================================

--  1️ KPI 1: Total de Ventas por Ciudad
CREATE OR REPLACE VIEW ai.kpi_sales_volume_by_city AS
SELECT
    city,
    SUM(total_invoice) AS total_sales
FROM clean.fct_sales
GROUP BY city
ORDER BY total_sales DESC;

-- Registrar en la capa semántica
INSERT INTO ai.table_context (table_name, table_type, business_domain, description)
VALUES (
    'kpi_sales_volume_by_city',
    'kpi',
    'Ventas',
    'Total de ventas agrupadas por ciudad según las facturas registradas en fct_sales.'
)
ON CONFLICT (table_name) DO NOTHING;

--  2️ KPI 2: Promedio de Volumen de Ventas (Ticket Promedio)
CREATE OR REPLACE VIEW ai.kpi_avg_sales_volume AS
SELECT
    city,
    ROUND(AVG(total_invoice), 2) AS avg_sale_volume
FROM clean.fct_sales
GROUP BY city
ORDER BY avg_sale_volume DESC;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description)
VALUES (
    'kpi_avg_sales_volume',
    'kpi',
    'Ventas',
    'Promedio del monto de venta (ticket promedio) por ciudad en el último periodo registrado.'
)
ON CONFLICT (table_name) DO NOTHING;

-- 3️ KPI 3: Tasa de Cierre de Leads (Closing Rate)
CREATE OR REPLACE VIEW ai.kpi_closing_rate AS
WITH leads_by_city AS (
  SELECT 
    -- Quitar espacios y sufijos de estado (con o sin coma)
    REGEXP_REPLACE(
      LOWER(TRIM(city)), 
      '[\s,]+(al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|wy)$', 
      ''
    ) AS city_clean,
    COUNT(DISTINCT lead_id) as total_leads
  FROM clean.dim_leads
  WHERE city IS NOT NULL
    AND LENGTH(city) > 2
    AND LOWER(city) NOT IN ('.', 'none', 'unknown', 'null')
  GROUP BY city_clean
),
sales_by_city AS (
  SELECT 
    -- Quitar espacios y sufijos de estado (con o sin coma)
    REGEXP_REPLACE(
      LOWER(TRIM(city)), 
      '[\s,]+(al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|wy)$', 
      ''
    ) AS city_clean,
    COUNT(DISTINCT customer_id) as total_conversions
  FROM clean.fct_housecallpro_jobs j
  INNER JOIN clean.fct_housecallpro_sales s 
    ON j.job_id = s.job_id
  WHERE j.city IS NOT NULL
    AND s.status = 'paid'
  GROUP BY city_clean
)
SELECT 
  l.city_clean,
  l.total_leads,
  COALESCE(s.total_conversions, 0) as total_conversions,
  ROUND(
    (COALESCE(s.total_conversions, 0)::FLOAT 
    / NULLIF(l.total_leads, 0) * 100)::numeric, 
    2
  ) AS closing_rate_pct
FROM leads_by_city l
LEFT JOIN sales_by_city s 
  ON l.city_clean = s.city_clean
WHERE l.total_leads >= 5
ORDER BY closing_rate_pct DESC;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description)
VALUES (
  'kpi_closing_rate',
  'kpi',
  'Leads / Ventas',
  'Porcentaje de leads que se convirtieron en ventas pagadas (closing rate) por ciudad'
)
ON CONFLICT (table_name) DO NOTHING;

-- =========================================================
-- Verificación de registro
-- =========================================================
SELECT * FROM ai.table_context WHERE table_type = 'kpi';
