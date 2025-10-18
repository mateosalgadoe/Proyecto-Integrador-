CREATE SCHEMA IF NOT EXISTS ai;

CREATE OR REPLACE VIEW ai.industry_benchmarks AS
SELECT
    'Landscaping Sector' AS sector,
    3.2::numeric AS gdp_growth_pct,
    5.8::numeric AS sales_growth_sector_pct,
    2.4::numeric AS avg_ticket_growth_sector_pct,
    3.1::numeric AS inflation_pct,
    3.7::numeric AS unemployment_pct,
    CURRENT_DATE AS reference_date;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description)
VALUES (
    'industry_benchmarks',
    'kpi',
    'Sector',
    'Benchmarks macro y sectoriales del sector Landscaping (PIB, ventas, ticket, inflación, desempleo).'
)
ON CONFLICT (table_name) DO NOTHING;
