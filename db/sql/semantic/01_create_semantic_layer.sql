-- ==============================================
-- 1. Crear schema y tablas de contexto (si no existen)
-- ==============================================

CREATE SCHEMA IF NOT EXISTS ai AUTHORIZATION datalysis_analytics_db_admin;

CREATE TABLE IF NOT EXISTS ai.table_context (
    table_name      TEXT PRIMARY KEY,
    table_type      TEXT CHECK (table_type IN ('fact','dimension')),
    business_domain TEXT,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS ai.column_context (
    id              SERIAL PRIMARY KEY,
    table_name      TEXT REFERENCES ai.table_context(table_name),
    column_name     TEXT,
    data_type       TEXT,
    business_meaning TEXT,
    example_value   TEXT
);

-- ==============================================
-- 2. Registrar tablas DIMENSIONES
-- ==============================================

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.dim_housecallpro_cities', 'dimension', 'Ubicación', 'Ciudades registradas en HousecallPro.')
ON CONFLICT DO NOTHING;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.dim_housecallpro_employees', 'dimension', 'Recursos Humanos', 'Empleados registrados en HousecallPro.')
ON CONFLICT DO NOTHING;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.dim_housecallpro_employees_with_jobs', 'dimension', 'Recursos Humanos', 'Relación de empleados con los trabajos que realizan.')
ON CONFLICT DO NOTHING;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.dim_housecallpro_job_estimate', 'dimension', 'Operaciones', 'Estimaciones asociadas a trabajos de HousecallPro.')
ON CONFLICT DO NOTHING;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.dim_housecallpro_lead_sources', 'dimension', 'Marketing', 'Fuentes de donde provienen los leads.')
ON CONFLICT DO NOTHING;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.dim_housecallpro_lead_sources_jobs', 'dimension', 'Marketing', 'Relación entre fuentes de leads y trabajos.')
ON CONFLICT DO NOTHING;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.dim_leads', 'dimension', 'Leads', 'Información de leads: datos personales, origen y estado.')
ON CONFLICT DO NOTHING;

-- ==============================================
-- 3. Registrar tablas HECHOS
-- ==============================================

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.fct_housecallpro_estimates', 'fact', 'Operaciones', 'Registro de estimaciones creadas en HousecallPro.')
ON CONFLICT DO NOTHING;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.fct_housecallpro_jobs', 'fact', 'Operaciones', 'Registro de trabajos realizados en HousecallPro.')
ON CONFLICT DO NOTHING;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.fct_housecallpro_sales', 'fact', 'Ventas', 'Ventas registradas en HousecallPro.')
ON CONFLICT DO NOTHING;

INSERT INTO ai.table_context (table_name, table_type, business_domain, description) VALUES
('clean.fct_sales', 'fact', 'Ventas', 'Registro de facturación general de RocknBlock.')
ON CONFLICT DO NOTHING;

-- ==============================================
-- 4. Ejemplos de columnas con significado
-- ==============================================

-- dim_leads
INSERT INTO ai.column_context (table_name, column_name, data_type, business_meaning, example_value) VALUES
('clean.dim_leads','lead_id','bigint','Identificador único del lead','101'),
('clean.dim_leads','first_name','text','Nombre del lead','John'),
('clean.dim_leads','status','text','Estado del lead (activo, perdido, convertido)','converted')
ON CONFLICT DO NOTHING;

-- fct_sales
INSERT INTO ai.column_context (table_name, column_name, data_type, business_meaning, example_value) VALUES
('clean.fct_sales','id_invoice','text','Identificador de la factura','349'),
('clean.fct_sales','total_invoice','numeric','Monto total de la factura','1136.25'),
('clean.fct_sales','city','text','Ciudad de la venta','Las Vegas')
ON CONFLICT DO NOTHING;

-- fct_housecallpro_jobs
INSERT INTO ai.column_context (table_name, column_name, data_type, business_meaning, example_value) VALUES
('clean.fct_housecallpro_jobs','job_id','bigint','Identificador único del trabajo','5501'),
('clean.fct_housecallpro_jobs','employee_id','bigint','Empleado asignado al trabajo','42'),
('clean.fct_housecallpro_jobs','status','text','Estado del trabajo (completado, pendiente, cancelado)','completed')
ON CONFLICT DO NOTHING;

-- dim_housecallpro_employees
INSERT INTO ai.column_context (table_name, column_name, data_type, business_meaning, example_value) VALUES
('clean.dim_housecallpro_employees','employee_id','bigint','Identificador único del empleado','42'),
('clean.dim_housecallpro_employees','name','text','Nombre completo del empleado','Alice Doe')
ON CONFLICT DO NOTHING;
