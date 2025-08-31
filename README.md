# Requisitos
Python 3.10+

PostgreSQL (local o en la nube)

(Opcional) DBeaver/pgAdmin para verificar tablas

# Instalación
## 1) Crear y activar venv (Windows)
python -m venv venv
venv\Scripts\activate

##  En macOS/Linux:
- python3 -m venv venv
- source venv/bin/activate

## 2) Instalar dependencias
pip install -r requirements.txt

# Configuración

Crea un archivo .env en la raíz:

## 3) PostgreSQL local (ejemplo)
PG_URI=postgresql+psycopg2://usuario:password@localhost:5432/tu_db

## 4) Supabase si quieres doble escritura

SUPABASE_URI=postgresql+psycopg2://postgres:PASS@db.xxxxx.supabase.co:5432/postgres?sslmode=require

## Parámetros por defecto del clima
LATITUDE=4.711
LONGITUDE=-74.072
TIMEZONE=America/Bogota

### Importante: asegúrate de que PG_URI apunta a la misma base que abres en DBeaver/pgAdmin.

# Datos de entrada (CSV)

El pipeline espera, al menos, estas columnas en el CSV:

- date (YYYY-MM-DD)
- product_id (int)
- category (string)
- price (float)
- sales (int/float)


# Ejemplos rapidos:

-- ¿Qué tablas hay?

SELECT table_name
FROM information_schema.tables
WHERE table_schema='public'
ORDER BY 1;

-- Forecast diario (debe tener histórico + próximos 7 días)

SELECT MIN(date), MAX(date), COUNT(*) FROM public.daily_sales_forecast;

-- Producto–día insertado

SELECT COUNT(*) FROM public.sales_predictions;

-- Vista de errores por categoría
SELECT * FROM public.vw_error_by_category;

-- Vista de línea actual vs forecast

SELECT * FROM public.vw_sales_actual_vs_forecast ORDER BY date;

# 5) Ejecutar el codigo 

venv\Scripts\activate

python -m etl.pipeline --csv data/ventas_mockaroo.csv
