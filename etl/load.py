import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Cargar .env al importar este módulo
load_dotenv()

# ----------------- Conexión y escritura genérica -----------------
def get_engine(uri_env: str):
    """
    Devuelve un SQLAlchemy engine a partir de una variable de entorno
    (PG_URI, SUPABASE_URI, etc.). Imprime la URI para transparencia.
    """
    uri = os.getenv(uri_env)
    if not uri:
        print(f" {uri_env} no está definido. No se puede conectar.")
        return None
    print(f" Usando {uri_env}: {uri}")
    return create_engine(uri)

def save_table(df: pd.DataFrame, table_name: str, engine, if_exists="replace", schema="public"):
    """
    Escribe un DataFrame en una tabla.
    """
    if df is None or df.empty:
        print(f" {table_name}: DataFrame vacío, no se escribe.")
        return
    if engine is None:
        raise RuntimeError(f"No hay engine para {table_name}")
    print(f" Escribiendo {table_name} ({len(df)} filas) en {schema}.{table_name} …")
    df.to_sql(table_name, engine, if_exists=if_exists, index=False, schema=schema)

def dual_write(df: pd.DataFrame, table: str, if_exists="replace", schema="public"):
    """
    Escribe en Postgres local (PG_URI).
    Si quieres habilitar Supabase, descomenta el bloque indicado.
    """
    pg = get_engine("PG_URI")
    save_table(df, table, pg, if_exists=if_exists, schema=schema)

    # ---- Si quieres doble escritura a Supabase, descomenta estas líneas: ----
    # supa = get_engine("SUPABASE_URI")
    # if supa is not None:
    #     save_table(df, table, supa, if_exists=if_exists, schema=schema)

# ----------------- Vistas útiles para Power BI -----------------
def ensure_views_and_summaries():
    """
    Crea/actualiza vistas para el tablero:
      - vw_error_by_category: MAE y MAPE por categoría (sobre sales_predictions)
      - vw_sales_actual_vs_forecast: serie real vs pronóstico (para línea continua)
    """
    pg = get_engine("PG_URI")
    if pg is None:
        print("  No se pudieron crear vistas (sin PG_URI).")
        return
    with pg.begin() as c:
        c.execute(text("""
        CREATE OR REPLACE VIEW public.vw_error_by_category AS
        SELECT
          category,
          AVG(ABS(sales - sales_prediction))::NUMERIC                    AS mae,
          AVG(ABS(sales - sales_prediction) / NULLIF(sales,0))::NUMERIC AS mape
        FROM public.sales_predictions
        GROUP BY category;
        """))
        c.execute(text("""
        CREATE OR REPLACE VIEW public.vw_sales_actual_vs_forecast AS
        SELECT 
          d.date,
          d.total_sales::FLOAT AS sales,
          NULL::FLOAT          AS sales_prediction,
          'actual'::text       AS source
        FROM public.daily_sales_agg d
        UNION ALL
        SELECT
          f.date,
          NULL::FLOAT          AS sales,
          f.sales_prediction::FLOAT,
          'pred'::text         AS source
        FROM public.daily_sales_forecast f;
        """))
    print(" Vistas creadas/actualizadas: vw_error_by_category, vw_sales_actual_vs_forecast")

# ----------------- Tabla sales_predictions (producto-día) -----------------
def create_sales_predictions_table(engine):
    """
    Crea la tabla public.sales_predictions si no existe.
    """
    if engine is None:
        print(" No hay engine para crear sales_predictions")
        return
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.sales_predictions (
            id SERIAL PRIMARY KEY,
            date DATE,
            product_id INT,
            category VARCHAR(50),
            sales FLOAT,
            price FLOAT,
            temperature FLOAT,
            sales_prediction FLOAT
        );
        """))
    print(" Tabla sales_predictions lista")

def insert_sales_predictions(df: pd.DataFrame, engine):
    """
    Inserta filas en public.sales_predictions (append).
    Espera columnas: date, product_id, category, sales, price, temperature, sales_prediction
    """
    if df is None or df.empty:
        print(" sales_predictions vacío, no se inserta")
        return
    cols = ["date","product_id","category","sales","price","temperature","sales_prediction"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para insertar sales_predictions: {missing}")
    df_to_insert = df[cols].copy()
    df_to_insert.to_sql("sales_predictions", engine,
                        if_exists="append", index=False, schema="public")
    print(f" Insertados {len(df_to_insert)} registros en sales_predictions")
