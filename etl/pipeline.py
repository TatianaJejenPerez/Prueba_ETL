import argparse
import os
import pandas as pd
from dotenv import load_dotenv

# -------- Extract --------
from .extract import (
    read_sales_csv,
    fetch_weather_archive,
    fetch_weather_forecast
)

# -------- Transform (agregado diario + forecast alineado a clima) --------
from .transform import (
    aggregate_sales_daily,
    build_climate_full,
    merge_sales_weather,
    train_daily_model_simple,
    build_prediction_frame_from_climate,
    predict_on_all_climate_dates,

    # producto-día
    aggregate_sales_product_daily,
    prepare_product_frame_for_model,
    train_and_predict_product_daily
)

# -------- Load --------
from .load import (
    dual_write,
    ensure_views_and_summaries,
    get_engine,
    create_sales_predictions_table,
    insert_sales_predictions
)

def parse_args():
    p = argparse.ArgumentParser(description="ETL ventas + clima + predicción")
    p.add_argument("--csv", required=True, help="Ruta al CSV de ventas")
    p.add_argument("--lat", type=float, default=float(os.getenv("LATITUDE", 4.711)))
    p.add_argument("--lon", type=float, default=float(os.getenv("LONGITUDE", -74.072)))
    p.add_argument("--tz", default=os.getenv("TIMEZONE", "America/Bogota"))
    p.add_argument("--forecast_days", type=int, default=7, help="1..7 días (API forecast)")
    return p.parse_args()

def main():
    # 1) Config
    load_dotenv()  # asegurar lectura de .env
    args = parse_args()
    print("PG_URI:", os.getenv("PG_URI"))

    # 2) Extract
    print(" Leyendo ventas:", args.csv)
    ventas = read_sales_csv(args.csv)

    today = pd.Timestamp.now(tz=args.tz).normalize().tz_localize(None)

    # rango total deseado (ventas + forecast)
    start = ventas["date"].min().normalize()
    end   = max(ventas["date"].max().normalize(),
                today + pd.Timedelta(days=args.forecast_days - 1))

    #  cortar bien según API
    hist_end = min(end, today - pd.Timedelta(days=1))                # solo pasado
    fore_end = min(end, today + pd.Timedelta(days=args.forecast_days - 1))  # hoy..+7

    print(f" Archive: {start.date()} → {hist_end.date()} | Forecast: {today.date()} → {fore_end.date()}")

    clima_hist = pd.DataFrame(columns=["date","temperature","precipitation"])
    if start <= hist_end:
        clima_hist = fetch_weather_archive(args.lat, args.lon, args.tz, start, hist_end)

    clima_fut  = fetch_weather_forecast(args.lat, args.lon, args.tz, today, fore_end)


    # 3) Transform (agregado diario)
    print(" Agregando ventas diarias…")
    ventas_daily = aggregate_sales_daily(ventas)

    print(" Unificando clima (hist+forecast) y simulando faltantes…")
    clima = build_climate_full(clima_hist, clima_fut,
                               ventas_daily["date"].min(),
                               end)

    print(" Merge ventas+clima e imputaciones…")
    df = merge_sales_weather(ventas_daily, clima)

    # 4) Modelo diario y predicción alineada a TODAS las fechas del clima
    print(" Entrenando modelo diario (histórico)…")
    model, feats, metrics, df_hist_pred = train_daily_model_simple(df)
    print(" Métricas → MAE: {:.2f} | R²: {:.3f}".format(metrics["mae"], metrics["r2"]))

    print(" Construyendo marco de predicción con TODAS las fechas de clima…")
    pred_frame = build_prediction_frame_from_climate(clima, ventas_daily)

    print(" Prediciendo ventas (hist + próximos 7 días que tenga el clima)…")
    df_allpred = predict_on_all_climate_dates(model, feats, pred_frame)

    # 5) Tablas de salida (agregado diario)
    daily_weather        = clima[["date","temperature","precipitation"]].drop_duplicates("date")
    daily_sales_agg      = ventas_daily.copy()
    daily_sales_forecast = df_allpred[["date","sales_prediction"]].copy()  # contiene hist + futuro

    # 6) LOAD (agregado diario)
    print(" Guardando tablas en Postgres (schema public)…")
    dual_write(ventas,               "raw_sales",            if_exists="replace")
    dual_write(daily_weather,        "daily_weather",        if_exists="replace")
    dual_write(daily_sales_agg,      "daily_sales_agg",      if_exists="replace")
    dual_write(daily_sales_forecast, "daily_sales_forecast", if_exists="replace")

    # 7) Producto-día (tabla sales_predictions)
    print(" Preparando dataset producto-día…")
    ventas_prod_daily = aggregate_sales_product_daily(ventas)
    prod_frame = prepare_product_frame_for_model(ventas_prod_daily, clima)

    print(" Entrenando modelo producto-día y generando sales_prediction…")
    prod_pred, prod_metrics = train_and_predict_product_daily(prod_frame)
    print(" Producto-día → MAE: {:.2f} | R²: {:.3f}".format(
        prod_metrics['mae'] if prod_metrics['mae'] is not None else float("nan"),
        prod_metrics['r2']  if prod_metrics['r2']  is not None else float("nan"))
    )

    pg_engine = get_engine("PG_URI")
    create_sales_predictions_table(pg_engine)
    insert_sales_predictions(prod_pred, pg_engine)

    # 8) Vistas para Power BI
    ensure_views_and_summaries()

    # 9) Fin
    print(" Pipeline completado.")
    print("   Filas → raw_sales={}, daily_weather={}, daily_sales_agg={}, daily_sales_forecast={}".format(
        len(ventas), len(daily_weather), len(daily_sales_agg), len(daily_sales_forecast)
    ))

if __name__ == "__main__":
    main()
