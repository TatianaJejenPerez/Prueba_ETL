import requests
import pandas as pd
import numpy as np

# -------- CSV --------
def read_sales_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    # normalizar nombres básicos si vienen en mayúsculas
    cols = {c.lower(): c for c in df.columns}
    # asegurar campos
    for need in ["date", "product_id", "category", "price", "sales"]:
        if need not in cols and need not in df.columns:
            raise ValueError(f"Falta columna obligatoria en CSV: {need}")
    # parseo de fecha
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # tipos
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    # limpieza mínima
    df = df.dropna(subset=["date", "product_id", "category"])
    return df

# -------- Open-Meteo --------
def _to_df_daily(d: dict) -> pd.DataFrame:
    daily = d.get("daily", {}) or {}
    times = daily.get("time") or []
    tmax  = daily.get("temperature_2m_max") or []
    tmin  = daily.get("temperature_2m_min") or []
    prec  = daily.get("precipitation_sum") or []
    df = pd.DataFrame({"date": pd.to_datetime(times)})
    if len(tmax) == len(times) and len(tmin) == len(times):
        df["temperature"] = (pd.to_numeric(tmax) + pd.to_numeric(tmin)) / 2.0
    elif len(tmax) == len(times):
        df["temperature"] = pd.to_numeric(tmax)
    elif len(tmin) == len(times):
        df["temperature"] = pd.to_numeric(tmin)
    else:
        df["temperature"] = np.nan
    df["precipitation"] = pd.to_numeric(prec, errors="coerce") if len(prec)==len(times) else np.nan
    df["date"] = df["date"].dt.normalize()
    return df[["date","temperature","precipitation"]]

def fetch_weather_archive(lat, lon, tz, start, end, timeout=40) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz,
        "start_date": pd.to_datetime(start).date(),
        "end_date":   pd.to_datetime(end).date(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return _to_df_daily(r.json())

def fetch_weather_forecast(lat, lon, tz, today, end, timeout=40) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    forecast_days = int((pd.to_datetime(end) - pd.to_datetime(today)).days) + 1
    forecast_days = max(1, min(7, forecast_days))  # API permite 1..7
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": forecast_days
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    df = _to_df_daily(r.json())
   
    df = df[(df["date"] >= pd.to_datetime(today).normalize()) &
            (df["date"] <= pd.to_datetime(end).normalize())]
    return df.reset_index(drop=True)
