import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge


# -------- Agregaciones --------
def aggregate_sales_daily(ventas: pd.DataFrame) -> pd.DataFrame:
    out = (
        ventas.groupby("date", as_index=False)
        .agg(total_sales=("sales","sum"),
             avg_price=("price","mean"),
             txns=("product_id","count"),
             unique_products=("product_id","nunique"))
        .sort_values("date")
    )
    return out

# -------- Clima full (hist + forecast + sim faltantes) --------
def _simulate_weather(dates: list) -> pd.DataFrame:
    n = len(dates)
    t = np.arange(n)
    temp_base = 18 + 4*np.sin(2*np.pi*t/30)
    temp_noise = np.random.normal(0, 1.2, n)
    prec = np.clip(np.random.gamma(1.5, 2.0, n) - 1.0, 0, None)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "temperature": temp_base + temp_noise,
        "precipitation": prec
    })

def build_climate_full(clima_hist: pd.DataFrame,
                       clima_fut: pd.DataFrame,
                       start_date,
                       end_date) -> pd.DataFrame:
    parts = []
    if clima_hist is not None and not clima_hist.empty:
        parts.append(clima_hist)
    if clima_fut is not None and not clima_fut.empty:
        parts.append(clima_fut)
    if parts:
        clima_api = pd.concat(parts, ignore_index=True).drop_duplicates("date")
    else:
        clima_api = pd.DataFrame(columns=["date","temperature","precipitation"])
    clima_api["date"] = pd.to_datetime(clima_api["date"]).dt.normalize()

    # completar días faltantes en rango ventas (o solicitado)
    full_range = pd.date_range(pd.to_datetime(start_date).normalize(),
                               pd.to_datetime(end_date).normalize(), freq="D")
    faltantes = sorted(set(full_range) - set(clima_api["date"]))
    if faltantes:
        clima_sim = _simulate_weather(faltantes)
        clima = (pd.concat([clima_api, clima_sim], ignore_index=True)
                 .sort_values("date")
                 .reset_index(drop=True))
    else:
        clima = clima_api.sort_values("date").reset_index(drop=True)
    return clima

# -------- Merge + imputaciones --------
def merge_sales_weather(ventas_daily: pd.DataFrame,
                        clima: pd.DataFrame) -> pd.DataFrame:
    df = ventas_daily.merge(clima, on="date", how="left")
    # imputaciones simples
    before_prec_na = df["precipitation"].isna().sum()
    df["precipitation"] = df["precipitation"].fillna(0)
    after_prec_na = df["precipitation"].isna().sum()

    before_temp_na = df["temperature"].isna().sum()
    df["temperature"] = df["temperature"].interpolate(limit_direction="both")
    df["temperature"] = df["temperature"].fillna(df["temperature"].median())
    after_temp_na = df["temperature"].isna().sum()

    print(f"Imputación precip: {before_prec_na}→{after_prec_na} | temp: {before_temp_na}→{after_temp_na}")
    return df

# -------- Modelo diario + predicción alineada a TODAS las fechas de clima --------
def _add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"]  = pd.to_datetime(d["date"]).dt.normalize()
    d["dow"]   = d["date"].dt.weekday        # 0..6
    d["is_we"] = (d["dow"] >= 5).astype(int) # fin de semana
    d["month"] = d["date"].dt.month
    return d

def train_daily_model_simple(df_daily: pd.DataFrame):
    """
    Entrena con histórico (donde existe 'sales') y devuelve:
    - model entrenado, lista feats, métricas, df_hist_pred (opcional)
    """
    df = df_daily.copy()
    df = _add_calendar_feats(df)

    feats = ["temperature","avg_price","dow","is_we","month"]
    dtrain = df.dropna(subset=feats + ["total_sales"])
    if dtrain.empty:
        raise ValueError("No hay filas con features y 'total_sales' para entrenar.")

    X = dtrain[feats].values
    y = dtrain["total_sales"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(Xtr, ytr)

    yhat = model.predict(Xte)
    metrics = {
        "mae": float(mean_absolute_error(yte, yhat)),
        "r2":  float(r2_score(yte, yhat)),
    }

    # in-sample (histórico) opcional
    df_hist = df_daily.copy()
    dfh = _add_calendar_feats(df_hist)
    mask = dfh[feats].notna().all(axis=1)
    df_hist["sales_prediction"] = np.nan
    df_hist.loc[mask, "sales_prediction"] = model.predict(dfh.loc[mask, feats].values)

    return model, feats, metrics, df_hist

def build_prediction_frame_from_climate(clima: pd.DataFrame,
                                        ventas_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Marco de predicción con TODAS las fechas del clima (hist+forecast):
    date, temperature, avg_price (proxy desde últimos 14 días).
    """
    base = clima[["date","temperature"]].drop_duplicates("date").copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    vd = ventas_daily.sort_values("date")
    price_proxy = float(vd["avg_price"].tail(14).mean()) if not vd.empty else float("nan")
    if np.isnan(price_proxy):
        price_proxy = float(vd["avg_price"].mean()) if "avg_price" in vd.columns and not vd.empty else 1.0
    base["avg_price"] = price_proxy
    return base

def predict_on_all_climate_dates(model, feats, pred_frame: pd.DataFrame) -> pd.DataFrame:
    dff = _add_calendar_feats(pred_frame)
    out = dff[["date","temperature","avg_price"]].copy()
    mask = dff[feats].notna().all(axis=1)
    out["sales_prediction"] = np.nan
    if mask.any():
        out.loc[mask, "sales_prediction"] = model.predict(dff.loc[mask, feats].values)
    return out[["date","sales_prediction","temperature","avg_price"]]

def aggregate_sales_product_daily(ventas: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa ventas por fecha + producto + categoría.
    """
    return (
        ventas.groupby(["date","product_id","category"], as_index=False)
        .agg(sales=("sales","sum"),
             price=("price","mean"))
        .sort_values("date")
    )

def prepare_product_frame_for_model(prod_daily: pd.DataFrame,
                                    clima: pd.DataFrame) -> pd.DataFrame:
    """
    Une clima con ventas por producto.
    """
    df = prod_daily.merge(clima[["date","temperature","precipitation"]], on="date", how="left")
    return df

def train_and_predict_product_daily(df: pd.DataFrame):
    """
    Entrena modelo simple Ridge para producto-día.
    Devuelve df con sales_prediction y métricas globales.
    """
    df = df.copy()
    df["dow"]   = pd.to_datetime(df["date"]).dt.weekday
    df["is_we"] = (df["dow"] >= 5).astype(int)
    df["month"] = pd.to_datetime(df["date"]).dt.month

    feats = ["price","temperature","precipitation","dow","is_we","month"]
    dtrain = df.dropna(subset=feats + ["sales"])
    if dtrain.empty:
        return df, {"mae": None, "r2": None}

    X = dtrain[feats].values
    y = dtrain["sales"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(Xtr, ytr)

    yhat = model.predict(Xte)
    metrics = {
        "mae": float(mean_absolute_error(yte, yhat)),
        "r2":  float(r2_score(yte, yhat)),
    }

    df["sales_prediction"] = model.predict(df[feats].fillna(0).values)
    return df, metrics
