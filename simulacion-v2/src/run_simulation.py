import os
import json
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import scipy.stats as stats
import warnings
import pytz

warnings.simplefilter(action='ignore', category=FutureWarning)

# ----------------- NETWORK HELPER WITH RETRIES -----------------

def requests_get_with_retries(url, params=None, timeout=10, max_retries=7, backoff_factor=5):
    import time
    last_exception = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                wait_time = backoff_factor * (2 ** attempt)
                print(f"      [Intento {attempt+1}/{max_retries}] API de consulta devolvió error HTTP {r.status_code}. Reintentando en {wait_time}s...")
                time.sleep(wait_time)
                continue
            return r
        except requests.RequestException as e:
            last_exception = e
            wait_time = backoff_factor * (2 ** attempt)
            print(f"      [Intento {attempt+1}/{max_retries}] Error de red/conexión: {e}. Reintentando en {wait_time}s...")
            time.sleep(wait_time)
            
    # Si todos los intentos fallan, lanzamos la excepción para detener la ejecución
    raise last_exception

CITIES = {
    "Madrid": {
        "lat": 40.4936, 
        "lon": -3.5667, 
        "timezone": "Europe/Madrid",
        "station_id": "LEMD",
        "city_slug": "madrid"
    },
    "London": {
        "lat": 51.505, 
        "lon": 0.055, 
        "timezone": "Europe/London",
        "station_id": "EGLC",
        "city_slug": "london",
        "cache_file": "eglc_historical.csv"
    }
}

MIN_PROB = 0.25
MIN_EDGE = 0.00
MAX_OPTS = 2

# ----------------- DATA DOWNLOAD HELPERS -----------------

def calculate_rh(t, td):
    if t is None or td is None:
        return np.nan
    try:
        t = float(t)
        td = float(td)
        es = 6.11 * (10 ** ((7.5 * t) / (237.3 + t)))
        e = 6.11 * (10 ** ((7.5 * td) / (237.3 + td)))
        rh = (e / es) * 100
        return min(100.0, max(0.0, rh))
    except:
        return np.nan

def download_ground_truth_aviation(station_id, tz_name):
    print(f"   [API Aviación] Intentando descargar reportes METAR para {station_id}...")
    url = "https://aviationweather.gov/api/data/metar"
    params = {
        "ids": station_id,
        "format": "json",
        "hours": "48"
    }
    
    r = requests_get_with_retries(url, params=params, timeout=15)
    if r.status_code != 200:
        raise Exception(f"AWC API error: {r.status_code}")
        
    data = r.json()
    if not data:
        raise Exception(f"No METAR reports returned for {station_id}")
        
    records = []
    local_tz = pytz.timezone(tz_name)
    
    for item in data:
        temp = item.get("temp")
        dewp = item.get("dewp")
        wspd = item.get("wspd")
        altim = item.get("altim")
        obs_time_ts = item.get("obsTime")
        
        if obs_time_ts is None or temp is None:
            continue
            
        utc_dt = datetime.fromtimestamp(obs_time_ts, pytz.utc)
        local_dt = utc_dt.astimezone(local_tz)
        fecha = local_dt.date()
        
        rh = calculate_rh(temp, dewp)
        wspd_kmh = float(wspd) * 1.852 if wspd is not None else np.nan
        press = float(altim) if altim is not None else np.nan
        
        records.append({
            "Fecha": pd.to_datetime(fecha),
            "temp": float(temp),
            "rh": rh,
            "wspd_kmh": wspd_kmh,
            "press": press
        })
        
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        raise Exception("No valid rows parsed from METAR data")
        
    df_daily = df_raw.groupby("Fecha").agg(
        Temp_Max_Real=("temp", "max"),
        Humidity_Real=("rh", "mean"),
        Wind_Speed_Real=("wspd_kmh", "max"),
        Pressure_Real=("press", "mean")
    ).reset_index()
    
    return df_daily

def download_ground_truth_iem(station_id, past_days, tz, cache_file=None):
    df_cache = None
    start_download = datetime.now() - pd.Timedelta(days=past_days + 30)
    
    # 1. Intentar cargar datos desde caché CSV
    if cache_file:
        cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", cache_file)
        if os.path.exists(cache_path):
            try:
                df_cache = pd.read_csv(cache_path)
                df_cache["Fecha"] = pd.to_datetime(df_cache["Fecha"])
                latest_cache_date = df_cache["Fecha"].max()
                # Descargar solo desde el día siguiente al último día en caché
                start_download = latest_cache_date + pd.Timedelta(days=1)
                print(f"   [Caché] Cargados datos de {station_id} desde {cache_file} hasta {latest_cache_date.strftime('%Y-%m-%d')}.")
            except Exception as e:
                print(f"   [Aviso Caché] Error al leer caché {cache_file}: {e}")
                df_cache = None

    end = datetime.now() + timedelta(days=1)
    
    # 2. Descargar el bloque faltante (gap)
    df_downloaded = pd.DataFrame()
    
    if start_download < end:
        # Intentar primero Iowa ASOS
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        params = {
            "station": station_id,
            "data": "tmpc,relh,sknt,mslp,alti",
            "year1": str(start_download.year), "month1": str(start_download.month), "day1": str(start_download.day),
            "year2": str(end.year), "month2": str(end.month), "day2": str(end.day),
            "tz": tz,
            "format": "onlydata"
        }
        
        try:
            r = requests_get_with_retries(url, params=params, timeout=30)
            # Detect rate-limit or error pages
            if "Too many requests" in r.text or r.status_code != 200:
                print(f"   [Aviso Iowa] Bloqueo por Rate-limit o error HTTP {r.status_code}. Usando fallback de Aviación.")
                df_downloaded = download_ground_truth_aviation(station_id, tz)
            else:
                lines = [line.split(",") for line in r.text.strip().split("\n") if not line.startswith("#") and "," in line]
                if len(lines) > 1:
                    df_raw = pd.DataFrame(lines[1:], columns=lines[0])
                    df_raw["tmpc"] = pd.to_numeric(df_raw["tmpc"], errors="coerce")
                    df_raw["relh"] = pd.to_numeric(df_raw["relh"], errors="coerce")
                    df_raw["sknt"] = pd.to_numeric(df_raw["sknt"], errors="coerce")
                    df_raw["mslp"] = pd.to_numeric(df_raw["mslp"], errors="coerce")
                    df_raw["alti"] = pd.to_numeric(df_raw["alti"], errors="coerce")
                    df_raw["mslp"] = df_raw["mslp"].fillna(df_raw["alti"] * 33.8639)
                    df_raw["wspd_kmh"] = df_raw["sknt"] * 1.852
                    df_raw["valid"] = pd.to_datetime(df_raw["valid"])
                    df_raw["Fecha"] = df_raw["valid"].dt.date
                    df_downloaded = df_raw.groupby("Fecha").agg(
                        Temp_Max_Real=("tmpc", "max"),
                        Humidity_Real=("relh", "mean"),
                        Wind_Speed_Real=("wspd_kmh", "max"),
                        Pressure_Real=("mslp", "mean")
                    ).reset_index()
                    df_downloaded["Fecha"] = pd.to_datetime(df_downloaded["Fecha"])
                    print(f"   [API Iowa] Descargados con éxito {len(df_downloaded)} días recientes para {station_id}.")
                else:
                    print(f"   [Aviso Iowa] Respuesta vacía de Iowa ASOS. Usando fallback de Aviación.")
                    df_downloaded = download_ground_truth_aviation(station_id, tz)
        except Exception as e:
            print(f"   [Aviso Iowa] Error en conexión a Iowa: {e}. Usando fallback de Aviación.")
            try:
                df_downloaded = download_ground_truth_aviation(station_id, tz)
            except Exception as e2:
                print(f"   [⚠️ ERROR CRÍTICO] Ambos servidores fallaron. Error en fallback: {e2}")
                df_downloaded = pd.DataFrame()

    # Combinar caché y descarga reciente
    if not df_downloaded.empty:
        if df_cache is not None:
            df_daily = pd.concat([df_cache, df_downloaded]).drop_duplicates(subset=["Fecha"], keep="last")
        else:
            df_daily = df_downloaded
    else:
        df_daily = df_cache if df_cache is not None else pd.DataFrame()

    if df_daily is None or df_daily.empty:
        raise Exception(f"No hay datos disponibles para la estación {station_id}")
        
    df_daily.sort_values(by="Fecha", inplace=True)
    df_daily = df_daily.ffill().bfill()
    
    # Filtrar estrictamente para no incluir el día de hoy (que aún no está terminado)
    local_tz = pytz.timezone(tz)
    today_date = pd.to_datetime(datetime.now(local_tz).date())
    df_daily = df_daily[df_daily["Fecha"] < today_date]
    
    return df_daily

def download_forecasts(lat, lon, past_days, tz):
    # 1. Descargar pronosticos historicos (previous-runs-api)
    url_hist = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params_hist = {
        "latitude": lat, "longitude": lon, "past_days": past_days + 15,
        "hourly": ["temperature_2m", "relative_humidity_2m", "cloud_cover", "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
        "timezone": tz
    }
    try:
        r_hist = requests_get_with_retries(url_hist, params=params_hist, timeout=30).json()
        df_hourly_hist = pd.DataFrame({
            "Fecha_Hora": pd.to_datetime(r_hist["hourly"]["time"]),
            "Temp_Predicha": r_hist["hourly"]["temperature_2m"],
            "Humidity_Predicha": r_hist["hourly"]["relative_humidity_2m"],
            "Cloud_Predicha": r_hist["hourly"]["cloud_cover"],
            "Pressure_Predicha": r_hist["hourly"]["surface_pressure"],
            "Wind_Speed_Predicha": r_hist["hourly"]["wind_speed_10m"],
            "Wind_Dir_Predicha": r_hist["hourly"]["wind_direction_10m"]
        })
    except Exception as e:
        print(f"   [Aviso Forecast] Error al descargar pronosticos historicos: {e}")
        df_hourly_hist = pd.DataFrame()

    # 2. Descargar pronostico en vivo para hoy (api estandar, sin retraso de archivado)
    url_live = "https://api.open-meteo.com/v1/forecast"
    params_live = {
        "latitude": lat, "longitude": lon, "forecast_days": 3,
        "hourly": ["temperature_2m", "relative_humidity_2m", "cloud_cover", "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
        "timezone": tz
    }
    try:
        r_live = requests_get_with_retries(url_live, params=params_live, timeout=10).json()
        df_hourly_live = pd.DataFrame({
            "Fecha_Hora": pd.to_datetime(r_live["hourly"]["time"]),
            "Temp_Predicha": r_live["hourly"]["temperature_2m"],
            "Humidity_Predicha": r_live["hourly"]["relative_humidity_2m"],
            "Cloud_Predicha": r_live["hourly"]["cloud_cover"],
            "Pressure_Predicha": r_live["hourly"]["surface_pressure"],
            "Wind_Speed_Predicha": r_live["hourly"]["wind_speed_10m"],
            "Wind_Dir_Predicha": r_live["hourly"]["wind_direction_10m"]
        })
    except Exception as e:
        print(f"   [Aviso Forecast] Error al descargar pronosticos en vivo: {e}")
        df_hourly_live = pd.DataFrame()

    # 3. Combinar: historico + live, priorizando live para hoy
    if not df_hourly_hist.empty and not df_hourly_live.empty:
        df_hourly = pd.concat([df_hourly_hist, df_hourly_live]).drop_duplicates(subset=["Fecha_Hora"], keep="last")
    elif not df_hourly_hist.empty:
        df_hourly = df_hourly_hist
    else:
        df_hourly = df_hourly_live

    df_hourly["Fecha"] = df_hourly["Fecha_Hora"].dt.date
    df_daily = df_hourly.groupby("Fecha").agg(
        Temp_Max_Predicha=("Temp_Predicha", "max"),
        Temp_Min_Predicha=("Temp_Predicha", "min"),
        Temp_Mean_Predicha=("Temp_Predicha", "mean"),
        Humidity_Mean_Predicha=("Humidity_Predicha", "mean"),
        Cloud_Mean_Predicha=("Cloud_Predicha", "mean"),
        Pressure_Mean_Predicha=("Pressure_Predicha", "mean"),
        Wind_Max_Predicha=("Wind_Speed_Predicha", "max"),
        Wind_Dir_Mean_Predicha=("Wind_Dir_Predicha", "mean")
    ).reset_index()
    df_daily["Fecha"] = pd.to_datetime(df_daily["Fecha"])
    return df_daily

# ----------------- DATASET BUILDER -----------------

def build_dataset(city_name, config):
    cache_file = config.get("cache_file")
    df_real = download_ground_truth_iem(config["station_id"], 1100, config["timezone"], cache_file=cache_file)
    df_fcst = download_forecasts(config["lat"], config["lon"], 1100, config["timezone"])
    df = pd.merge(df_fcst, df_real, on="Fecha", how="left")
    df.sort_values(by="Fecha", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Temp_Max_Real_Ayer"] = df["Temp_Max_Real"].shift(1)
    df["Humidity_Real_Ayer"] = df["Humidity_Real"].shift(1)
    df["Wind_Speed_Real_Ayer"] = df["Wind_Speed_Real"].shift(1)
    df["Pressure_Real_Ayer"] = df["Pressure_Real"].shift(1)
    lag_cols = ["Temp_Max_Real_Ayer", "Humidity_Real_Ayer", "Wind_Speed_Real_Ayer", "Pressure_Real_Ayer"]
    df[lag_cols] = df[lag_cols].bfill()
    df["Error_Base"] = df["Temp_Max_Real"] - df["Temp_Max_Predicha"]
    df["Error_Ayer"] = df["Error_Base"].shift(1).bfill()
    df["Error_Delta"] = (df["Error_Base"].shift(1) - df["Error_Base"].shift(2)).bfill()
    df["Error_Mean_3d"] = df["Error_Base"].shift(1).rolling(window=3, min_periods=1).mean().bfill()
    df["Error_Mean_7d"] = df["Error_Base"].shift(1).rolling(window=7, min_periods=1).mean().bfill()
    
    # Trends for Candidate B
    df["Pressure_Trend_Predicha"] = df["Pressure_Mean_Predicha"] - df["Pressure_Real_Ayer"]
    df["Temp_Spread_Predicha"] = df["Temp_Max_Predicha"] - df["Temp_Min_Predicha"]
    df["Temp_Trend_Predicha"] = df["Temp_Max_Predicha"] - df["Temp_Max_Real_Ayer"]
    
    df["Mes"] = df["Fecha"].dt.month
    df["Dia_Ano"] = df["Fecha"].dt.dayofyear
    df["Wind_Dir_Sin"] = np.sin(np.radians(df["Wind_Dir_Mean_Predicha"]))
    df["Wind_Dir_Cos"] = np.cos(np.radians(df["Wind_Dir_Mean_Predicha"]))
    
    # Columnas de características necesarias para predecir hoy
    feature_cols = [
        "Temp_Max_Predicha", "Temp_Min_Predicha", "Temp_Mean_Predicha",
        "Humidity_Mean_Predicha", "Cloud_Mean_Predicha", "Pressure_Mean_Predicha",
        "Wind_Max_Predicha", "Wind_Dir_Sin", "Wind_Dir_Cos", "Mes", "Dia_Ano",
        "Temp_Max_Real_Ayer", "Humidity_Real_Ayer", "Wind_Speed_Real_Ayer", "Pressure_Real_Ayer",
        "Error_Ayer", "Error_Mean_3d", "Error_Mean_7d", "Error_Delta",
        "Pressure_Trend_Predicha", "Temp_Spread_Predicha", "Temp_Trend_Predicha"
    ]
    df = df.dropna(subset=feature_cols)
    
    # Mantener hoy (aunque no tenga Temp_Max_Real todavía), pero descartar días pasados que no tengan datos reales
    local_tz = pytz.timezone(config["timezone"])
    today_date = pd.to_datetime(datetime.now(local_tz).date())
    historical_mask = df["Fecha"] < today_date
    df = df[~historical_mask | df["Temp_Max_Real"].notna()]
    
    return df

# ----------------- TRAINING AND PREDICTION -----------------

def get_trained_models_for_city(df_all, target_date):
    # Static train end: June 12, 2026 (the start of our Polymarket test phase)
    # This prevents daily retraining noise and overfitting.
    train_end = pd.to_datetime("2026-06-12")
    
    # Base features (for V1 and Candidate A)
    base_features = [
        "Temp_Max_Predicha", "Temp_Min_Predicha", "Temp_Mean_Predicha",
        "Humidity_Mean_Predicha", "Cloud_Mean_Predicha", "Pressure_Mean_Predicha",
        "Wind_Max_Predicha", "Wind_Dir_Sin", "Wind_Dir_Cos", "Mes", "Dia_Ano",
        "Temp_Max_Real_Ayer", "Humidity_Real_Ayer", "Wind_Speed_Real_Ayer", "Pressure_Real_Ayer",
        "Error_Ayer", "Error_Mean_3d", "Error_Mean_7d"
    ]
    
    # Trend features (for Candidate B)
    trend_features = base_features + ["Error_Delta", "Pressure_Trend_Predicha", "Temp_Spread_Predicha", "Temp_Trend_Predicha"]
    
    # Validation set (9 days gap: target_date - 9 to target_date - 1)
    val_start = target_date - pd.Timedelta(days=9)
    val_end = target_date - pd.Timedelta(days=1)
    df_val = df_all[(df_all["Fecha"] >= val_start) & (df_all["Fecha"] <= val_end)].copy()
    
    val_mae_v1 = None
    val_acc_v1 = None
    val_mae_a = None
    val_acc_a = None
    val_mae_b = None
    val_acc_b = None
    
    # 1. Train Bot V1 (Base - 3.0 years, max_depth=4, no trends)
    df_v1 = df_all[df_all["Fecha"] < train_end].copy()
    model_v1 = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    model_v1.fit(df_v1[base_features], df_v1["Error_Base"])
    pred_v1_train = df_v1["Temp_Max_Predicha"] + model_v1.predict(df_v1[base_features])
    sigma_v1 = (df_v1["Temp_Max_Real"] - pred_v1_train).std()
    if not df_val.empty:
        pred_v1_val = df_val["Temp_Max_Predicha"] + model_v1.predict(df_val[base_features])
        val_mae_v1 = float((df_val["Temp_Max_Real"] - pred_v1_val).abs().mean())
        hits_v1 = int((pred_v1_val.round().astype(int) == df_val["Temp_Max_Real"].round().astype(int)).sum())
        val_acc_v1 = float(hits_v1 / len(df_val) * 100)
    
    # 2. Train Bot Candidate A (Max ROI - 2.5 years/912 days, max_depth=4, no trends)
    train_start_a = train_end - pd.Timedelta(days=912)
    df_a = df_all[(df_all["Fecha"] < train_end) & (df_all["Fecha"] >= train_start_a)].copy()
    model_a = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    model_a.fit(df_a[base_features], df_a["Error_Base"])
    pred_a_train = df_a["Temp_Max_Predicha"] + model_a.predict(df_a[base_features])
    sigma_a = (df_a["Temp_Max_Real"] - pred_a_train).std()
    if not df_val.empty:
        pred_a_val = df_val["Temp_Max_Predicha"] + model_a.predict(df_val[base_features])
        val_mae_a = float((df_val["Temp_Max_Real"] - pred_a_val).abs().mean())
        hits_a = int((pred_a_val.round().astype(int) == df_val["Temp_Max_Real"].round().astype(int)).sum())
        val_acc_a = float(hits_a / len(df_val) * 100)
    
    # 3. Train Bot Candidate B (Max Accuracy - 2.0 years/730 days, max_depth=3, with trends)
    train_start_b = train_end - pd.Timedelta(days=730)
    df_b = df_all[(df_all["Fecha"] < train_end) & (df_all["Fecha"] >= train_start_b)].copy()
    model_b = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    model_b.fit(df_b[trend_features], df_b["Error_Base"])
    pred_b_train = df_b["Temp_Max_Predicha"] + model_b.predict(df_b[trend_features])
    sigma_b = (df_b["Temp_Max_Real"] - pred_b_train).std()
    if not df_val.empty:
        pred_b_val = df_val["Temp_Max_Predicha"] + model_b.predict(df_val[trend_features])
        val_mae_b = float((df_val["Temp_Max_Real"] - pred_b_val).abs().mean())
        hits_b = int((pred_b_val.round().astype(int) == df_val["Temp_Max_Real"].round().astype(int)).sum())
        val_acc_b = float(hits_b / len(df_val) * 100)
    
    return {
        "bot_v1": {"model": model_v1, "features": base_features, "sigma": sigma_v1, "val_mae": val_mae_v1, "val_acc": val_acc_v1},
        "bot_cand_a": {"model": model_a, "features": base_features, "sigma": sigma_a, "val_mae": val_mae_a, "val_acc": val_acc_a},
        "bot_cand_b": {"model": model_b, "features": trend_features, "sigma": sigma_b, "val_mae": val_mae_b, "val_acc": val_acc_b}
    }

def get_probability_from_distribution(col_name, pred_val, sigma):
    if "or below" in col_name:
        try:
            val = float(col_name.split("°C")[0].strip())
            return stats.norm.cdf(val + 0.5, loc=pred_val, scale=sigma)
        except: return 0.0
    elif "or higher" in col_name:
        try:
            val = float(col_name.split("°C")[0].strip())
            return 1.0 - stats.norm.cdf(val - 0.5, loc=pred_val, scale=sigma)
        except: return 0.0
    else:
        try:
            val = float(col_name.replace("°C", "").strip())
            return stats.norm.cdf(val + 0.5, loc=pred_val, scale=sigma) - stats.norm.cdf(val - 0.5, loc=pred_val, scale=sigma)
        except: return 0.0

def verify_source_alignment(city_name, config, target_date, df_all, state, today_str):
    print(f"   [+] Ejecutando auditoría de alineación para la ventana de validación de 9 días ({city_name})...")
    try:
        mismatches = []
        for day_offset in range(1, 10):
            check_date = target_date - pd.Timedelta(days=day_offset)
            check_date_str = check_date.strftime("%Y-%m-%d")
            
            day_row = df_all[df_all["Fecha"] == check_date]
            if day_row.empty:
                # Alerta si falta el dato de la estación pero el mercado en Polymarket ya cerró y tiene ganador
                event = fetch_active_polymarket_event(config["city_slug"], check_date)
                if event:
                    resolved_winner = None
                    for m in event.get("markets", []):
                        outcome_prices = m.get("outcomePrices")
                        if outcome_prices:
                            try:
                                prices = json.loads(outcome_prices)
                                if len(prices) >= 1 and float(prices[0]) > 0.95:
                                    resolved_winner = m.get("groupItemTitle")
                                    break
                            except:
                                pass
                    if resolved_winner:
                        mismatches.append(f"{check_date_str} (Dato de estación FALTANTE pero Polymarket resuelto en '{resolved_winner}')")
                continue
            real_val = day_row["Temp_Max_Real"].values[0]
            real_rounded = int(round(real_val))
            
            event = fetch_active_polymarket_event(config["city_slug"], check_date)
            if not event:
                continue
                
            resolved_winner = None
            for m in event.get("markets", []):
                outcome_prices = m.get("outcomePrices")
                if outcome_prices:
                    try:
                        prices = json.loads(outcome_prices)
                        if len(prices) >= 1 and float(prices[0]) > 0.95:
                            resolved_winner = m.get("groupItemTitle")
                            break
                    except:
                        pass
                        
            if resolved_winner:
                clean_winner = resolved_winner.replace("°C", "").strip()
                is_aligned = False
                if "or below" in clean_winner:
                    val = int(float(clean_winner.split("or")[0].strip()))
                    is_aligned = (real_rounded <= val)
                elif "or higher" in clean_winner:
                    val = int(float(clean_winner.split("or")[0].strip()))
                    is_aligned = (real_rounded >= val)
                else:
                    try:
                        val = int(float(clean_winner))
                        is_aligned = (real_rounded == val)
                    except:
                        pass
                        
                if not is_aligned:
                    mismatches.append(f"{check_date_str} (ASOS {real_val}°C vs Polymarket '{resolved_winner}')")
                    
        if mismatches:
            mismatch_str = ", ".join(mismatches)
            print(f"   [⚠️ ALERTA AUDITORÍA] Desalineación detectada en ventana de validación para {city_name}: {mismatch_str}")
            if "audit_alerts" not in state:
                state["audit_alerts"] = {}
            state["audit_alerts"][city_name] = {
                "date": today_str,
                "message": f"Desalineación en validación de {city_name}: {mismatch_str}"
            }
        else:
            print(f"   [OK AUDITORIA] Todo alineado en la ventana de validacion de 9 dias para {city_name}.")
            if "audit_alerts" in state and city_name in state["audit_alerts"]:
                del state["audit_alerts"][city_name]
    except Exception as e:
        print(f"   [AUDITORÍA] Error al verificar alineación de fuentes: {e}")

# ----------------- POLYMARKET API CLIENT -----------------

def fetch_active_polymarket_event(city_slug, target_date):
    # e.g., slug style: highest-temperature-in-madrid-on-june-12-2026
    mes = target_date.strftime('%B').lower()
    dia = target_date.day
    ano = target_date.year
    exact_slug = f"highest-temperature-in-{city_slug}-on-{mes}-{dia}-{ano}"
    
    # Try exact slug first
    url = f"https://gamma-api.polymarket.com/events?slug={exact_slug}"
    try:
        r = requests_get_with_retries(url, timeout=10)
        if r.status_code == 200 and r.json():
            return r.json()[0]
    except:
        pass
        
    # Search fallback if slug naming differed
    search_url = "https://gamma-api.polymarket.com/events"
    params = {
        "closed": "false",
        "limit": 20,
        "query": f"{city_slug} temperature"
    }
    try:
        r = requests_get_with_retries(search_url, params=params, timeout=10)
        if r.status_code == 200:
            events = r.json()
            for e in events:
                # Match title words for date and ensure it corresponds to the correct city
                title = e.get("title", "").lower()
                if city_slug in title and mes in title and str(dia) in title:
                    return e
    except:
        pass
    return None

def update_history_curve(state):
    dates = set()
    for h in state.get("history", []):
        dates.add(h["date"])
    for b_group in state.get("resolved_bets", []):
        dates.add(b_group["date"])
    for b_group in state.get("active_bets", []):
        dates.add(b_group["date"])
    
    sorted_dates = sorted(list(dates))
    new_history = []
    
    for d_str in sorted_dates:
        entry = {"date": d_str}
        for bot_id, bot_data in state["bots"].items():
            initial = bot_data["initial_balance"]
            net_profit = 0.0
            
            for b_group in state.get("resolved_bets", []):
                if b_group["date"] <= d_str:
                    for b in b_group["bets"]:
                        if b["bot"] == bot_id:
                            res_str = b.get("result", "")
                            if res_str.startswith("+"):
                                try:
                                    net_profit += float(res_str.replace("+", "").replace("USD", "").strip())
                                except:
                                    pass
                            elif res_str.startswith("-"):
                                try:
                                    net_profit -= float(res_str.replace("-", "").replace("USD", "").strip())
                                except:
                                    pass
            
            entry[f"{bot_id}_balance"] = round(initial + net_profit, 2)
        new_history.append(entry)
        
    state["history"] = new_history

def recalculate_statistics(state):
    for bot_id, bot_data in state["bots"].items():
        decisions_count = 0
        decisions_won = 0
        
        for group in state.get("resolved_bets", []):
            bot_bets = [b for b in group.get("bets", []) if b["bot"] == bot_id]
            if bot_bets:
                decisions_count += 1
                if any(b.get("result", "").startswith("+") for b in bot_bets):
                    decisions_won += 1
                    
        bot_data["trades_count"] = decisions_count
        bot_data["wins"] = decisions_won
        bot_data["win_rate"] = round((decisions_won / decisions_count) * 100, 2) if decisions_count > 0 else 0.0
        
        initial = bot_data["initial_balance"]
        current = bot_data["balance"]
        bot_data["roi"] = round(((current - initial) / initial) * 100, 2)

# ----------------- MAIN PIPELINE RUNNER -----------------

def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    state_file = os.path.join(PROJECT_ROOT, "docs", "simulacion-v2", "data", "simulation_state.json")
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
        
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    print(f"======================= SIMULADOR CLOUD ({today_str}) =======================")

    # 1. RESOLVE YESTERDAY'S OPEN BETS
    active_bets = state.get("active_bets", [])
    resolved_bets = state.get("resolved_bets", [])
    new_active_bets = []
    
    for bet_group in active_bets:
        bet_date_str = bet_group.get("date")
        city = bet_group.get("city")
        print(f"\n[+] Verificando resolución de Polymarket para {city} ({bet_date_str})...")
        
        # Query all markets involved in this bet group
        resolved_bets_data = []
        all_resolved = True
        
        for b in bet_group.get("bets", []):
            m_id = b.get("market_id")
            m_url = f"https://gamma-api.polymarket.com/markets/{m_id}"
            try:
                mr = requests_get_with_retries(m_url, timeout=10)
                if mr.status_code == 200:
                    m_data = mr.json()
                    # Check if resolved officially
                    if m_data.get("umaResolutionStatus") == "resolved":
                        resolved_bets_data.append((b, m_data))
                    else:
                        all_resolved = False
                        break
                else:
                    all_resolved = False
                    break
            except Exception as e:
                print(f"   Error al consultar mercado {m_id}: {e}")
                all_resolved = False
                break
                
        if all_resolved and len(resolved_bets_data) > 0:
            print(f"   ¡Mercados resueltos oficialmente para {city} ({bet_date_str})!")
            
            # Determine the winning option (either one of our bets, or another option if we lost)
            resolved_winner = "Otra opción (Perdimos)"
            
            for b, m_data in resolved_bets_data:
                outcome_prices = m_data.get("outcomePrices")
                if outcome_prices:
                    prices = json.loads(outcome_prices)
                    if len(prices) >= 1 and float(prices[0]) > 0.95:
                        resolved_winner = b.get("option")
                        break
            
            # If we couldn't find a winner among our bets, we check the actual temperature from Polymarket event
            if resolved_winner == "Otra opción (Perdimos)":
                try:
                    event_data = fetch_active_polymarket_event(city.lower(), datetime.strptime(bet_date_str, "%Y-%m-%d"))
                    if event_data:
                        for m in event_data.get("markets", []):
                            outcome_prices = m.get("outcomePrices")
                            if outcome_prices:
                                prices = json.loads(outcome_prices)
                                if len(prices) >= 1 and float(prices[0]) > 0.95:
                                    resolved_winner = m.get("groupItemTitle")
                                    break
                except Exception as e:
                    print(f"   No se pudo obtener el nombre exacto de la opción ganadora: {e}")
            
            print(f"   Ganador oficial determinado: {resolved_winner}")
            
            # Distribute payoffs
            for b, m_data in resolved_bets_data:
                bot_id = b.get("bot")
                invested = b.get("invested")
                option_bought = b.get("option")
                buy_price = b.get("price")
                
                bot_ref = state["bots"][bot_id]
                
                # Check if this specific bet won (YES price > 0.95)
                outcome_prices = m_data.get("outcomePrices")
                won = False
                if outcome_prices:
                    prices = json.loads(outcome_prices)
                    if len(prices) >= 1 and float(prices[0]) > 0.95:
                        won = True
                
                if won:
                    payoff = invested / buy_price
                    net_profit = payoff - invested
                    bot_ref["balance"] += payoff
                    b["result"] = f"+{net_profit:.2f} USD"
                    print(f"   - {bot_ref['name']}: GANADOR de {option_bought} (Pago: +${payoff:.2f})")
                else:
                    b["result"] = f"-{invested:.2f} USD"
                    print(f"   - {bot_ref['name']}: PERDEDOR de {option_bought} (Pago: $0.00)")
                    
            # Add to resolved logs
            resolved_bets.append({
                "date": bet_date_str,
                "city": city,
                "question": bet_group.get("question"),
                "winner_option": resolved_winner,
                "bets": bet_group.get("bets")
            })
            
            # Recalculate statistics
            recalculate_statistics(state)
        else:
            print("   Mercado aún no resuelto en la API de Polymarket. Se reintentará en el próximo ciclo.")
            new_active_bets.append(bet_group)
            
    state["active_bets"] = new_active_bets
    state["resolved_bets"] = resolved_bets

    # 2. RUN TODAY'S PREDICTIONS AND SIMULATE NEW BETS
    print(f"\n[+] Buscando y evaluando nuevos mercados para hoy ({today_str})...")
    
    for city, config in CITIES.items():
        print(f"\n[{city.upper()}]:")
        
        # Check if we already have active bets for this city today to prevent double betting
        already_bet = False
        for bet_group in state.get("active_bets", []):
            if bet_group.get("date") == today_str and bet_group.get("city") == city:
                already_bet = True
                break
        if already_bet:
            print(f"   [!] Ya existen apuestas activas registradas para {city} hoy ({today_str}). Omitiendo simulación de compra para evitar duplicados.")
            continue
            
        # Find active event for today
        event = fetch_active_polymarket_event(config["city_slug"], today)
        if not event:
            print(f"   No se encontró mercado activo en Polymarket para {city} en la fecha {today_str}.")
            continue
            
        markets = event.get("markets", [])
        if not markets:
            print(f"   El evento de Polymarket para {city} no contiene sub-mercados activos.")
            continue
            
        print(f"   Mercado encontrado: {event.get('title')}")
        
        # Build dataset and train models
        try:
            df_all = build_dataset(city, config)
            target_date_dt = pd.to_datetime(today.date())
            models = get_trained_models_for_city(df_all, target_date_dt)
            verify_source_alignment(city, config, target_date_dt, df_all, state, today_str)
        except Exception as e:
            print(f"   Error al entrenar modelos / descargar datos para {city}: {e}")
            continue
            
        # Get today's prediction data
        today_row = df_all[df_all["Fecha"].dt.date == today.date()]
        if today_row.empty:
            # Fallback to the latest available day if today's forecast isn't fully updated yet
            today_row = df_all.iloc[-1:]
            
        # Place bets for each bot
        bets_placed_today = []
        
        for bot_id, bot_data in models.items():
            model = bot_data["model"]
            features = bot_data["features"]
            sigma = bot_data["sigma"]
            
            # Predict
            pred_bias = model.predict(today_row[features])[0]
            pred_ia = today_row["Temp_Max_Predicha"].values[0] + pred_bias
            
            bot_balance = state["bots"][bot_id]["balance"]
            # Skip if bot runs out of money (protection)
            if bot_balance <= 0:
                print(f"   - {state['bots'][bot_id]['name']}: Banca a cero ($0). No puede operar.")
                continue
                
            choices = []
            for m in markets:
                prices_str = m.get("outcomePrices")
                if not prices_str: continue
                prices = json.loads(prices_str)
                if len(prices) < 1: continue
                
                price = float(prices[0]) # YES Price
                if price <= 0.05: price = 0.05
                
                col_name = m.get("groupItemTitle")
                
                # Check range constraint: only bet on options within 1.5C of the prediction center
                # Helper function inside the loop context (or defined above in the file)
                def is_option_within_range(option_name, center_pred, range_limit=1.5):
                    lower_limit = center_pred - range_limit
                    upper_limit = center_pred + range_limit
                    if "or below" in option_name:
                        try:
                            val = float(option_name.split("°C")[0].strip())
                            return lower_limit <= val + 0.5
                        except: return True
                    elif "or higher" in option_name:
                        try:
                            val = float(option_name.split("°C")[0].strip())
                            return upper_limit >= val - 0.5
                        except: return True
                    else:
                        try:
                            val = float(option_name.replace("°C", "").strip())
                            return max(lower_limit, val - 0.5) <= min(upper_limit, val + 0.5)
                        except: return True
                        
                if not is_option_within_range(col_name, pred_ia, range_limit=1.5):
                    continue
                    
                prob = get_probability_from_distribution(col_name, pred_ia, sigma)
                edge = prob - price
                
                if prob >= MIN_PROB and edge >= MIN_EDGE:
                    choices.append({
                        "market_id": m.get("id"),
                        "option": col_name,
                        "price": price,
                        "edge": edge,
                        "prob": prob
                    })
                    
            # Select top 2 bets by Edge
            choices = sorted(choices, key=lambda x: x["edge"], reverse=True)[:MAX_OPTS]
            
            if choices:
                # Allocation: $10 total or remaining balance if below $10
                bet_amount = min(10.0, bot_balance)
                money_per_bet = bet_amount / len(choices)
                
                # Deduct balance
                state["bots"][bot_id]["balance"] -= bet_amount
                
                for c in choices:
                    # Aplicar penalización por deslizamiento (Slippage) basada en la inversión de esta apuesta
                    # 0.002 de recargo por dólar invertido (refleja el deslizamiento típico de Polymarket en estos mercados)
                    effective_price = min(0.99, c["price"] + 0.002 * money_per_bet)
                    effective_price = round(effective_price, 3)
                    
                    bets_placed_today.append({
                        "bot": bot_id,
                        "market_id": c["market_id"],
                        "option": c["option"],
                        "price": effective_price,
                        "invested": round(money_per_bet, 2),
                        "prob_ia": round(c["prob"] * 100, 1),
                        "pred_ia_temp": round(pred_ia, 2),
                        "pred_sat_temp": round(float(today_row["Temp_Max_Predicha"].values[0]), 2),
                        "ia_bias_correction": round(float(pred_bias), 2),
                        "sigma": round(float(sigma), 2),
                        "validation_mae_9d": round(float(bot_data["val_mae"]), 2) if bot_data["val_mae"] is not None else None,
                        "validation_acc_9d": round(float(bot_data["val_acc"]), 1) if bot_data["val_acc"] is not None else None,
                        "result": "Pendiente"
                    })
                    val_err_str = f"Val MAE 9d: {bot_data['val_mae']:.2f}°C, Val Acc: {bot_data['val_acc']:.1f}%" if bot_data["val_mae"] is not None else "N/A"
                    print(f"   - {state['bots'][bot_id]['name']}: Apostó ${money_per_bet:.2f} a '{c['option']}' (Precio Promedio: ${effective_price:.3f}, IA: {c['prob']*100:.1f}%, Satélite: {today_row['Temp_Max_Predicha'].values[0]:.2f}°C, Sesgo IA: {pred_bias:+.2f}°C, Pred_IA: {pred_ia:.2f}°C, {val_err_str})")
            else:
                print(f"   - {state['bots'][bot_id]['name']}: Sin operaciones (No se encontró edge o probabilidad mínima).")
                
        if bets_placed_today:
            state["active_bets"].append({
                "date": today_str,
                "city": city,
                "question": event.get("title"),
                "bets": bets_placed_today
            })

    # 3. RECALCULAR CURVA DE CAPITAL HISTÓRICA
    update_history_curve(state)
    recalculate_statistics(state)
    
    # Save state
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        
    print("\n[+] Simulación diaria completada con éxito y balances guardados.")
    print("=" * 77)

if __name__ == "__main__":
    main()
