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
MIN_EDGE = 0.08  # Filtro óptimo de margen
MAX_OPTS = 2

# ----------------- DATA DOWNLOAD HELPERS -----------------
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
        
        wspd_kmh = float(wspd) * 1.852 if wspd is not None else np.nan
        press = float(altim) if altim is not None else np.nan
        
        records.append({
            "Fecha": pd.to_datetime(fecha),
            "temp": float(temp),
            "rh": 50.0, # valor aproximado de fallback
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
    
    if cache_file:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_path = os.path.join(project_root, "data", cache_file)
        if os.path.exists(cache_path):
            try:
                df_cache = pd.read_csv(cache_path)
                df_cache["Fecha"] = pd.to_datetime(df_cache["Fecha"])
                latest_cache_date = df_cache["Fecha"].max()
                start_download = latest_cache_date + pd.Timedelta(days=1)
            except Exception as e:
                df_cache = None

    end = datetime.now() + timedelta(days=1)
    df_downloaded = pd.DataFrame()
    
    if start_download < end:
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
            if "Too many requests" in r.text or r.status_code != 200:
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
                else:
                    df_downloaded = download_ground_truth_aviation(station_id, tz)
        except Exception as e:
            try:
                df_downloaded = download_ground_truth_aviation(station_id, tz)
            except Exception as e2:
                df_downloaded = pd.DataFrame()

    if not df_downloaded.empty:
        if df_cache is not None:
            df_daily = pd.concat([df_cache, df_downloaded]).drop_duplicates(subset=["Fecha"], keep="last")
        else:
            df_daily = df_downloaded
    else:
        df_daily = df_cache if df_cache is not None else pd.DataFrame()

    if df_daily is None or df_daily.empty:
        raise Exception(f"No hay observaciones disponibles para {station_id}")
        
    df_daily.sort_values(by="Fecha", inplace=True)
    df_daily = df_daily.ffill().bfill()
    
    local_tz = pytz.timezone(tz)
    today_date = pd.to_datetime(datetime.now(local_tz).date())
    df_daily = df_daily[df_daily["Fecha"] < today_date]
    return df_daily

def download_historical_archive(lat, lon, past_days, tz):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": (datetime.now() - timedelta(days=past_days + 45)).strftime("%Y-%m-%d"),
        "end_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "surface_pressure", "shortwave_radiation",
            "temperature_850hPa", "geopotential_height_500hPa", "wind_speed_10m"
        ],
        "timezone": tz
    }
    
    r = requests_get_with_retries(url, params=params, timeout=30)
    data = r.json()
    
    df_hourly = pd.DataFrame({
        "Fecha_Hora": pd.to_datetime(data["hourly"]["time"]),
        "Temp_2m": data["hourly"]["temperature_2m"],
        "RH_2m": data["hourly"]["relative_humidity_2m"],
        "Pressure_Sfc": data["hourly"]["surface_pressure"],
        "Radiation": data["hourly"]["shortwave_radiation"],
        "Temp_850hPa": data["hourly"]["temperature_850hPa"],
        "Geopotential_500hPa": data["hourly"]["geopotential_height_500hPa"],
        "Wind_Speed_10m": data["hourly"]["wind_speed_10m"]
    })
    
    df_hourly["Fecha"] = df_hourly["Fecha_Hora"].dt.date
    df_daily = df_hourly.groupby("Fecha").agg(
        Temp_Max_Global_Sfc=("Temp_2m", "max"),
        Temp_Mean_850hPa=("Temp_850hPa", "mean"),
        Temp_Max_850hPa=("Temp_850hPa", "max"),
        Geopotential_Mean_500hPa=("Geopotential_500hPa", "mean"),
        Radiation_Sum=("Radiation", "sum"),
        Pressure_Mean_Sfc=("Pressure_Sfc", "mean"),
        RH_Mean_Sfc=("RH_2m", "mean"),
        Wind_Max_Sfc=("Wind_Speed_10m", "max")
    ).reset_index()
    
    df_daily["Fecha"] = pd.to_datetime(df_daily["Fecha"])
    return df_daily

def download_forecasts(lat, lon, tz, target_date):
    run_str = target_date.strftime("%Y-%m-%d") + "T00:00"
    url_single = "https://single-runs-api.open-meteo.com/v1/forecast"
    params_single = {
        "latitude": lat, "longitude": lon, "run": run_str,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "surface_pressure", "shortwave_radiation",
            "temperature_850hPa", "geopotential_height_500hPa", "wind_speed_10m"
        ],
        "timezone": tz
    }
    try:
        r_single = requests_get_with_retries(url_single, params=params_single, timeout=15).json()
        df_hourly = pd.DataFrame({
            "Fecha_Hora": pd.to_datetime(r_single["hourly"]["time"]),
            "Temp_2m": r_single["hourly"]["temperature_2m"],
            "RH_2m": r_single["hourly"]["relative_humidity_2m"],
            "Pressure_Sfc": r_single["hourly"]["surface_pressure"],
            "Radiation": r_single["hourly"]["shortwave_radiation"],
            "Temp_850hPa": r_single["hourly"]["temperature_850hPa"],
            "Geopotential_500hPa": r_single["hourly"]["geopotential_height_500hPa"],
            "Wind_Speed_10m": r_single["hourly"]["wind_speed_10m"]
        })
        df_hourly["Fecha"] = df_hourly["Fecha_Hora"].dt.date
        df_hourly = df_hourly[df_hourly["Fecha"] == target_date.date()]
        df_daily = df_hourly.groupby("Fecha").agg(
            Temp_Max_Global_Sfc_FCST=("Temp_2m", "max"),
            Temp_Mean_850hPa_FCST=("Temp_850hPa", "mean"),
            Temp_Max_850hPa_FCST=("Temp_850hPa", "max"),
            Geopotential_Mean_500hPa_FCST=("Geopotential_500hPa", "mean"),
            Radiation_Sum_FCST=("Radiation", "sum"),
            Pressure_Mean_Sfc_FCST=("Pressure_Sfc", "mean"),
            RH_Mean_Sfc_FCST=("RH_2m", "mean"),
            Wind_Max_Sfc_FCST=("Wind_Speed_10m", "max")
        ).reset_index()
        df_daily["Fecha"] = pd.to_datetime(df_daily["Fecha"])
        return df_daily
    except Exception as e:
        print(f"   [Error Forecast 00z] Corrida {run_str} falló: {e}")
        return pd.DataFrame()

def build_dataset(city_name, config, target_date):
    cache_file = config.get("cache_file")
    df_real = download_ground_truth_iem(config["station_id"], 1200, config["timezone"], cache_file=cache_file)
    df_train_archive = download_historical_archive(config["lat"], config["lon"], 1200, config["timezone"])
    
    df = pd.merge(df_real, df_train_archive, on="Fecha", how="inner")
    
    # Cast float
    for col in df.columns:
        if col != "Fecha":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df["Temp_Max_Real_Ayer"] = df["Temp_Max_Real"].shift(1)
    df["Humidity_Real_Ayer"] = df["Humidity_Real"].shift(1)
    df["Wind_Speed_Real_Ayer"] = df["Wind_Speed_Real"].shift(1)
    df["Pressure_Real_Ayer"] = df["Pressure_Real"].shift(1)
    df["Temp_Max_850hPa_Ayer"] = df["Temp_Max_850hPa"].shift(1)
    
    df["Mes"] = df["Fecha"].dt.month
    df["Dia_Ano"] = df["Fecha"].dt.dayofyear
    
    # Rellenar nulos de variables autoregresivas
    lag_cols = ["Temp_Max_Real_Ayer", "Humidity_Real_Ayer", "Wind_Speed_Real_Ayer", "Pressure_Real_Ayer", "Temp_Max_850hPa_Ayer"]
    df[lag_cols] = df[lag_cols].bfill()
    df.dropna(inplace=True)
    return df

# ----------------- TRAINING AND PREDICTION -----------------
def get_trained_models_for_city(df_all, target_date):
    train_end = target_date - pd.Timedelta(days=10)
    
    features = [
        "Temp_Mean_850hPa", "Temp_Max_850hPa", "Geopotential_Mean_500hPa",
        "Radiation_Sum", "Pressure_Mean_Sfc", "RH_Mean_Sfc", "Wind_Max_Sfc",
        "Temp_Max_Real_Ayer", "Humidity_Real_Ayer", "Wind_Speed_Real_Ayer", "Pressure_Real_Ayer",
        "Temp_Max_850hPa_Ayer", "Mes", "Dia_Ano"
    ]
    
    val_start = target_date - pd.Timedelta(days=9)
    val_end = target_date - pd.Timedelta(days=1)
    df_val = df_all[(df_all["Fecha"] >= val_start) & (df_all["Fecha"] <= val_end)].copy()
    
    # Model configuration for Downscaling
    df_v1 = df_all[df_all["Fecha"] < train_end].copy()
    model_v1 = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model_v1.fit(df_v1[features], df_v1["Temp_Max_Real"])
    pred_v1_train = model_v1.predict(df_v1[features])
    sigma_v1 = (df_v1["Temp_Max_Real"] - pred_v1_train).std()
    
    val_mae_v1, val_acc_v1 = None, None
    if not df_val.empty:
        pred_v1_val = model_v1.predict(df_val[features])
        val_mae_v1 = float((df_val["Temp_Max_Real"] - pred_v1_val).abs().mean())
        hits_v1 = int((pred_v1_val.round().astype(int) == df_val["Temp_Max_Real"].round().astype(int)).sum())
        val_acc_v1 = float(hits_v1 / len(df_val) * 100)

    # Bot A (ROI Máximo)
    train_start_a = train_end - pd.Timedelta(days=912)
    df_a = df_all[(df_all["Fecha"] < train_end) & (df_all["Fecha"] >= train_start_a)].copy()
    model_a = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model_a.fit(df_a[features], df_a["Temp_Max_Real"])
    pred_a_train = model_a.predict(df_a[features])
    sigma_a = (df_a["Temp_Max_Real"] - pred_a_train).std()
    
    val_mae_a, val_acc_a = None, None
    if not df_val.empty:
        pred_a_val = model_a.predict(df_val[features])
        val_mae_a = float((df_val["Temp_Max_Real"] - pred_a_val).abs().mean())
        hits_a = int((pred_a_val.round().astype(int) == df_val["Temp_Max_Real"].round().astype(int)).sum())
        val_acc_a = float(hits_a / len(df_val) * 100)

    # Bot B (Acc Máximo)
    train_start_b = train_end - pd.Timedelta(days=730)
    df_b = df_all[(df_all["Fecha"] < train_end) & (df_all["Fecha"] >= train_start_b)].copy()
    model_b = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model_b.fit(df_b[features], df_b["Temp_Max_Real"])
    pred_b_train = model_b.predict(df_b[features])
    sigma_b = (df_b["Temp_Max_Real"] - pred_b_train).std()
    
    val_mae_b, val_acc_b = None, None
    if not df_val.empty:
        pred_b_val = model_b.predict(df_val[features])
        val_mae_b = float((df_val["Temp_Max_Real"] - pred_b_val).abs().mean())
        hits_b = int((pred_b_val.round().astype(int) == df_val["Temp_Max_Real"].round().astype(int)).sum())
        val_acc_b = float(hits_b / len(df_val) * 100)

    return {
        "bot_v1": {"model": model_v1, "features": features, "sigma": sigma_v1, "val_mae": val_mae_v1, "val_acc": val_acc_v1},
        "bot_cand_a": {"model": model_a, "features": features, "sigma": sigma_a, "val_mae": val_mae_a, "val_acc": val_acc_a},
        "bot_cand_b": {"model": model_b, "features": features, "sigma": sigma_b, "val_mae": val_mae_b, "val_acc": val_acc_b}
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

def fetch_active_polymarket_event(city_slug, target_date):
    mes = target_date.strftime('%B').lower()
    dia = target_date.day
    ano = target_date.year
    exact_slug = f"highest-temperature-in-{city_slug}-on-{mes}-{dia}-{ano}"
    
    url = f"https://gamma-api.polymarket.com/events?slug={exact_slug}"
    try:
        r = requests_get_with_retries(url, timeout=10)
        if r.status_code == 200 and r.json():
            return r.json()[0]
    except:
        pass
        
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
    state_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "simulacion-v4", "data", "simulation_state.json")
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
        
    local_tz = pytz.timezone("Europe/Madrid")
    today = datetime.now(local_tz)
    today_str = today.strftime("%Y-%m-%d")
    print(f"======================= SIMULADOR CLOUD V4 ({today_str}) =======================")

    # 1. RESOLVE YESTERDAY'S OPEN BETS
    active_bets = state.get("active_bets", [])
    resolved_bets = state.get("resolved_bets", [])
    new_active_bets = []
    
    for bet_group in active_bets:
        bet_date_str = bet_group.get("date")
        city = bet_group.get("city")
        print(f"\n[+] Verificando resolución de Polymarket para {city} ({bet_date_str})...")
        
        resolved_bets_data = []
        all_resolved = True
        
        for b in bet_group.get("bets", []):
            m_id = b.get("market_id")
            m_url = f"https://gamma-api.polymarket.com/markets/{m_id}"
            try:
                mr = requests_get_with_retries(m_url, timeout=10)
                if mr.status_code == 200:
                    m_data = mr.json()
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
            
            resolved_winner = "Otra opción (Perdimos)"
            for b, m_data in resolved_bets_data:
                outcome_prices = m_data.get("outcomePrices")
                if outcome_prices:
                    prices = json.loads(outcome_prices)
                    if len(prices) >= 1 and float(prices[0]) > 0.95:
                        resolved_winner = b.get("option")
                        break
            
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
                    pass
            
            print(f"   Ganador oficial determinado: {resolved_winner}")
            
            for b, m_data in resolved_bets_data:
                bot_id = b.get("bot")
                invested = b.get("invested")
                option_bought = b.get("option")
                buy_price = b.get("price")
                
                bot_ref = state["bots"][bot_id]
                
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
                    
            resolved_bets.append({
                "date": bet_date_str,
                "city": city,
                "question": bet_group.get("question"),
                "winner_option": resolved_winner,
                "bets": bet_group.get("bets")
            })
            
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
        
        already_bet = False
        for bet_group in state.get("active_bets", []):
            if bet_group.get("date") == today_str and bet_group.get("city") == city:
                already_bet = True
                break
        if already_bet:
            print(f"   [!] Ya existen apuestas activas registradas para {city} hoy ({today_str}). Omitiendo simulación de compra.")
            continue
            
        event = fetch_active_polymarket_event(config["city_slug"], today)
        if not event:
            print(f"   No se encontró mercado activo en Polymarket para {city} en la fecha {today_str}.")
            continue
            
        markets = event.get("markets", [])
        if not markets:
            print(f"   El evento de Polymarket para {city} no contiene sub-mercados activos.")
            continue
            
        print(f"   Mercado encontrado: {event.get('title')}")
        
        try:
            df_all = build_dataset(city, config, today)
            
            # Obtener pronóstico de Single Run 00z de hoy
            df_fcst_today = download_forecasts(config["lat"], config["lon"], config["timezone"], today)
            if df_fcst_today.empty:
                print(f"   [!] Falló la obtención de la predicción Single Run de las 00:00 UTC para hoy. Omitiendo apuestas para {city}.")
                continue
                
            models = get_trained_models_for_city(df_all, today)
        except Exception as e:
            print(f"   Error al entrenar modelos / descargar datos para {city}: {e}")
            continue
            
        # Preparar vector de características para hoy
        df_yesterday_real = df_all[df_all["Fecha"] == (today - pd.Timedelta(days=1))]
        if df_yesterday_real.empty:
            df_yesterday_real = df_all.iloc[-1:]
            
        real_ayer_temp = df_yesterday_real["Temp_Max_Real"].values[0]
        real_ayer_hum = df_yesterday_real["Humidity_Real"].values[0]
        real_ayer_wind = df_yesterday_real["Wind_Speed_Real"].values[0]
        real_ayer_press = df_yesterday_real["Pressure_Real"].values[0]
        yesterday_850hPa = df_yesterday_real["Temp_Max_850hPa"].values[0]
        
        fcst_today = df_fcst_today.iloc[0]
        
        input_data = pd.DataFrame([{
            "Temp_Mean_850hPa": fcst_today["Temp_Mean_850hPa_FCST"],
            "Temp_Max_850hPa": fcst_today["Temp_Max_850hPa_FCST"],
            "Geopotential_Mean_500hPa": fcst_today["Geopotential_Mean_500hPa_FCST"],
            "Radiation_Sum": fcst_today["Radiation_Sum_FCST"],
            "Pressure_Mean_Sfc": fcst_today["Pressure_Mean_Sfc_FCST"],
            "RH_Mean_Sfc": fcst_today["RH_Mean_Sfc_FCST"],
            "Wind_Max_Sfc": fcst_today["Wind_Max_Sfc_FCST"],
            "Temp_Max_Real_Ayer": real_ayer_temp,
            "Humidity_Real_Ayer": real_ayer_hum,
            "Wind_Speed_Real_Ayer": real_ayer_wind,
            "Pressure_Real_Ayer": real_ayer_press,
            "Temp_Max_850hPa_Ayer": yesterday_850hPa,
            "Mes": today.month,
            "Dia_Ano": today.dayofyear
        }])
        
        bets_placed_today = []
        
        for bot_id, bot_data in models.items():
            model = bot_data["model"]
            features = bot_data["features"]
            sigma = bot_data["sigma"]
            
            # Predicción final con Downscaling Físico
            pred_ia = model.predict(input_data[features])[0]
            
            bot_balance = state["bots"][bot_id]["balance"]
            if bot_balance <= 0:
                print(f"   - {state['bots'][bot_id]['name']}: Banca a cero ($0). No puede operar.")
                continue
                
            choices = []
            for m in markets:
                prices_str = m.get("outcomePrices")
                if not prices_str: continue
                prices = json.loads(prices_str)
                if len(prices) < 1: continue
                
                price = float(prices[0])
                if price <= 0.05: price = 0.05
                
                col_name = m.get("groupItemTitle")
                prob = get_probability_from_distribution(col_name, pred_ia, sigma)
                edge = prob - price
                
                # Regla óptima de la V3/V4: Edge mínimo de 8%
                if prob >= MIN_PROB and edge >= MIN_EDGE:
                    choices.append({
                        "market_id": m.get("id"),
                        "option": col_name,
                        "price": price,
                        "edge": edge,
                        "prob": prob
                    })
                    
            choices = sorted(choices, key=lambda x: x["edge"], reverse=True)[:MAX_OPTS]
            
            if choices:
                bet_amount = min(10.0, bot_balance)
                money_per_bet = bet_amount / len(choices)
                
                state["bots"][bot_id]["balance"] -= bet_amount
                
                for c in choices:
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
                        "pred_sat_temp": round(float(fcst_today["Temp_Max_Global_Sfc_FCST"]), 2),
                        "ia_bias_correction": round(float(pred_ia - fcst_today["Temp_Max_Global_Sfc_FCST"]), 2),
                        "sigma": round(float(sigma), 2),
                        "validation_mae_9d": round(float(bot_data["val_mae"]), 2) if bot_data["val_mae"] is not None else None,
                        "validation_acc_9d": round(float(bot_data["val_acc"]), 1) if bot_data["val_acc"] is not None else None,
                        "result": "Pendiente"
                    })
                    val_err_str = f"Val MAE: {bot_data['val_mae']:.2f}°C, Val Acc: {bot_data['val_acc']:.1f}%" if bot_data["val_mae"] is not None else "N/A"
                    print(f"   - {state['bots'][bot_id]['name']}: Apostó ${money_per_bet:.2f} a '{c['option']}' (Precio: ${effective_price:.3f}, IA: {c['prob']*100:.1f}%, Satélite: {fcst_today['Temp_Max_Global_Sfc_FCST']:.2f}°C, Pred_IA: {pred_ia:.2f}°C, {val_err_str})")
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
    
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        
    print("\n[+] Simulación diaria V4 completada con éxito y balances guardados.")
    print("=" * 77)

if __name__ == "__main__":
    main()
