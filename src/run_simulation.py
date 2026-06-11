import os
import json
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import scipy.stats as stats
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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
        "city_slug": "london"
    }
}

MIN_PROB = 0.25
MIN_EDGE = 0.00
MAX_OPTS = 2

# ----------------- DATA DOWNLOAD HELPERS -----------------

def download_ground_truth_iem(station_id, past_days, tz):
    end = datetime.now()
    start = datetime.now() - pd.Timedelta(days=past_days + 30)
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station": station_id,
        "data": "tmpc,relh,sknt,mslp,alti",
        "year1": str(start.year), "month1": str(start.month), "day1": str(start.day),
        "year2": str(end.year), "month2": str(end.month), "day2": str(end.day),
        "tz": tz,
        "format": "onlydata"
    }
    r = requests.get(url, params=params)
    lines = [line.split(",") for line in r.text.strip().split("\n") if not line.startswith("#") and "," in line]
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
    df_daily = df_raw.groupby("Fecha").agg(
        Temp_Max_Real=("tmpc", "max"),
        Humidity_Real=("relh", "mean"),
        Wind_Speed_Real=("wspd_kmh", "max"),
        Pressure_Real=("mslp", "mean")
    ).reset_index()
    df_daily["Fecha"] = pd.to_datetime(df_daily["Fecha"])
    return df_daily.ffill().bfill()

def download_forecasts(lat, lon, past_days, tz):
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "past_days": past_days + 15,
        "hourly": ["temperature_2m", "relative_humidity_2m", "cloud_cover", "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
        "timezone": tz
    }
    r = requests.get(url, params=params)
    data = r.json()
    df_hourly = pd.DataFrame({
        "Fecha_Hora": pd.to_datetime(data["hourly"]["time"]),
        "Temp_Predicha": data["hourly"]["temperature_2m"],
        "Humidity_Predicha": data["hourly"]["relative_humidity_2m"],
        "Cloud_Predicha": data["hourly"]["cloud_cover"],
        "Pressure_Predicha": data["hourly"]["surface_pressure"],
        "Wind_Speed_Predicha": data["hourly"]["wind_speed_10m"],
        "Wind_Dir_Predicha": data["hourly"]["wind_direction_10m"]
    })
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
    df_real = download_ground_truth_iem(config["station_id"], 1100, config["timezone"])
    df_fcst = download_forecasts(config["lat"], config["lon"], 1100, config["timezone"])
    df = pd.merge(df_real, df_fcst, on="Fecha", how="inner")
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
    return df.dropna()

# ----------------- TRAINING AND PREDICTION -----------------

def get_trained_models_for_city(df_all):
    # Train end strictly before May 1, 2026
    train_end = pd.to_datetime('2026-05-01')
    
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
    
    # 1. Train Bot V1 (Base - 3.0 years, max_depth=4, no trends)
    df_v1 = df_all[df_all["Fecha"] < train_end].copy()
    model_v1 = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    model_v1.fit(df_v1[base_features], df_v1["Error_Base"])
    pred_v1_train = df_v1["Temp_Max_Predicha"] + model_v1.predict(df_v1[base_features])
    sigma_v1 = (df_v1["Temp_Max_Real"] - pred_v1_train).std()
    
    # 2. Train Bot Candidate A (Max ROI - 2.5 years/912 days, max_depth=4, no trends)
    train_start_a = train_end - pd.Timedelta(days=912)
    df_a = df_all[(df_all["Fecha"] < train_end) & (df_all["Fecha"] >= train_start_a)].copy()
    model_a = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    model_a.fit(df_a[base_features], df_a["Error_Base"])
    pred_a_train = df_a["Temp_Max_Predicha"] + model_a.predict(df_a[base_features])
    sigma_a = (df_a["Temp_Max_Real"] - pred_a_train).std()
    
    # 3. Train Bot Candidate B (Max Accuracy - 2.0 years/730 days, max_depth=3, with trends)
    train_start_b = train_end - pd.Timedelta(days=730)
    df_b = df_all[(df_all["Fecha"] < train_end) & (df_all["Fecha"] >= train_start_b)].copy()
    model_b = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    model_b.fit(df_b[trend_features], df_b["Error_Base"])
    pred_b_train = df_b["Temp_Max_Predicha"] + model_b.predict(df_b[trend_features])
    sigma_b = (df_b["Temp_Max_Real"] - pred_b_train).std()
    
    return {
        "bot_v1": {"model": model_v1, "features": base_features, "sigma": sigma_v1},
        "bot_cand_a": {"model": model_a, "features": base_features, "sigma": sigma_a},
        "bot_cand_b": {"model": model_b, "features": trend_features, "sigma": sigma_b}
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
        r = requests.get(url)
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
        r = requests.get(search_url, params=params)
        if r.status_code == 200:
            events = r.json()
            for e in events:
                # Match title words for date e.g., "June 12" and "2026"
                title = e.get("title", "").lower()
                if mes in title and str(dia) in title and str(ano) in title:
                    return e
    except:
        pass
    return None

# ----------------- MAIN PIPELINE RUNNER -----------------

def main():
    state_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "data", "simulation_state.json")
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
        
        # We need to query Polymarket to see which market won
        markets_resolved = []
        resolved_winner = None
        
        # Retrieve all markets involved to verify status
        for b in bet_group.get("bets", []):
            m_id = b.get("market_id")
            m_url = f"https://gamma-api.polymarket.com/markets/{m_id}"
            try:
                mr = requests.get(m_url)
                if mr.status_code == 200:
                    m_data = mr.json()
                    if m_data.get("resolved") == True:
                        markets_resolved.append(m_data)
            except Exception as e:
                print(f"   Error al consultar mercado {m_id}: {e}")
                
        # If the markets are resolved, find the winner option
        if len(markets_resolved) > 0 and all(m.get("resolved") for m in markets_resolved):
            # The winning market is the one where Yes outcomePrices is 1.0 (or close to it)
            for m in markets_resolved:
                outcome_prices = m.get("outcomePrices")
                if outcome_prices:
                    prices = json.loads(outcome_prices)
                    if len(prices) >= 1 and float(prices[0]) > 0.95:
                        resolved_winner = m.get("groupItemTitle")
                        break
                        
            if resolved_winner:
                print(f"   ¡Mercado resuelto oficialmente! Ganador: {resolved_winner}")
                # Distribute payoffs
                for b in bet_group.get("bets", []):
                    bot_id = b.get("bot")
                    invested = b.get("invested")
                    option_bought = b.get("option")
                    buy_price = b.get("price")
                    
                    bot_ref = state["bots"][bot_id]
                    bot_ref["trades_count"] += 1
                    
                    if option_bought == resolved_winner:
                        payoff = invested / buy_price
                        net_profit = payoff - invested
                        bot_ref["balance"] += payoff
                        bot_ref["wins"] += 1
                        b["result"] = f"+{net_profit:.2f} USD"
                        print(f"   - {bot_ref['name']}: GANADOR (Pago: +${payoff:.2f})")
                    else:
                        b["result"] = f"-{invested:.2f} USD"
                        print(f"   - {bot_ref['name']}: PERDEDOR (Pago: $0.00)")
                        
                # Recalculate ROI and Win Rate
                for bot_id, bot_ref in state["bots"].items():
                    initial = bot_ref["initial_balance"]
                    current = bot_ref["balance"]
                    bot_ref["roi"] = round(((current - initial) / initial) * 100, 2)
                    bot_ref["win_rate"] = round((bot_ref["wins"] / bot_ref["trades_count"]) * 100, 2) if bot_ref["trades_count"] > 0 else 0.0
                    
                # Add to resolved logs
                resolved_bets.append({
                    "date": bet_date_str,
                    "city": city,
                    "question": bet_group.get("question"),
                    "winner_option": resolved_winner,
                    "bets": bet_group.get("bets")
                })
            else:
                print("   Error: Todos los mercados cerrados pero no se pudo determinar un ganador. Reintentando mañana.")
                new_active_bets.append(bet_group)
        else:
            print("   Mercado aún no resuelto en la API de Polymarket. Se reintentará en el próximo ciclo.")
            new_active_bets.append(bet_group)
            
    state["active_bets"] = new_active_bets
    state["resolved_bets"] = resolved_bets

    # 2. RUN TODAY'S PREDICTIONS AND SIMULATE NEW BETS
    print(f"\n[+] Buscando y evaluando nuevos mercados para hoy ({today_str})...")
    
    for city, config in CITIES.items():
        print(f"\n[{city.upper()}]:")
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
            models = get_trained_models_for_city(df_all)
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
                    bets_placed_today.append({
                        "bot": bot_id,
                        "market_id": c["market_id"],
                        "option": c["option"],
                        "price": c["price"],
                        "invested": round(money_per_bet, 2),
                        "prob_ia": round(c["prob"] * 100, 1),
                        "pred_ia_temp": round(pred_ia, 2),
                        "result": "Pendiente"
                    })
                    print(f"   - {state['bots'][bot_id]['name']}: Apostó ${money_per_bet:.2f} a '{c['option']}' (Precio: ${c['price']:.2f}, IA: {c['prob']*100:.1f}%, Pred: {pred_ia:.2f}°C)")
            else:
                print(f"   - {state['bots'][bot_id]['name']}: Sin operaciones (No se encontró edge o probabilidad mínima).")
                
        if bets_placed_today:
            state["active_bets"].append({
                "date": today_str,
                "city": city,
                "question": event.get("title"),
                "bets": bets_placed_today
            })

    # 3. LOG CURVA DE CAPITAL HISTÓRICA
    history_entry = {
        "date": today_str,
        "bot_v1_balance": round(state["bots"]["bot_v1"]["balance"] + sum(b["invested"] for g in state["active_bets"] for b in g["bets"] if b["bot"] == "bot_v1"), 2),
        "bot_cand_a_balance": round(state["bots"]["bot_cand_a"]["balance"] + sum(b["invested"] for g in state["active_bets"] for b in g["bets"] if b["bot"] == "bot_cand_a"), 2),
        "bot_cand_b_balance": round(state["bots"]["bot_cand_b"]["balance"] + sum(b["invested"] for g in state["active_bets"] for b in g["bets"] if b["bot"] == "bot_cand_b"), 2)
    }
    
    # Overwrite today's history log if it exists, or append new one
    history_list = state.get("history", [])
    history_list = [h for h in history_list if h["date"] != today_str]
    history_list.append(history_entry)
    state["history"] = history_list

    # Save state
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        
    print("\n[+] Simulación diaria completada con éxito y balances guardados.")
    print("=" * 77)

if __name__ == "__main__":
    main()
