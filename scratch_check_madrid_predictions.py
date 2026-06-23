import json
import os
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import scipy.stats as stats
import warnings
import pytz

warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from run_simulation import build_dataset, get_trained_models_for_city, fetch_active_polymarket_event, get_probability_from_distribution, CITIES, MIN_PROB, MIN_EDGE

def main():
    today = datetime.now()
    city = "Madrid"
    config = CITIES[city]
    
    print("Fetching active event...")
    event = fetch_active_polymarket_event(config["city_slug"], today)
    if not event:
        print("No active event found.")
        return
        
    print(f"Event Title: {event.get('title')}")
    markets = event.get("markets", [])
    print(f"Found {len(markets)} sub-markets:")
    for m in markets:
        print(f"  Option: {m.get('groupItemTitle')} - ID: {m.get('id')} - Prices: {m.get('outcomePrices')}")
        
    print("\nBuilding dataset and running predictions...")
    try:
        df_all = build_dataset(city, config)
        target_date_dt = pd.to_datetime(today.date())
        models = get_trained_models_for_city(df_all, target_date_dt)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    today_row = df_all[df_all["Fecha"].dt.date == today.date()]
    if today_row.empty:
        today_row = df_all.iloc[-1:]
        
    pred_sat = float(today_row["Temp_Max_Predicha"].values[0])
    print(f"Satellite Forecast Max Temp today: {pred_sat:.2f}°C")
    
    for bot_id, bot_data in models.items():
        print(f"\n--- {bot_id.upper()} ---")
        model = bot_data["model"]
        features = bot_data["features"]
        sigma = bot_data["sigma"]
        
        pred_bias = model.predict(today_row[features])[0]
        pred_ia = pred_sat + pred_bias
        print(f"  Predicted Bias: {pred_bias:+.2f}°C")
        print(f"  Predicted IA Temperature: {pred_ia:.2f}°C")
        print(f"  Sigma (Std Dev): {sigma:.2f}°C")
        
        choices = []
        for m in markets:
            prices_str = m.get("outcomePrices")
            if not prices_str: continue
            prices = json.loads(prices_str)
            if len(prices) < 1: continue
            price = float(prices[0])
            col_name = m.get("groupItemTitle")
            
            prob = get_probability_from_distribution(col_name, pred_ia, sigma)
            edge = prob - price
            print(f"    Option: {col_name:<12} | Price: {price:.3f} | IA Prob: {prob*100:5.1f}% | Edge: {edge:+.3f}")
            
            if prob >= MIN_PROB and edge >= MIN_EDGE:
                choices.append({
                    "option": col_name,
                    "price": price,
                    "edge": edge,
                    "prob": prob
                })
        
        choices = sorted(choices, key=lambda x: x["edge"], reverse=True)
        print("  Eligible choices:")
        for c in choices[:2]:
            print(f"    * {c['option']} (price: {c['price']:.3f}, prob: {c['prob']*100:.1f}%, edge: {c['edge']:+.3f})")

if __name__ == "__main__":
    # We must make sure we run this from a script that can access the import run_simulation
    # Let's copy it to simulacion-v1 folder first or run it from there.
    main()
