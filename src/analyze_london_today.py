import requests
import json
import pandas as pd
from datetime import datetime

STATION_ID = "EGLC"
TZ = "Europe/London"
LAT = 51.505
LON = 0.055

def get_live_weather_openmeteo():
    # Fetch real-time weather from Open-Meteo for EGLC coordinates
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "timezone": TZ,
        "forecast_days": 1
    }
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame({
                "time": pd.to_datetime(data["hourly"]["time"]),
                "temp": data["hourly"]["temperature_2m"]
            })
            # Filter up to current local hour (current local time is 13:07)
            now = datetime.now()
            df = df[df["time"] <= now]
            return df
    except Exception as e:
        print(f"Error fetching Open-Meteo: {e}")
    return pd.DataFrame()

def get_live_weather_asos():
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    now = datetime.now()
    params = {
        "station": STATION_ID,
        "data": "tmpc",
        "year1": str(now.year), "month1": str(now.month), "day1": str(now.day),
        "year2": str(now.year), "month2": str(now.month), "day2": str(now.day + 1 if now.day < 28 else now.day),
        "tz": TZ,
        "format": "onlydata"
    }
    try:
        r = requests.get(url, params=params)
        lines = [line.split(",") for line in r.text.strip().split("\n") if not line.startswith("#") and "," in line]
        if len(lines) > 1:
            df = pd.DataFrame(lines[1:], columns=lines[0])
            df["tmpc"] = pd.to_numeric(df["tmpc"], errors="coerce")
            df["valid"] = pd.to_datetime(df["valid"])
            return df.dropna().sort_values("valid")
    except Exception as e:
        print(f"Error fetching ASOS: {e}")
    return pd.DataFrame()

def get_polymarket_prices():
    url = "https://gamma-api.polymarket.com/events?slug=highest-temperature-in-london-on-june-12-2026"
    r = requests.get(url)
    if r.status_code == 200 and r.json():
        event = r.json()[0]
        markets = []
        for m in event.get("markets", []):
            prices_str = m.get("outcomePrices")
            yes_price = 0.0
            if prices_str:
                prices = json.loads(prices_str)
                if len(prices) >= 1:
                    yes_price = float(prices[0])
            markets.append({
                "option": m.get("groupItemTitle"),
                "price": yes_price,
                "question": m.get("question")
            })
        return event.get("title"), markets
    return None, []

def main():
    print("======================================================================")
    print(" ANÁLISIS EN TIEMPO REAL: LONDRES (EGLC) - HOY 12 DE JUNIO")
    print("======================================================================")
    
    # 1. Weather info
    df_weather_om = get_live_weather_openmeteo()
    df_weather_as = get_live_weather_asos()
    
    print("\n[+] TEMPERATURAS REGISTRADAS HOY EN LONDRES (EGLC):")
    
    max_temp = -99.0
    max_time = ""
    
    # Check ASOS first, fallback to Open-Meteo
    if df_weather_as is not None and not df_weather_as.empty:
        print("--- (Fuente: Estación Meteorológica Aeropuerto London City - ASOS/METAR) ---")
        for idx, row in df_weather_as.iterrows():
            print(f"  {row['valid'].strftime('%H:%M')} : {row['tmpc']:.1f} °C")
            if row["tmpc"] > max_temp:
                max_temp = row["tmpc"]
                max_time = row['valid'].strftime('%H:%M')
    elif df_weather_om is not None and not df_weather_om.empty:
        print("--- (Fuente: Open-Meteo Real-Time Analysis) ---")
        for idx, row in df_weather_om.iterrows():
            print(f"  {row['time'].strftime('%H:%M')} : {row['temp']:.1f} °C")
            if row["temp"] > max_temp:
                max_temp = row["temp"]
                max_time = row['time'].strftime('%H:%M')
    else:
        print("No se pudieron recuperar temperaturas en vivo para Londres.")
        
    if max_temp > -90.0:
        print(f"\n>> Temperatura Máxima Registrada hasta ahora: {max_temp:.1f}°C a las {max_time}")
    
    # 2. Polymarket info
    title, markets = get_polymarket_prices()
    if title:
        print(f"\n[+] COTIZACIÓN ACTUAL EN POLYMARKET:")
        print(f"Mercado: {title}")
        print("-" * 60)
        print(f"{'Opción Bracket':<22} | {'Precio YES (Prob. Implícita)':<30}")
        print("-" * 60)
        
        # Sort options (e.g. numeric sort)
        def sort_key(m):
            opt = m["option"]
            if "below" in opt: return 0
            if "higher" in opt: return 99
            try:
                return int(opt.replace("°C", "").strip())
            except:
                return 50
        
        sorted_markets = sorted(markets, key=sort_key)
        for m in sorted_markets:
            print(f"{m['option']:<22} | ${m['price']:.3f} ({m['price']*100:.1f}%)")
    else:
        print("\nNo se encontró evento activo en Polymarket para Londres hoy.")
        
    print("======================================================================")

if __name__ == "__main__":
    main()
