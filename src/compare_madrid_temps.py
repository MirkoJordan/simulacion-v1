import requests
import pandas as pd
from datetime import datetime

STATION_ID = "LEMD"
TZ = "Europe/Madrid"

def get_hourly_data(day_str):
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    # day_str: YYYY-MM-DD
    dt = pd.to_datetime(day_str)
    next_day = dt + pd.Timedelta(days=1)
    
    params = {
        "station": STATION_ID,
        "data": "tmpc",
        "year1": str(dt.year), "month1": str(dt.month), "day1": str(dt.day),
        "year2": str(next_day.year), "month2": str(next_day.month), "day2": str(next_day.day),
        "tz": TZ,
        "format": "onlydata"
    }
    r = requests.get(url, params=params)
    lines = [line.split(",") for line in r.text.strip().split("\n") if not line.startswith("#") and "," in line]
    df = pd.DataFrame(lines[1:], columns=lines[0])
    df["tmpc"] = pd.to_numeric(df["tmpc"], errors="coerce")
    df["valid"] = pd.to_datetime(df["valid"])
    df["Hora"] = df["valid"].dt.strftime("%H:%M")
    df.dropna(subset=["tmpc"], inplace=True)
    return df.sort_values("valid")[["Hora", "tmpc"]]

def main():
    print("[+] Descargando observaciones para comparar ayer y hoy...")
    df_ayer = get_hourly_data("2026-06-11")
    df_hoy = get_hourly_data("2026-06-12")
    
    # Merge on Hora
    df_compare = pd.merge(df_ayer, df_hoy, on="Hora", suffixes=("_Ayer", "_Hoy"))
    
    print("\n============================================================")
    print(" COMPARATIVA DE TEMPERATURAS HORARIAS EN LEMD (MADRID)")
    print("============================================================")
    print(f"{'Hora':<7} | {'Temp Ayer (11-Jun)':<20} | {'Temp Hoy (12-Jun)':<20} | {'Diferencia':<10}")
    print("-" * 65)
    
    # Print typical morning hours up to current time (17:40)
    for idx, row in df_compare.iterrows():
        hora_h = int(row["Hora"].split(":")[0])
        # Only show morning hours from 08:00 to 17:00
        if 8 <= hora_h <= 17:
            diff = row["tmpc_Hoy"] - row["tmpc_Ayer"]
            print(f"{row['Hora']:<7} | {row['tmpc_Ayer']:<17.1f} °C | {row['tmpc_Hoy']:<17.1f} °C | {diff:+.1f} °C")
            
    print("============================================================")

if __name__ == "__main__":
    main()
