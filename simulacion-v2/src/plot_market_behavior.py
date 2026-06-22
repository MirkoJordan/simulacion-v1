import os
import json
import pandas as pd
from datetime import datetime
import requests

# Try to import matplotlib, install if missing
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    import subprocess
    print("[+] Matplotlib is missing. Installing...")
    subprocess.run(["pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

DATA_DIR = r"c:\Users\fjorp\OneDrive\Escritorio\Polymarket\prueba_real_interfaz_y_3_modelos\analisis_mercado_madrid"
OUTPUT_IMG = os.path.join(DATA_DIR, "grafico_mercado_madrid_june10.png")
STATION_ID = "LEMD"
TZ = "Europe/Madrid"

# Download weather for June 10, 2026
def get_weather():
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station": STATION_ID,
        "data": "tmpc",
        "year1": "2026", "month1": "6", "day1": "10",
        "year2": "2026", "month2": "6", "day2": "11",
        "tz": TZ,
        "format": "onlydata"
    }
    r = requests.get(url, params=params)
    lines = [line.split(",") for line in r.text.strip().split("\n") if not line.startswith("#") and "," in line]
    df = pd.DataFrame(lines[1:], columns=lines[0])
    df["tmpc"] = pd.to_numeric(df["tmpc"], errors="coerce")
    df["valid"] = pd.to_datetime(df["valid"])
    df.dropna(subset=["tmpc"], inplace=True)
    df.sort_values("valid", inplace=True)
    # Filter only for June 10
    df = df[df["valid"].dt.date == pd.to_datetime("2026-06-10").date()]
    return df

def main():
    weather_df = get_weather()
    
    # Load 31C (False Favorite) and 32C (Winner) histories for June 10
    with open(os.path.join(DATA_DIR, "madrid_2026-06-10_31C.json"), "r") as f:
        data_31 = json.load(f)
    with open(os.path.join(DATA_DIR, "madrid_2026-06-10_32C.json"), "r") as f:
        data_32 = json.load(f)
        
    df_31 = pd.DataFrame(data_31["history"])
    df_32 = pd.DataFrame(data_32["history"])
    
    df_31["datetime"] = pd.to_datetime(df_31["t"], unit="s").dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)
    df_32["datetime"] = pd.to_datetime(df_32["t"], unit="s").dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)
    weather_df["valid"] = weather_df["valid"].dt.tz_localize(None)
    
    # Filter both to June 10 local day (e.g. between 08:00 and 22:00)
    start_t = pd.to_datetime("2026-06-10 08:00:00")
    end_t = pd.to_datetime("2026-06-10 22:00:00")
    
    df_31 = df_31[(df_31["datetime"] >= start_t) & (df_31["datetime"] <= end_t)]
    df_32 = df_32[(df_32["datetime"] >= start_t) & (df_32["datetime"] <= end_t)]
    weather_df = weather_df[(weather_df["valid"] >= start_t) & (weather_df["valid"] <= end_t)]
    
    # Set up matplotlib style (dark/premium look)
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    
    # Plot prices
    color_31 = '#ff5252' # Red for false favorite
    color_32 = '#2ecc71' # Green for winner
    
    ax1.plot(df_31["datetime"], df_31["p"], label="Precio Contrato YES 31°C (Perdedor)", color=color_31, linewidth=2.5)
    ax1.plot(df_32["datetime"], df_32["p"], label="Precio Contrato YES 32°C (Ganador)", color=color_32, linewidth=2.5)
    
    ax1.set_xlabel("Hora Local (Madrid)", fontsize=11, labelpad=10)
    ax1.set_ylabel("Precio del Contrato ($)", fontsize=11, labelpad=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    # Instantiate a second axis that shares the same x-axis for temperature
    ax2 = ax1.twinx()  
    color_temp = '#f39c12' # Orange for temperature
    ax2.plot(weather_df["valid"], weather_df["tmpc"], label="Temperatura Real (LEMD)", color=color_temp, linewidth=2.0, linestyle=":")
    ax2.set_ylabel("Temperatura Real (°C)", color=color_temp, fontsize=11, labelpad=10)
    ax2.tick_params(axis='y', labelcolor=color_temp)
    ax2.set_ylim(20, 35)
    
    # Title & Legends
    plt.title("El Gran Colapso de Polymarket: Madrid (10 de Junio de 2026)\nContratos de Temperatura Máxima vs Medición en Tiempo Real", fontsize=13, fontweight='bold', pad=15)
    
    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor='#111', edgecolor='#333')
    
    # Format X axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    
    # Add annotations for key moments
    # False favorite peaks at 17:40
    ax1.annotate('Favorito falso\ncotiza a $0.95', xy=(pd.to_datetime("2026-06-10 17:40:00"), 0.95), 
                 xytext=(pd.to_datetime("2026-06-10 14:00:00"), 0.8),
                 arrowprops=dict(facecolor=color_31, shrink=0.08, width=1, headwidth=6),
                 fontsize=10, color=color_31, bbox=dict(boxstyle="round,pad=0.3", fc="#222", ec="#444", lw=1))
                 
    # Max temp is reached (32.0C at 18:30)
    ax1.annotate('La Tmax real alcanza 32.0°C\nGanador definitivo', xy=(pd.to_datetime("2026-06-10 18:30:00"), 0.5), 
                 xytext=(pd.to_datetime("2026-06-10 19:30:00"), 0.5),
                 arrowprops=dict(facecolor=color_32, shrink=0.08, width=1, headwidth=6),
                 fontsize=10, color=color_32, bbox=dict(boxstyle="round,pad=0.3", fc="#222", ec="#444", lw=1))
                 
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"[+] Market progression chart generated successfully at: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
