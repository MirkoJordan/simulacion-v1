import os
import requests
import json
from datetime import datetime
import time

CITIES_SLUG = "madrid"
OUTPUT_DIR = r"c:\Users\fjorp\OneDrive\Escritorio\Polymarket\prueba_real_interfaz_y_3_modelos\analisis_mercado_madrid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_today_event():
    # Today is June 12, 2026
    url = "https://gamma-api.polymarket.com/events?slug=highest-temperature-in-madrid-on-june-12-2026"
    try:
        r = requests.get(url)
        if r.status_code == 200 and r.json():
            return r.json()[0]
    except Exception as e:
        print(f"Error fetching today's event: {e}")
    return None

def download_price_history(token_id, fidelity=1):
    url = "https://clob.polymarket.com/prices-history"
    params = {
        "market": token_id,
        "fidelity": fidelity,
        "interval": "max"
    }
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200:
            return r.json().get("history", [])
    except Exception as e:
        print(f"Error fetching price history for token {token_id}: {e}")
    return []

def main():
    print("=== DOWNLOADING MINUTE-BY-MINUTE PRICES FOR MADRID TODAY ===")
    event = fetch_today_event()
    if not event:
        print("Today's Madrid event was not found on Polymarket.")
        return
        
    print(f"Event Found: {event.get('title')}")
    
    for m in event.get("markets", []):
        title = m.get("groupItemTitle")
        clob_token_ids_str = m.get("clobTokenIds")
        
        if not clob_token_ids_str:
            continue
            
        try:
            clob_tokens = json.loads(clob_token_ids_str)
            yes_token = clob_tokens[0]
        except Exception as e:
            print(f"  Could not parse token IDs for option {title}: {e}")
            continue
            
        print(f"  Fetching 1-minute history for {title} (Token: {yes_token[:10]}...)...")
        history = download_price_history(yes_token, fidelity=1)
        
        if history:
            filename = f"madrid_today_minute_{title.replace('°', '').replace('/', '_')}.json"
            file_path = os.path.join(OUTPUT_DIR, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "date": "2026-06-12",
                    "option": title,
                    "token_id": yes_token,
                    "history": history
                }, f, indent=2)
            print(f"    Saved {len(history)} data points to {filename}")
        else:
            print(f"    No price history returned for option {title}")
            
        time.sleep(0.5)

if __name__ == "__main__":
    main()
