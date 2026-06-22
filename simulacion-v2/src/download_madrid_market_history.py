import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time

CITIES_SLUG = "madrid"
START_DATE = datetime(2026, 6, 2)
END_DATE = datetime(2026, 6, 11)

OUTPUT_DIR = r"c:\Users\fjorp\OneDrive\Escritorio\Polymarket\prueba_real_interfaz_y_3_modelos\analisis_mercado_madrid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_event_for_date(date_dt):
    mes = date_dt.strftime('%B').lower()
    dia = date_dt.day
    ano = date_dt.year
    slug = f"highest-temperature-in-{CITIES_SLUG}-on-{mes}-{dia}-{ano}"
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    try:
        r = requests.get(url)
        if r.status_code == 200 and r.json():
            return r.json()[0]
    except Exception as e:
        print(f"Error fetching event {slug}: {e}")
    return None

def download_price_history(token_id, fidelity=5):
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
    print("=== STARTING DOWNLOAD OF MADRID MARKET HISTORY ===")
    curr_date = START_DATE
    while curr_date <= END_DATE:
        date_str = curr_date.strftime("%Y-%m-%d")
        print(f"\nProcessing date: {date_str}...")
        
        event = fetch_event_for_date(curr_date)
        if not event:
            print(f"  No event found on Polymarket for {date_str}")
            curr_date += timedelta(days=1)
            continue
            
        print(f"  Event Title: {event.get('title')}")
        
        # Store metadata for all markets on this date
        markets_metadata = []
        
        for m in event.get("markets", []):
            title = m.get("groupItemTitle")
            m_id = m.get("id")
            clob_token_ids_str = m.get("clobTokenIds")
            
            if not clob_token_ids_str:
                continue
                
            try:
                clob_tokens = json.loads(clob_token_ids_str)
                # The first token is YES
                yes_token = clob_tokens[0]
            except Exception as e:
                print(f"  Could not parse token IDs for market {title}: {e}")
                continue
                
            print(f"    Fetching history for option {title} (Token: {yes_token[:10]}...)...")
            history = download_price_history(yes_token, fidelity=5) # 5-minute intervals
            
            if history:
                filename = f"madrid_{date_str}_{title.replace('°', '').replace('/', '_')}.json"
                file_path = os.path.join(OUTPUT_DIR, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "date": date_str,
                        "option": title,
                        "market_id": m_id,
                        "token_id": yes_token,
                        "history": history
                    }, f, indent=2)
                print(f"      Saved {len(history)} price points to {filename}")
            else:
                print(f"      No price history returned for option {title}")
                
            markets_metadata.append({
                "market_id": m_id,
                "option": title,
                "token_id": yes_token,
                "resolved": m.get("resolved"),
                "outcomePrices": m.get("outcomePrices")
            })
            
            # Rate limiting safety
            time.sleep(0.5)
            
        # Save daily metadata summary
        summary_path = os.path.join(OUTPUT_DIR, f"summary_{date_str}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "date": date_str,
                "event_title": event.get("title"),
                "markets": markets_metadata
            }, f, indent=2)
            
        curr_date += timedelta(days=1)
        
    print("\n=== DOWNLOAD COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
