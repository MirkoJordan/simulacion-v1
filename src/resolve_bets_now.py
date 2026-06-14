import os
import json
import requests
from datetime import datetime

# Ajustar ruta de trabajo al directorio raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fetch_active_polymarket_event(city_slug, target_date):
    mes = target_date.strftime('%B').lower()
    dia = target_date.day
    ano = target_date.year
    exact_slug = f"highest-temperature-in-{city_slug}-on-{mes}-{dia}-{ano}"
    
    url = f"https://gamma-api.polymarket.com/events?slug={exact_slug}"
    try:
        r = requests.get(url, timeout=10)
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
        r = requests.get(search_url, params=params, timeout=10)
        if r.status_code == 200:
            events = r.json()
            for e in events:
                title = e.get("title", "").lower()
                if mes in title and str(dia) in title and str(ano) in title:
                    return e
    except:
        pass
    return None

def main():
    state_file = os.path.join(PROJECT_ROOT, "docs", "data", "simulation_state.json")
    if not os.path.exists(state_file):
        print(f"[❌ ERROR] No se encontró el archivo de estado en {state_file}")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
        
    active_bets = state.get("active_bets", [])
    resolved_bets = state.get("resolved_bets", [])
    new_active_bets = []
    
    any_updated = False
    print("======================= RESOLUCIÓN MANUAL DE APUESTAS =======================")

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
                mr = requests.get(m_url, timeout=10)
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
            any_updated = True
            
            # Determinar opción ganadora
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
                    print(f"   No se pudo obtener el nombre exacto de la opción ganadora: {e}")
            
            print(f"   Ganador oficial determinado: {resolved_winner}")
            
            # Distribuir ganancias
            for b, m_data in resolved_bets_data:
                bot_id = b.get("bot")
                invested = b.get("invested")
                option_bought = b.get("option")
                buy_price = b.get("price")
                
                bot_ref = state["bots"][bot_id]
                bot_ref["trades_count"] += 1
                
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
                    bot_ref["wins"] += 1
                    b["result"] = f"+{net_profit:.2f} USD"
                    print(f"   - {bot_ref['name']}: GANADOR de {option_bought} (Pago: +${payoff:.2f})")
                else:
                    b["result"] = f"-{invested:.2f} USD"
                    print(f"   - {bot_ref['name']}: PERDEDOR de {option_bought} (Pago: $0.00)")
                    
            # Recalcular ROI y Win Rate de los bots
            for bot_id, bot_ref in state["bots"].items():
                initial = bot_ref["initial_balance"]
                current = bot_ref["balance"]
                bot_ref["roi"] = round(((current - initial) / initial) * 100, 2)
                bot_ref["win_rate"] = round((bot_ref["wins"] / bot_ref["trades_count"]) * 100, 2) if bot_ref["trades_count"] > 0 else 0.0
                
            resolved_bets.append({
                "date": bet_date_str,
                "city": city,
                "question": bet_group.get("question"),
                "winner_option": resolved_winner,
                "bets": bet_group.get("bets")
            })
        else:
            print("   Mercado aún no resuelto en la API de Polymarket. Queda pendiente.")
            new_active_bets.append(bet_group)
            
    if any_updated:
        state["active_bets"] = new_active_bets
        state["resolved_bets"] = resolved_bets
        
        # Guardar el estado actualizado
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            
        print("\n[+] El archivo simulation_state.json ha sido actualizado con los resultados oficiales.")
    else:
        print("\n[.] No se detectaron mercados resueltos recientemente. No hubo cambios en el estado.")
    print("=" * 77)

if __name__ == "__main__":
    main()
