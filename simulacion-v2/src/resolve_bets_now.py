import os
import json
import requests
from datetime import datetime

# Ajustar ruta de trabajo al directorio raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    except Exception as e:
        print(f"   [Error Evento] Falló obtener evento exacto ({exact_slug}): {e}")
        
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
    except Exception as e:
        print(f"   [Error Evento] Falló búsqueda general de eventos: {e}")
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

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    state_file = os.path.join(repo_root, "docs", "simulacion-v2", "data", "simulation_state.json")
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
            print("   Mercado aún no resuelto en la API de Polymarket. Queda pendiente.")
            new_active_bets.append(bet_group)
            
    # Siempre actualizar el historial de capital y guardar el estado para mantener la coherencia
    state["active_bets"] = new_active_bets
    state["resolved_bets"] = resolved_bets
    update_history_curve(state)
    recalculate_statistics(state)
    
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        
    if any_updated:
        print("\n[+] El archivo simulation_state.json y su historial de capital han sido actualizados con los nuevos resultados oficiales.")
    else:
        print("\n[+] Se ha recalculado el historial de capital. simulation_state.json guardado sin nuevos resultados.")
    print("=" * 77)

if __name__ == "__main__":
    main()
