document.addEventListener("DOMContentLoaded", () => {
    // Path to the simulation state JSON file
    const dataUrl = "data/simulation_state.json";

    fetch(dataUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error("No se pudo cargar el archivo JSON de estado");
            }
            return response.json();
        })
        .then(data => {
            updateDashboard(data);
        })
        .catch(error => {
            console.error("Error cargando simulación:", error);
            document.getElementById("update-badge").innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Error al cargar datos`;
        });
});

function updateDashboard(data) {
    // 1. Update Badge / Last Updated Time
    const badge = document.getElementById("update-badge");
    if (data.history && data.history.length > 0) {
        const lastDate = data.history[data.history.length - 1].date;
        badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Último día: ${lastDate}`;
    } else {
        badge.innerHTML = `<i class="fa-solid fa-clock"></i> Simulación Iniciada`;
    }

    // 2. Update Bot Summary Cards
    const bots = data.bots;

    // Bot V1
    document.getElementById("balance-v1").innerText = `$${bots.bot_v1.balance.toFixed(2)}`;
    document.getElementById("roi-v1").innerText = `${bots.bot_v1.roi > 0 ? "+" : ""}${bots.bot_v1.roi.toFixed(2)}%`;
    document.getElementById("accuracy-v1").innerText = `${bots.bot_v1.win_rate.toFixed(1)}%`;
    document.getElementById("ops-v1").innerText = bots.bot_v1.trades_count;
    adjustRoiColor("roi-v1", bots.bot_v1.roi);

    // Bot A
    document.getElementById("balance-a").innerText = `$${bots.bot_cand_a.balance.toFixed(2)}`;
    document.getElementById("roi-a").innerText = `${bots.bot_cand_a.roi > 0 ? "+" : ""}${bots.bot_cand_a.roi.toFixed(2)}%`;
    document.getElementById("accuracy-a").innerText = `${bots.bot_cand_a.win_rate.toFixed(1)}%`;
    document.getElementById("ops-a").innerText = bots.bot_cand_a.trades_count;
    adjustRoiColor("roi-a", bots.bot_cand_a.roi);

    // Bot B
    document.getElementById("balance-b").innerText = `$${bots.bot_cand_b.balance.toFixed(2)}`;
    document.getElementById("roi-b").innerText = `${bots.bot_cand_b.roi > 0 ? "+" : ""}${bots.bot_cand_b.roi.toFixed(2)}%`;
    document.getElementById("accuracy-b").innerText = `${bots.bot_cand_b.win_rate.toFixed(1)}%`;
    document.getElementById("ops-b").innerText = bots.bot_cand_b.trades_count;
    adjustRoiColor("roi-b", bots.bot_cand_b.roi);

    // 3. Render Capital History Chart
    renderChart(data.history);

    // 4. Render Active Bets
    renderActiveBets(data.active_bets);

    // 5. Render Resolved Bets History
    renderResolvedBets(data.resolved_bets);
}

function adjustRoiColor(elementId, roiValue) {
    const el = document.getElementById(elementId);
    if (roiValue > 0) {
        el.style.color = "#34D399"; // Green
    } else if (roiValue < 0) {
        el.style.color = "#F87171"; // Red
    } else {
        el.style.color = "#FFF";
    }
}

function renderChart(history) {
    const ctx = document.getElementById("capital-chart").getContext("2d");
    
    // Extract dates and balance curves
    const labels = history.map(h => h.date);
    const v1Data = history.map(h => h.bot_v1_balance);
    const aData = history.map(h => h.bot_cand_a_balance);
    const bData = history.map(h => h.bot_cand_b_balance);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Bot V1 (Base)',
                    data: v1Data,
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.05)',
                    tension: 0.2,
                    fill: true
                },
                {
                    label: 'Bot A (Max ROI)',
                    data: aData,
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.05)',
                    tension: 0.2,
                    fill: true
                },
                {
                    label: 'Bot B (Max Acierto)',
                    data: bData,
                    borderColor: '#EC4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.05)',
                    tension: 0.2,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9FA6B2' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9FA6B2' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#F0F2F5', font: { family: 'Outfit' } }
                }
            }
        }
    });
}

function renderActiveBets(activeBets) {
    const container = document.getElementById("active-bets-container");
    if (!activeBets || activeBets.length === 0) {
        container.innerHTML = `<p class="empty-msg">No hay apuestas activas para hoy o esperando resolución.</p>`;
        return;
    }

    container.innerHTML = "";
    activeBets.forEach(group => {
        const div = document.createElement("div");
        div.className = "active-bet-group";
        
        let betsHtml = "";
        group.bets.forEach(b => {
            const botDotClass = b.bot === "bot_v1" ? "dot-v1" : (b.bot === "bot_cand_a" ? "dot-a" : "dot-b");
            const botLabel = b.bot === "bot_v1" ? "V1 Base" : (b.bot === "bot_cand_a" ? "Bot A (ROI)" : "Bot B (Acc)");
            betsHtml += `
                <div class="bet-bot-row">
                    <span class="bet-bot-name">
                        <span class="bet-dot ${botDotClass}"></span>
                        ${botLabel}
                    </span>
                    <span class="bet-details-tag">
                        ${b.option} @ $${b.price.toFixed(2)} | Inv: $${b.invested.toFixed(2)} | IA: ${b.prob_ia}% (Pred: ${b.pred_ia_temp}°C)
                    </span>
                </div>
            `;
        });

        div.innerHTML = `
            <div class="bet-group-header">
                <span>${group.city} - ${group.question}</span>
                <span style="color: var(--accent-v1);">${group.date}</span>
            </div>
            ${betsHtml}
        `;
        container.appendChild(div);
    });
}

function renderResolvedBets(resolvedBets) {
    const tbody = document.getElementById("resolved-bets-tbody");
    if (!resolvedBets || resolvedBets.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="empty-msg">No hay apuestas resueltas todavía.</td></tr>`;
        return;
    }

    tbody.innerHTML = "";
    // Show latest resolved bets first
    [...resolvedBets].reverse().forEach(row => {
        const tr = document.createElement("tr");
        
        let detailsHtml = "";
        row.bets.forEach(b => {
            const botLabel = b.bot === "bot_v1" ? "V1" : (b.bot === "bot_cand_a" ? "Bot A" : "Bot B");
            const isWin = b.result.includes("+");
            const pillClass = isWin ? "pill-win" : "pill-loss";
            detailsHtml += `
                <span class="table-bet-pill ${pillClass}">
                    ${botLabel}: ${b.option} (${b.result})
                </span>
            `;
        });

        tr.innerHTML = `
            <td><strong>${row.date}</strong></td>
            <td>${row.city}</td>
            <td><span class="table-bet-pill pill-win"><i class="fa-solid fa-award"></i> ${row.winner_option}</span></td>
            <td>${detailsHtml}</td>
        `;
        tbody.appendChild(tr);
    });
}
