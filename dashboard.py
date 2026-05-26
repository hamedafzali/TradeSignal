import os
from flask import Flask, jsonify, render_template_string
from database import init_db, get_stats, get_recent_signals

app = Flask(__name__)

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Signal Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body { background:#0f1117; color:#e0e0e0; }
  .card { background:#1a1d27; border:1px solid #2a2d3a; }
  .card-title { color:#8b8fa8; font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
  .stat-val { font-size:2rem; font-weight:700; }
  .badge-correct   { background:#1a4a2e; color:#4ade80; }
  .badge-incorrect { background:#4a1a1a; color:#f87171; }
  .badge-pending   { background:#2a2d3a; color:#8b8fa8; }
  .badge-neutral   { background:#2a2520; color:#fbbf24; }
  .badge-BUY  { background:#1a3a4a; color:#38bdf8; }
  .badge-SELL { background:#3a1a2a; color:#f472b6; }
  .badge-STRONG { background:#3a2a0a; color:#fbbf24; }
  .badge-AI     { background:#1a2a3a; color:#a78bfa; }
  .badge-RULE   { background:#1a2a1a; color:#4ade80; }
  table { font-size:.85rem; }
  thead th { color:#8b8fa8; border-color:#2a2d3a !important; }
  tbody td { border-color:#2a2d3a !important; }
  .refresh-note { color:#555; font-size:.75rem; }
</style>
</head>
<body>
<div class="container-fluid py-4 px-4">
  <div class="d-flex justify-content-between align-items-center mb-4">
    <h4 class="mb-0">📈 Trading Signal Dashboard</h4>
    <span class="refresh-note" id="last-refresh"></span>
  </div>

  <!-- Stat cards -->
  <div class="row g-3 mb-4" id="stat-cards">
    <div class="col-6 col-md-3"><div class="card p-3"><div class="card-title">Total Users</div><div class="stat-val" id="s-users">—</div></div></div>
    <div class="col-6 col-md-3"><div class="card p-3"><div class="card-title">Total Signals</div><div class="stat-val" id="s-total">—</div></div></div>
    <div class="col-6 col-md-3"><div class="card p-3"><div class="card-title">Signals Today</div><div class="stat-val" id="s-today">—</div></div></div>
    <div class="col-6 col-md-3"><div class="card p-3"><div class="card-title">Accuracy</div><div class="stat-val" id="s-acc">—</div></div></div>
  </div>

  <!-- Charts row -->
  <div class="row g-3 mb-4">
    <div class="col-md-6"><div class="card p-3"><canvas id="chart-daily" height="160"></canvas></div></div>
    <div class="col-md-6"><div class="card p-3"><canvas id="chart-symbol" height="160"></canvas></div></div>
  </div>

  <!-- Recent signals + leaderboard row -->
  <div class="row g-3 mb-4">
    <div class="col-md-8">
      <div class="card p-3">
        <h6 class="mb-3">Recent Signals</h6>
        <div class="table-responsive">
          <table class="table table-dark table-hover mb-0">
            <thead><tr>
              <th>Time</th><th>Symbol</th><th>Action</th><th>Strength</th>
              <th>Price</th><th>RSI</th><th>AI Conf.</th><th>Outcome</th><th>Outcome Price</th>
            </tr></thead>
            <tbody id="signals-body"></tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="col-md-4">
      <div class="card p-3 h-100">
        <h6 class="mb-3">🏆 Top Traders</h6>
        <table class="table table-dark table-hover mb-0">
          <thead><tr><th>User</th><th>Trades</th><th>Avg P&L</th></tr></thead>
          <tbody id="leaderboard-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
let dailyChart, symbolChart;

function badge(text, cls) {
  return `<span class="badge rounded-pill ${cls} px-2 py-1">${text}</span>`;
}

function outcomeBadge(o) {
  if (o === 'correct')   return badge('✓ Correct',   'badge-correct');
  if (o === 'incorrect') return badge('✗ Wrong',     'badge-incorrect');
  if (o === 'neutral')   return badge('~ Neutral',   'badge-neutral');
  return badge('pending', 'badge-pending');
}

function formatTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso + 'Z');
  return d.toLocaleString();
}

async function refresh() {
  const [stats, signals] = await Promise.all([
    fetch('/api/stats').then(r => r.json()),
    fetch('/api/signals').then(r => r.json()),
  ]);

  document.getElementById('s-users').textContent = stats.total_users;
  document.getElementById('s-total').textContent = stats.total_signals;
  document.getElementById('s-today').textContent = stats.today_signals;
  document.getElementById('s-acc').textContent = stats.resolved > 0
    ? stats.accuracy + '%' : 'N/A';
  document.getElementById('last-refresh').textContent =
    'Updated ' + new Date().toLocaleTimeString();

  // Daily chart
  const days   = stats.daily.map(d => d.day.slice(5));
  const counts = stats.daily.map(d => d.count);
  if (!dailyChart) {
    dailyChart = new Chart(document.getElementById('chart-daily'), {
      type: 'line',
      data: { labels: days, datasets: [{ label: 'Signals per day',
        data: counts, borderColor: '#38bdf8', backgroundColor: '#38bdf820',
        tension: 0.3, fill: true }] },
      options: { plugins: { legend: { labels: { color: '#8b8fa8' } } },
        scales: { x: { ticks: { color:'#8b8fa8' }, grid: { color:'#2a2d3a' } },
                  y: { ticks: { color:'#8b8fa8' }, grid: { color:'#2a2d3a' } } } }
    });
  } else {
    dailyChart.data.labels = days;
    dailyChart.data.datasets[0].data = counts;
    dailyChart.update();
  }

  // Per-symbol accuracy chart
  const syms = Object.keys(stats.per_symbol);
  const accs = syms.map(s => stats.per_symbol[s].accuracy);
  if (!symbolChart) {
    symbolChart = new Chart(document.getElementById('chart-symbol'), {
      type: 'bar',
      data: { labels: syms, datasets: [{ label: 'Accuracy %',
        data: accs, backgroundColor: ['#38bdf8','#4ade80','#f472b6','#fbbf24'] }] },
      options: { plugins: { legend: { labels: { color: '#8b8fa8' } } },
        scales: { x: { ticks: { color:'#8b8fa8' }, grid: { color:'#2a2d3a' } },
                  y: { min:0, max:100, ticks: { color:'#8b8fa8' }, grid: { color:'#2a2d3a' } } } }
    });
  } else {
    symbolChart.data.labels = syms;
    symbolChart.data.datasets[0].data = accs;
    symbolChart.update();
  }

  // Leaderboard
  const lb = document.getElementById('leaderboard-body');
  lb.innerHTML = (stats.top_pnl || []).map(u => {
    const avg = u.avg_pnl ? u.avg_pnl.toFixed(2) : '0.00';
    const sign = u.avg_pnl >= 0 ? '+' : '';
    const color = u.avg_pnl >= 0 ? '#4ade80' : '#f87171';
    const name = u.username ? '@' + u.username : u.chat_id;
    return `<tr><td>${name}</td><td>${u.trades}</td><td style="color:${color}">${sign}${avg}%</td></tr>`;
  }).join('') || '<tr><td colspan="3" class="text-center" style="color:#555">No closed trades yet</td></tr>';

  // Signals table
  const tbody = document.getElementById('signals-body');
  tbody.innerHTML = signals.map(s => `<tr>
    <td>${formatTime(s.sent_at)}</td>
    <td><strong>${s.symbol}</strong></td>
    <td>${badge(s.action, 'badge-' + s.action)}</td>
    <td>${badge(s.strength || 'RULE', 'badge-' + (s.strength || 'RULE'))}</td>
    <td>$${s.price ? s.price.toFixed(2) : '—'}</td>
    <td>${s.rsi ? s.rsi.toFixed(1) : '—'}</td>
    <td>${s.ai_confidence ? (s.ai_confidence*100).toFixed(0)+'%' : '—'}</td>
    <td>${outcomeBadge(s.outcome)}</td>
    <td>${s.outcome_price ? '$'+s.outcome_price.toFixed(2) : '—'}</td>
  </tr>`).join('');
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/api/signals")
def api_signals():
    return jsonify(get_recent_signals(50))


if __name__ == "__main__":
    init_db()
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "5000"))
    app.run(host=host, port=port)
