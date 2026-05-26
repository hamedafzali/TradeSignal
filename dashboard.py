import os
from flask import Flask, jsonify, render_template_string, request
from database import (
    init_db, get_stats, get_recent_signals,
    get_ml_accuracy_stats,
    get_all_symbols_with_status, add_symbol, remove_symbol,
)

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
  .badge-BUY   { background:#1a3a4a; color:#38bdf8; }
  .badge-SELL  { background:#3a1a2a; color:#f472b6; }
  .badge-STRONG{ background:#3a2a0a; color:#fbbf24; }
  .badge-AI    { background:#1a2a3a; color:#a78bfa; }
  .badge-RULE  { background:#1a2a1a; color:#4ade80; }
  .badge-active  { background:#1a4a2e; color:#4ade80; }
  .badge-inactive{ background:#4a1a1a; color:#f87171; }
  table { font-size:.85rem; }
  thead th { color:#8b8fa8; border-color:#2a2d3a !important; }
  tbody td { border-color:#2a2d3a !important; }
  .refresh-note { color:#555; font-size:.75rem; }
  .nav-tabs .nav-link { color:#8b8fa8; border-color:#2a2d3a; }
  .nav-tabs .nav-link.active { color:#e0e0e0; background:#1a1d27; border-color:#2a2d3a #2a2d3a #1a1d27; }
  .nav-tabs { border-color:#2a2d3a; }
  .progress { background:#2a2d3a; height:8px; }
  .accuracy-bar-high { background:#4ade80; }
  .accuracy-bar-mid  { background:#fbbf24; }
  .accuracy-bar-low  { background:#f87171; }
  .sym-tag { display:inline-block; padding:2px 8px; border-radius:4px;
             background:#1a2d3a; color:#38bdf8; font-size:.8rem; font-weight:600; }
  .form-control, .btn { font-size:.85rem; }
  .form-control { background:#0f1117; border-color:#2a2d3a; color:#e0e0e0; }
  .form-control:focus { background:#0f1117; border-color:#38bdf8; color:#e0e0e0; box-shadow:none; }
  .btn-add  { background:#1a4a2e; color:#4ade80; border:1px solid #4ade8040; }
  .btn-add:hover  { background:#1f5a36; color:#4ade80; }
  .btn-del  { background:#4a1a1a; color:#f87171; border:1px solid #f8717140; padding:2px 8px; font-size:.75rem; }
  .btn-del:hover  { background:#5a1f1f; color:#f87171; }
  .training-chip { font-size:.72rem; color:#8b8fa8; background:#1a1d27;
                   border:1px solid #2a2d3a; border-radius:4px; padding:1px 6px; }
</style>
</head>
<body>
<div class="container-fluid py-4 px-4">
  <div class="d-flex justify-content-between align-items-center mb-4">
    <h4 class="mb-0">📈 Trading Signal Dashboard</h4>
    <span class="refresh-note" id="last-refresh"></span>
  </div>

  <!-- Tab navigation -->
  <ul class="nav nav-tabs mb-4" id="mainTabs">
    <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-overview">Overview</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-ml">🧠 ML Accuracy</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-admin">⚙️ Admin</button></li>
  </ul>

  <div class="tab-content">

  <!-- ── OVERVIEW TAB ─────────────────────────────────────────────────── -->
  <div class="tab-pane fade show active" id="tab-overview">

    <div class="row g-3 mb-4">
      <div class="col-6 col-md-3"><div class="card p-3"><div class="card-title">Total Users</div><div class="stat-val" id="s-users">—</div></div></div>
      <div class="col-6 col-md-3"><div class="card p-3"><div class="card-title">Total Signals</div><div class="stat-val" id="s-total">—</div></div></div>
      <div class="col-6 col-md-3"><div class="card p-3"><div class="card-title">Signals Today</div><div class="stat-val" id="s-today">—</div></div></div>
      <div class="col-6 col-md-3"><div class="card p-3"><div class="card-title">Accuracy</div><div class="stat-val" id="s-acc">—</div></div></div>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-6"><div class="card p-3"><canvas id="chart-daily" height="160"></canvas></div></div>
      <div class="col-md-6"><div class="card p-3"><canvas id="chart-symbol" height="160"></canvas></div></div>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-8">
        <div class="card p-3">
          <h6 class="mb-3">Recent Signals</h6>
          <div class="table-responsive">
            <table class="table table-dark table-hover mb-0">
              <thead><tr>
                <th>Time</th><th>Symbol</th><th>Action</th><th>Strength</th>
                <th>Price</th><th>RSI</th><th>AI Conf.</th><th>Outcome</th><th>Exit Price</th>
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

  <!-- ── ML ACCURACY TAB ──────────────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-ml">

    <div class="row g-3 mb-4">
      <div class="col-12">
        <div class="card p-3">
          <h6 class="mb-3">AI Confidence vs Accuracy</h6>
          <p class="text-muted small mb-3">
            How accurate the model is at different confidence levels.
            Higher confidence should mean higher accuracy if the model is working well.
          </p>
          <canvas id="chart-buckets" height="100"></canvas>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-12">
        <div class="card p-3">
          <h6 class="mb-3">Per-Symbol ML Performance</h6>
          <div class="table-responsive">
            <table class="table table-dark table-hover mb-0" id="ml-table">
              <thead><tr>
                <th>Symbol</th>
                <th>AI Signals</th><th>AI Accuracy</th>
                <th>STRONG Signals</th><th>STRONG Accuracy</th>
                <th>Rule Only</th><th>Rule Accuracy</th>
                <th>Last Trained</th><th>Train Samples</th><th>Outcome Samples</th><th>Runs</th>
              </tr></thead>
              <tbody id="ml-body"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-3">Signal Type Breakdown</h6>
          <canvas id="chart-strength" height="200"></canvas>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-1">How Training Works</h6>
          <p class="small mt-2" style="color:#8b8fa8; line-height:1.7">
            <strong style="color:#4ade80">Every 24 hours</strong>, the model retrains on 60 days of hourly data.<br>
            <strong style="color:#fbbf24">Confirmed outcomes</strong> (signals marked correct/incorrect after 24h) are
            injected back at <strong style="color:#a78bfa">3× weight</strong> so the model learns from its real mistakes.<br>
            <strong style="color:#38bdf8">Gradient Boosting</strong> runs two classifiers — one for BUY, one for SELL —
            each outputting a probability from 0 to 1.<br>
            A signal fires when confidence exceeds <strong style="color:#fbbf24">65%</strong>.
            STRONG signals require both the rule engine and AI to agree.
          </p>
        </div>
      </div>
    </div>
  </div>

  <!-- ── ADMIN TAB ─────────────────────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-admin">

    <div class="row g-3 mb-4">
      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-3">⚙️ Watched Symbols</h6>
          <p class="small mb-3" style="color:#8b8fa8">
            Changes take effect within 5 minutes — the bot refreshes this list automatically.
          </p>

          <div class="d-flex gap-2 mb-3">
            <input type="text" id="new-symbol" class="form-control" placeholder="e.g. GOOGL or SOL-USD" style="max-width:200px">
            <button class="btn btn-add px-3" onclick="addSymbol()">+ Add</button>
          </div>
          <div id="add-msg" class="small mb-2" style="min-height:1.2em"></div>

          <div id="symbol-list"></div>
        </div>
      </div>

      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-3">📊 Symbol Signal Stats</h6>
          <div id="symbol-stats"></div>
        </div>
      </div>
    </div>
  </div>

  </div><!-- tab-content -->
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
let dailyChart, symbolChart, bucketsChart, strengthChart;

function badge(text, cls) {
  return `<span class="badge rounded-pill ${cls} px-2 py-1">${text}</span>`;
}
function outcomeBadge(o) {
  if (o==='correct')   return badge('✓ Correct',   'badge-correct');
  if (o==='incorrect') return badge('✗ Wrong',     'badge-incorrect');
  if (o==='neutral')   return badge('~ Neutral',   'badge-neutral');
  return badge('pending', 'badge-pending');
}
function formatTime(iso) {
  if (!iso) return '—';
  return new Date(iso+'Z').toLocaleString();
}
function accColor(acc) {
  if (acc === null || acc === undefined) return '#8b8fa8';
  if (acc >= 60) return '#4ade80';
  if (acc >= 50) return '#fbbf24';
  return '#f87171';
}
function accBar(acc, total) {
  if (!total) return '<span style="color:#555">no data</span>';
  const cls = acc >= 60 ? 'accuracy-bar-high' : (acc >= 50 ? 'accuracy-bar-mid' : 'accuracy-bar-low');
  return `<div class="d-flex align-items-center gap-2">
    <div class="progress flex-grow-1" style="height:6px">
      <div class="progress-bar ${cls}" style="width:${acc}%"></div>
    </div>
    <span style="color:${accColor(acc)};min-width:42px;text-align:right">${acc}%</span>
    <span style="color:#555;font-size:.75rem">(${total})</span>
  </div>`;
}

// ── Overview ──────────────────────────────────────────────────────────────────
async function refreshOverview() {
  const [stats, signals] = await Promise.all([
    fetch('/api/stats').then(r=>r.json()),
    fetch('/api/signals').then(r=>r.json()),
  ]);

  document.getElementById('s-users').textContent = stats.total_users;
  document.getElementById('s-total').textContent = stats.total_signals;
  document.getElementById('s-today').textContent = stats.today_signals;
  document.getElementById('s-acc').textContent = stats.resolved > 0 ? stats.accuracy+'%' : 'N/A';
  document.getElementById('last-refresh').textContent = 'Updated '+new Date().toLocaleTimeString();

  const days = stats.daily.map(d=>d.day.slice(5));
  const counts = stats.daily.map(d=>d.count);
  if (!dailyChart) {
    dailyChart = new Chart(document.getElementById('chart-daily'), {
      type:'line',
      data:{labels:days, datasets:[{label:'Signals per day',
        data:counts, borderColor:'#38bdf8', backgroundColor:'#38bdf820', tension:0.3, fill:true}]},
      options:{plugins:{legend:{labels:{color:'#8b8fa8'}}},
        scales:{x:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}},
                y:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}}}}
    });
  } else { dailyChart.data.labels=days; dailyChart.data.datasets[0].data=counts; dailyChart.update(); }

  const syms = Object.keys(stats.per_symbol);
  const accs = syms.map(s=>stats.per_symbol[s].accuracy);
  const colors = ['#38bdf8','#4ade80','#f472b6','#fbbf24','#a78bfa','#fb923c',
                  '#34d399','#f87171','#60a5fa','#e879f9'];
  if (!symbolChart) {
    symbolChart = new Chart(document.getElementById('chart-symbol'), {
      type:'bar',
      data:{labels:syms, datasets:[{label:'Accuracy %',
        data:accs, backgroundColor:colors.slice(0,syms.length)}]},
      options:{plugins:{legend:{labels:{color:'#8b8fa8'}}},
        scales:{x:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}},
                y:{min:0,max:100,ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}}}}
    });
  } else { symbolChart.data.labels=syms; symbolChart.data.datasets[0].data=accs; symbolChart.update(); }

  document.getElementById('leaderboard-body').innerHTML =
    (stats.top_pnl||[]).map(u => {
      const avg = u.avg_pnl?u.avg_pnl.toFixed(2):'0.00';
      const sign = u.avg_pnl>=0?'+':'';
      const color = u.avg_pnl>=0?'#4ade80':'#f87171';
      const name = u.username?'@'+u.username:u.chat_id;
      return `<tr><td>${name}</td><td>${u.trades}</td><td style="color:${color}">${sign}${avg}%</td></tr>`;
    }).join('') || '<tr><td colspan="3" class="text-center" style="color:#555">No closed trades yet</td></tr>';

  document.getElementById('signals-body').innerHTML = signals.map(s=>`<tr>
    <td>${formatTime(s.sent_at)}</td>
    <td><strong>${s.symbol}</strong></td>
    <td>${badge(s.action,'badge-'+s.action)}</td>
    <td>${badge(s.strength||'RULE','badge-'+(s.strength||'RULE'))}</td>
    <td>$${s.price?s.price.toFixed(2):'—'}</td>
    <td>${s.rsi?s.rsi.toFixed(1):'—'}</td>
    <td>${s.ai_confidence?(s.ai_confidence*100).toFixed(0)+'%':'—'}</td>
    <td>${outcomeBadge(s.outcome)}</td>
    <td>${s.outcome_price?'$'+s.outcome_price.toFixed(2):'—'}</td>
  </tr>`).join('');
}

// ── ML ────────────────────────────────────────────────────────────────────────
async function refreshML() {
  const ml = await fetch('/api/ml-stats').then(r=>r.json());

  // Confidence buckets bar chart
  const bLabels = ml.confidence_buckets.map(b=>b.label);
  const bAcc    = ml.confidence_buckets.map(b=>b.accuracy||0);
  const bTotal  = ml.confidence_buckets.map(b=>b.total);
  const bColors = bAcc.map(a => a>=60?'#4ade80':(a>=50?'#fbbf24':'#f87171'));
  if (!bucketsChart) {
    bucketsChart = new Chart(document.getElementById('chart-buckets'), {
      type:'bar',
      data:{labels:bLabels, datasets:[
        {label:'Accuracy %', data:bAcc, backgroundColor:bColors, yAxisID:'y'},
        {label:'# Signals',  data:bTotal, type:'line',
          borderColor:'#38bdf880', backgroundColor:'transparent',
          tension:0.3, yAxisID:'y2'},
      ]},
      options:{plugins:{legend:{labels:{color:'#8b8fa8'}}},
        scales:{
          x:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}},
          y:{min:0,max:100,title:{display:true,text:'Accuracy %',color:'#8b8fa8'},
             ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}},
          y2:{position:'right',title:{display:true,text:'# Signals',color:'#8b8fa8'},
              ticks:{color:'#8b8fa8'},grid:{drawOnChartArea:false}},
        }}
    });
  } else {
    bucketsChart.data.labels=bLabels;
    bucketsChart.data.datasets[0].data=bAcc;
    bucketsChart.data.datasets[0].backgroundColor=bColors;
    bucketsChart.data.datasets[1].data=bTotal;
    bucketsChart.update();
  }

  // Per-symbol table
  const rows = Object.entries(ml.per_symbol).map(([sym, d]) => {
    const tr = d.training;
    const trainedAt = tr ? new Date(tr.trained_at+'Z').toLocaleString() : '—';
    const samples   = tr ? tr.train_samples : '—';
    const outcomes  = tr ? tr.outcome_samples : '—';
    return `<tr>
      <td><span class="sym-tag">${sym}</span></td>
      <td>${d.ai_total||0}</td>
      <td>${d.ai_total ? accBar(d.ai_accuracy, d.ai_total) : '<span style="color:#555">no data</span>'}</td>
      <td>${d.strong_total||0}</td>
      <td>${d.strong_total ? accBar(d.strong_accuracy, d.strong_total) : '<span style="color:#555">no data</span>'}</td>
      <td>${d.rule_total||0}</td>
      <td>${d.rule_total ? accBar(d.rule_accuracy, d.rule_total) : '<span style="color:#555">no data</span>'}</td>
      <td style="font-size:.75rem;color:#8b8fa8">${trainedAt}</td>
      <td>${samples}</td>
      <td>${outcomes}</td>
      <td>${d.train_runs||0}</td>
    </tr>`;
  }).join('');
  document.getElementById('ml-body').innerHTML = rows ||
    '<tr><td colspan="11" class="text-center" style="color:#555">No resolved signals yet</td></tr>';

  // Strength breakdown (from stats)
  const stats = await fetch('/api/stats').then(r=>r.json());
  const bs = stats.buy_sell || {};
  const allSigs = Object.entries(stats.per_symbol||{});
  // We don't have per-strength in stats, just show buy/sell pie
  if (!strengthChart) {
    strengthChart = new Chart(document.getElementById('chart-strength'), {
      type:'doughnut',
      data:{labels:Object.keys(bs), datasets:[{
        data:Object.values(bs),
        backgroundColor:['#38bdf8','#f472b6','#4ade80'],
        borderColor:'#1a1d27', borderWidth:3,
      }]},
      options:{plugins:{legend:{labels:{color:'#8b8fa8'}}}}
    });
  } else {
    strengthChart.data.labels=Object.keys(bs);
    strengthChart.data.datasets[0].data=Object.values(bs);
    strengthChart.update();
  }
}

// ── Admin ─────────────────────────────────────────────────────────────────────
async function refreshAdmin() {
  const [syms, stats] = await Promise.all([
    fetch('/api/symbols').then(r=>r.json()),
    fetch('/api/stats').then(r=>r.json()),
  ]);

  const list = document.getElementById('symbol-list');
  list.innerHTML = syms.map(s => {
    const statusBadge = s.active
      ? '<span class="badge badge-active ms-2">Active</span>'
      : '<span class="badge badge-inactive ms-2">Inactive</span>';
    const delBtn = s.active
      ? `<button class="btn btn-del ms-auto" onclick="removeSymbol('${s.symbol}')">Remove</button>`
      : `<button class="btn btn-add ms-auto" onclick="restoreSymbol('${s.symbol}')" style="font-size:.75rem;padding:2px 8px">Restore</button>`;
    return `<div class="d-flex align-items-center mb-2 p-2" style="background:#0f1117;border-radius:6px">
      <span class="sym-tag">${s.symbol}</span>${statusBadge}
      <span class="ms-3 text-muted" style="font-size:.72rem">${s.added_by} · ${s.added_at.slice(0,10)}</span>
      ${delBtn}
    </div>`;
  }).join('') || '<div style="color:#555">No symbols configured yet.</div>';

  const statsDiv = document.getElementById('symbol-stats');
  statsDiv.innerHTML = Object.entries(stats.per_symbol||{}).map(([sym, v]) => {
    const accStr = v.resolved > 0 ? `${v.accuracy}% (${v.correct}/${v.resolved})` : 'pending';
    const accC = v.resolved > 0 ? accColor(v.accuracy) : '#555';
    return `<div class="d-flex justify-content-between align-items-center mb-2">
      <span class="sym-tag">${sym}</span>
      <span class="ms-2" style="color:#8b8fa8;font-size:.8rem">${v.total} signals</span>
      <span class="ms-auto" style="color:${accC};font-weight:600">${accStr}</span>
    </div>`;
  }).join('') || '<div style="color:#555">No signal data yet.</div>';
}

async function addSymbol() {
  const inp = document.getElementById('new-symbol');
  const sym = inp.value.trim().toUpperCase();
  if (!sym) return;
  const msg = document.getElementById('add-msg');
  const res = await fetch('/api/symbols', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({symbol:sym}),
  }).then(r=>r.json());
  if (res.ok) {
    msg.style.color='#4ade80';
    msg.textContent = `✓ ${sym} added — bot picks it up within 5 min`;
    inp.value = '';
    refreshAdmin();
  } else {
    msg.style.color='#f87171';
    msg.textContent = res.error || 'Failed';
  }
}

async function removeSymbol(sym) {
  if (!confirm(`Remove ${sym} from watched symbols?`)) return;
  const res = await fetch(`/api/symbols/${sym}`, {method:'DELETE'}).then(r=>r.json());
  if (res.ok) refreshAdmin();
}

async function restoreSymbol(sym) {
  const res = await fetch('/api/symbols', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({symbol:sym}),
  }).then(r=>r.json());
  if (res.ok) refreshAdmin();
}

// ── Init ──────────────────────────────────────────────────────────────────────
refreshOverview();
setInterval(refreshOverview, 30000);

document.querySelectorAll('[data-bs-target="#tab-ml"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshML())
);
document.querySelectorAll('[data-bs-target="#tab-admin"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshAdmin())
);
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


@app.route("/api/ml-stats")
def api_ml_stats():
    return jsonify(get_ml_accuracy_stats())


@app.route("/api/symbols", methods=["GET"])
def api_symbols_get():
    return jsonify(get_all_symbols_with_status())


@app.route("/api/symbols", methods=["POST"])
def api_symbols_add():
    data = request.get_json(force=True)
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"ok": False, "error": "Symbol required"}), 400
    added = add_symbol(symbol, added_by="dashboard")
    if added:
        return jsonify({"ok": True, "symbol": symbol})
    return jsonify({"ok": False, "error": f"{symbol} is already active"}), 409


@app.route("/api/symbols/<symbol>", methods=["DELETE"])
def api_symbols_remove(symbol):
    removed = remove_symbol(symbol.upper())
    if removed:
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Symbol not found or already inactive"}), 404


if __name__ == "__main__":
    init_db()
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "5000"))
    app.run(host=host, port=port)
