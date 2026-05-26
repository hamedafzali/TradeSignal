import os
import urllib.request
import urllib.parse
import json as _json
from flask import Flask, jsonify, render_template_string, request
from database import (
    init_db, get_stats, get_recent_signals,
    get_ml_accuracy_stats, get_ml_activity_log,
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
  /* Override Bootstrap 5 CSS variables so all components inherit dark theme */
  :root {
    --bs-body-color: #e0e0e0;
    --bs-body-bg: #0f1117;
    --bs-secondary-color: #8b8fa8;
    --bs-border-color: #2a2d3a;
    --bs-table-color: #e0e0e0;
    --bs-table-bg: transparent;
    --bs-table-border-color: #2a2d3a;
    --bs-table-striped-color: #e0e0e0;
    --bs-table-hover-color: #e0e0e0;
    --bs-card-color: #e0e0e0;
    --bs-heading-color: #e0e0e0;
    --bs-link-color: #38bdf8;
    --bs-link-hover-color: #7dd3fc;
    --bs-emphasis-color: #ffffff;
    --bs-secondary-bg: #1a1d27;
    --bs-tertiary-bg: #0f1117;
    --bs-tertiary-color: #555;
  }
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
  /* Bootstrap text utilities on dark background */
  .text-muted { color:#8b8fa8 !important; }
  .text-body  { color:#e0e0e0 !important; }
  /* Bootstrap table resets */
  .table { --bs-table-color:#e0e0e0; --bs-table-bg:transparent;
           --bs-table-border-color:#2a2d3a; color:#e0e0e0; }
  .table td, .table th { color:#e0e0e0; }
  /* Dropdowns */
  .dropdown-menu { background:#1a1d27; border-color:#2a2d3a; }
  .dropdown-item { color:#e0e0e0; }
  .dropdown-item:hover { background:#0f1117; color:#e0e0e0; }
  /* Modal */
  .modal-content { background:#1a1d27; color:#e0e0e0; border-color:#2a2d3a; }
  .modal-header, .modal-footer { border-color:#2a2d3a; }
  .health-card { background:#0f1117; border-radius:8px; padding:14px 16px; border:1px solid #2a2d3a; }
  .trend-up   { color:#4ade80 }
  .trend-down { color:#f87171 }
  .trend-flat { color:#fbbf24 }
  .event-row { border-left:3px solid #2a2d3a; padding:8px 12px; margin-bottom:8px;
               background:#0f1117; border-radius:0 6px 6px 0; font-size:.82rem; }
  .event-row.has-outcomes { border-left-color:#a78bfa; }
  .outcome-row { padding:7px 10px; margin-bottom:6px; background:#0f1117;
                 border-radius:6px; font-size:.82rem; }
  .outcome-correct   { border-left:3px solid #4ade80; }
  .outcome-incorrect { border-left:3px solid #f87171; }
  .outcome-neutral   { border-left:3px solid #fbbf24; }
  .suggestion-item { padding:10px 14px; border-radius:6px; margin-bottom:8px;
                     border:1px solid #2a2d3a; font-size:.85rem; }
  .sug-warn  { border-color:#f8717140; background:#4a1a1a22; }
  .sug-ok    { border-color:#4ade8040; background:#1a4a2e22; }
  .sug-info  { border-color:#38bdf840; background:#1a2d3a22; }
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
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-activity">🔄 Learning Activity</button></li>
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
            <strong style="color:#4ade80">Every 20 minutes</strong>, the bot checks each symbol for new resolved outcomes.<br>
            As soon as <strong style="color:#fbbf24">3 new outcomes</strong> arrive since the last training run,
            the model retrains immediately — no waiting.<br>
            As a fallback, it always retrains every <strong style="color:#38bdf8">6 hours</strong> even without new outcomes.<br>
            <strong style="color:#fbbf24">Confirmed outcomes</strong> are injected back at
            <strong style="color:#a78bfa">3× weight</strong> so the model learns from its real mistakes.<br>
            <strong style="color:#38bdf8">Gradient Boosting</strong> runs two classifiers — one for BUY, one for SELL —
            each outputting a probability from 0 to 1.<br>
            A signal fires when confidence exceeds <strong style="color:#fbbf24">65%</strong>.
            STRONG signals require both the rule engine and AI to agree.
          </p>
        </div>
      </div>
    </div>
  </div>

  <!-- ── LEARNING ACTIVITY TAB ───────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-activity">

    <!-- Model health cards per symbol -->
    <div class="row g-3 mb-4">
      <div class="col-12">
        <div class="card p-3">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h6 class="mb-0">🩺 Model Health per Symbol</h6>
            <button class="btn btn-sm" style="background:#1a2d3a;color:#38bdf8;border:1px solid #2a2d3a"
              onclick="refreshActivity()">↻ Refresh</button>
          </div>
          <div id="health-cards" class="row g-3"></div>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-4">
      <!-- Training event log -->
      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h6 class="mb-3">📋 Training Event Log</h6>
          <div id="train-log" style="max-height:420px;overflow-y:auto"></div>
        </div>
      </div>

      <!-- Recent outcome feed -->
      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h6 class="mb-3">📬 Recent Resolved Outcomes</h6>
          <div id="outcome-feed" style="max-height:420px;overflow-y:auto"></div>
        </div>
      </div>
    </div>

    <!-- Improvement suggestions -->
    <div class="row g-3 mb-4">
      <div class="col-12">
        <div class="card p-3">
          <h6 class="mb-3">💡 What Needs Attention</h6>
          <div id="suggestions"></div>
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

          <!-- Search input with autocomplete -->
          <div class="position-relative mb-3" style="max-width:340px">
            <div class="d-flex gap-2">
              <input type="text" id="new-symbol" class="form-control"
                placeholder="Search by name or ticker…"
                autocomplete="off"
                oninput="searchSymbol(this.value)"
                onkeydown="if(event.key==='Enter'){addSymbol();event.preventDefault()}">
              <button class="btn btn-add px-3" onclick="addSymbol()">+ Add</button>
            </div>
            <div id="search-dropdown" class="position-absolute w-100 mt-1"
              style="z-index:1000;display:none;background:#1a1d27;border:1px solid #2a2d3a;
                     border-radius:6px;max-height:220px;overflow-y:auto"></div>
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
// Global Chart.js dark theme defaults
Chart.defaults.color = '#8b8fa8';
Chart.defaults.plugins.tooltip.backgroundColor = '#1a1d27';
Chart.defaults.plugins.tooltip.titleColor = '#e0e0e0';
Chart.defaults.plugins.tooltip.bodyColor = '#8b8fa8';
Chart.defaults.plugins.tooltip.borderColor = '#2a2d3a';
Chart.defaults.plugins.tooltip.borderWidth = 1;

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

let _searchTimer = null;
function searchSymbol(q) {
  clearTimeout(_searchTimer);
  const dd = document.getElementById('search-dropdown');
  if (!q || q.length < 1) { dd.style.display='none'; return; }
  _searchTimer = setTimeout(async () => {
    const res = await fetch('/api/search?q='+encodeURIComponent(q)).then(r=>r.json());
    if (!res.length) { dd.style.display='none'; return; }
    dd.innerHTML = res.map(item => `
      <div class="d-flex align-items-center gap-2 px-3 py-2 search-row"
        style="cursor:pointer;border-bottom:1px solid #2a2d3a"
        onmouseover="this.style.background='#0f1117'"
        onmouseout="this.style.background=''"
        onclick="selectSymbol('${item.symbol}')">
        <span class="sym-tag" style="min-width:70px">${item.symbol}</span>
        <span style="color:#e0e0e0;font-size:.82rem">${item.name}</span>
        <span class="ms-auto" style="color:#555;font-size:.72rem">${item.exchange} · ${item.type}</span>
      </div>`).join('');
    dd.style.display='block';
  }, 300);
}

function selectSymbol(sym) {
  document.getElementById('new-symbol').value = sym;
  document.getElementById('search-dropdown').style.display='none';
}

document.addEventListener('click', e => {
  if (!e.target.closest('#new-symbol') && !e.target.closest('#search-dropdown'))
    document.getElementById('search-dropdown').style.display = 'none';
});

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

// ── Learning Activity ─────────────────────────────────────────────────────────
function trendIcon(t) {
  if (t === 'improving') return '<span class="trend-up">↑ Improving</span>';
  if (t === 'declining') return '<span class="trend-down">↓ Declining</span>';
  return '<span class="trend-flat">→ Stable</span>';
}
function modelAge(ts) {
  if (!ts) return 'never trained';
  const mins = Math.round((Date.now() - new Date(ts+'Z')) / 60000);
  if (mins < 60) return `${mins}m ago`;
  if (mins < 1440) return `${Math.round(mins/60)}h ago`;
  return `${Math.round(mins/1440)}d ago`;
}

async function refreshActivity() {
  const data = await fetch('/api/ml-activity').then(r=>r.json());

  // ── Model health cards ──
  const health = data.symbol_health || {};
  document.getElementById('health-cards').innerHTML =
    Object.entries(health).map(([sym, h]) => {
      const accColor = h.recent_acc >= 60 ? '#4ade80' : (h.recent_acc >= 50 ? '#fbbf24' : '#f87171');
      const barW = Math.min(h.recent_acc, 100);
      const barCls = h.recent_acc >= 60 ? 'accuracy-bar-high' : (h.recent_acc >= 50 ? 'accuracy-bar-mid' : 'accuracy-bar-low');
      return `<div class="col-6 col-md-4 col-lg-3">
        <div class="health-card">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <span class="sym-tag">${sym}</span>
            ${trendIcon(h.trend)}
          </div>
          <div class="progress mb-1" style="height:5px">
            <div class="progress-bar ${barCls}" style="width:${barW}%"></div>
          </div>
          <div class="d-flex justify-content-between" style="font-size:.75rem;color:#8b8fa8">
            <span>Recent acc: <strong style="color:${accColor}">${h.recent_acc}%</strong> (${h.recent_count})</span>
            <span>All: ${h.all_acc}%</span>
          </div>
          <div class="mt-2" style="font-size:.72rem;color:#555">
            <span>📊 ${h.total_outcomes} outcomes · ⏳ ${h.pending} pending</span><br>
            <span>🕒 Trained ${modelAge(h.last_trained)}</span>
          </div>
        </div>
      </div>`;
    }).join('') || '<div class="col-12" style="color:#555">No training data yet — models train automatically once signals accumulate outcomes.</div>';

  // ── Training event log ──
  const events = data.training_events || [];
  document.getElementById('train-log').innerHTML = events.length
    ? events.map(e => {
        const cls = e.outcome_samples > 0 ? 'has-outcomes' : '';
        const trigger = e.outcome_samples >= 3
          ? `<span style="color:#a78bfa">⚡ ${e.outcome_samples} outcomes triggered</span>`
          : `<span style="color:#38bdf8">⏱ time-based</span>`;
        return `<div class="event-row ${cls}">
          <div class="d-flex justify-content-between">
            <strong style="color:#e0e0e0">${e.symbol}</strong>
            <span style="color:#555">${new Date(e.trained_at+'Z').toLocaleString()}</span>
          </div>
          <div class="mt-1">${trigger} &nbsp;·&nbsp;
            <span style="color:#8b8fa8">${e.train_samples} samples trained</span>
            ${e.outcome_samples > 0 ? `&nbsp;·&nbsp;<span style="color:#fbbf24">${e.outcome_samples} real outcomes blended (3× weight)</span>` : ''}
          </div>
        </div>`;
      }).join('')
    : '<div style="color:#555;font-size:.85rem">No training runs yet. The model will train automatically on startup and then every time 3 new outcomes resolve.</div>';

  // ── Outcome feed ──
  const outcomes = data.recent_outcomes || [];
  document.getElementById('outcome-feed').innerHTML = outcomes.length
    ? outcomes.map(o => {
        const cls = `outcome-${o.outcome}`;
        const emoji = o.outcome === 'correct' ? '✅' : o.outcome === 'incorrect' ? '❌' : '➖';
        const conf = o.ai_confidence ? ` · AI ${(o.ai_confidence*100).toFixed(0)}%` : '';
        const when = o.outcome_at ? new Date(o.outcome_at+'Z').toLocaleString() : '—';
        return `<div class="outcome-row ${cls}">
          <div class="d-flex justify-content-between">
            <span><strong>${o.symbol}</strong> ${badge(o.action,'badge-'+o.action)} ${badge(o.strength||'RULE','badge-'+(o.strength||'RULE'))}</span>
            <span style="color:#555;font-size:.72rem">${when}</span>
          </div>
          <div style="color:#8b8fa8;margin-top:3px">
            ${emoji} <strong style="color:#e0e0e0">${o.outcome.toUpperCase()}</strong>
            · Entry $${o.price ? o.price.toFixed(2) : '—'}${conf}
          </div>
        </div>`;
      }).join('')
    : '<div style="color:#555;font-size:.85rem">No resolved outcomes yet. Outcomes are checked every 30 minutes — signals need at least 1 hour to resolve.</div>';

  // ── Suggestions ──
  const sugs = [];
  for (const [sym, h] of Object.entries(health)) {
    if (h.total_outcomes === 0) {
      sugs.push({cls:'sug-info', icon:'ℹ️',
        text:`<strong>${sym}</strong>: no outcomes resolved yet — model is running on market data only, no real feedback loop active`});
    } else if (h.recent_acc < 45 && h.recent_count >= 5) {
      sugs.push({cls:'sug-warn', icon:'⚠️',
        text:`<strong>${sym}</strong>: recent accuracy ${h.recent_acc}% is below 45% on last ${h.recent_count} outcomes — model may be overfitting or market conditions changed`});
    } else if (h.trend === 'declining') {
      sugs.push({cls:'sug-warn', icon:'📉',
        text:`<strong>${sym}</strong>: accuracy declining (recent ${h.recent_acc}% vs all-time ${h.all_acc}%) — more outcomes are needed to retrain`});
    } else if (h.trend === 'improving') {
      sugs.push({cls:'sug-ok', icon:'📈',
        text:`<strong>${sym}</strong>: accuracy improving! Recent ${h.recent_acc}% vs all-time ${h.all_acc}% — continuous learning is working`});
    } else if (h.pending > 10) {
      sugs.push({cls:'sug-info', icon:'⏳',
        text:`<strong>${sym}</strong>: ${h.pending} signals still pending outcome — check outcome interval or if TP/SL were set on signals`});
    }
  }
  if (!Object.keys(health).length) {
    sugs.push({cls:'sug-info', icon:'🚀',
      text:'No models trained yet. Send a few signals first — outcomes resolve after 1h and training starts automatically.'});
  }
  document.getElementById('suggestions').innerHTML = sugs.length
    ? sugs.map(s => `<div class="suggestion-item ${s.cls}">${s.icon} ${s.text}</div>`).join('')
    : '<div class="suggestion-item sug-ok">✅ All models look healthy — no issues detected.</div>';
}

// ── Init ──────────────────────────────────────────────────────────────────────
refreshOverview();
setInterval(refreshOverview, 30000);

document.querySelectorAll('[data-bs-target="#tab-ml"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshML())
);
document.querySelectorAll('[data-bs-target="#tab-activity"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshActivity())
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


@app.route("/api/ml-activity")
def api_ml_activity():
    return jsonify(get_ml_activity_log())


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    try:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/search"
            f"?q={urllib.parse.quote(q)}&quotesCount=10&newsCount=0"
            "&enableFuzzyQuery=false&quotesQueryId=tss_match_phrase_query"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read())
        results = []
        for item in data.get("quotes", []):
            symbol = item.get("symbol", "")
            if not symbol:
                continue
            results.append({
                "symbol": symbol,
                "name": item.get("shortname") or item.get("longname") or symbol,
                "exchange": item.get("exchDisp") or item.get("exchange") or "—",
                "type": item.get("typeDisp") or "Equity",
            })
        return jsonify(results)
    except Exception as e:
        return jsonify([])


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
