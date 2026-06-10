import os
import urllib.request
import urllib.parse
import json as _json
from flask import Flask, jsonify, render_template_string, request
from database import (
    init_db, get_stats, get_recent_signals,
    get_ml_accuracy_stats, get_ml_activity_log,
    get_all_symbols_with_status, add_symbol, remove_symbol, delete_symbol,
    create_action_request, get_ops_snapshot,
    get_running_training_job, get_recent_training_jobs,
    get_all_settings, set_setting, get_all_sentiment_cache,
    set_symbol_market, get_accuracy_by_direction, get_wf_results,
    get_rolling_performance, get_equity_curve, get_regime_attribution,
    get_recent_scans, get_last_scan, get_pipeline_trace, get_event_timeline,
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
  :root {
    --bg:       #0d0f18;
    --surface:  #131620;
    --surface2: #191d2c;
    --border:   #1e2232;
    --border2:  #262a3c;
    --text:     #dde1f0;
    --muted:    #5a6080;
    --dim:      #363a52;
    --accent:   #38bdf8;
    --green:    #22c55e;
    --red:      #ef4444;
    --yellow:   #eab308;
    --purple:   #a78bfa;
    --pink:     #f472b6;
    /* Bootstrap overrides */
    --bs-body-color: var(--text);
    --bs-body-bg: var(--bg);
    --bs-border-color: var(--border);
    --bs-secondary-bg: var(--surface);
    --bs-tertiary-bg: var(--bg);
    --bs-link-color: var(--accent);
    --bs-link-hover-color: #7dd3fc;
    --bs-emphasis-color: #fff;
  }

  /* ── Base ── */
  body { background:var(--bg); color:var(--text); font-size:.84rem; }
  ::-webkit-scrollbar { width:5px; height:5px; }
  ::-webkit-scrollbar-track { background:var(--bg); }
  ::-webkit-scrollbar-thumb { background:var(--border2); border-radius:3px; }

  /* ── Cards ── */
  .card { background:var(--surface); border:1px solid var(--border); border-radius:12px; }
  .card-title { color:var(--muted); font-size:.66rem; text-transform:uppercase; letter-spacing:.1em; font-weight:600; }
  .section-head { font-size:.82rem; font-weight:600; color:var(--text); margin-bottom:10px; }

  /* ── Stat cards ── */
  .stat-val { font-size:1.5rem; font-weight:700; line-height:1.1; }

  /* ── Ops cards ── */
  .ops-card { background:var(--bg); border:1px solid var(--border); border-radius:10px; padding:10px 14px; min-height:66px; }
  .ops-label { color:var(--muted); font-size:.64rem; text-transform:uppercase; letter-spacing:.09em; font-weight:600; }
  .ops-value { color:#fff; font-size:.92rem; font-weight:700; margin-top:3px; }
  .ops-meta { color:var(--dim); font-size:.66rem; margin-top:3px; line-height:1.4; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

  /* ── Status dots ── */
  .status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; flex-shrink:0; }
  .status-ok   { background:var(--green); box-shadow:0 0 5px #22c55e70; }
  .status-warn { background:var(--yellow); }
  .status-bad  { background:var(--red); }

  /* ── Badges ── */
  .badge-correct   { background:#0d2e1a; color:#4ade80; }
  .badge-incorrect { background:#2e0d0d; color:#f87171; }
  .badge-pending   { background:var(--border); color:var(--muted); }
  .badge-neutral   { background:#2e200a; color:#fbbf24; }
  .badge-BUY    { background:#082233; color:var(--accent); }
  .badge-SELL   { background:#330820; color:var(--pink); }
  .badge-STRONG { background:#33260a; color:var(--yellow); }
  .badge-AI     { background:#150e30; color:var(--purple); }
  .badge-RULE   { background:#082211; color:var(--green); }
  .badge-active   { background:#0d2e1a; color:#4ade80; }
  .badge-inactive { background:#2e0d0d; color:#f87171; }

  /* ── Sym tag ── */
  .sym-tag { display:inline-block; padding:2px 7px; border-radius:5px; background:#09202e; color:var(--accent); font-size:.74rem; font-weight:700; letter-spacing:.02em; }

  /* ── Tables ── */
  table { font-size:.78rem; }
  thead th { color:var(--muted); border-color:var(--border) !important; font-size:.66rem; text-transform:uppercase; letter-spacing:.05em; font-weight:600; padding:7px 10px; white-space:nowrap; }
  tbody td { border-color:var(--border) !important; vertical-align:middle; padding:7px 10px; }
  .table { --bs-table-color:var(--text); --bs-table-bg:transparent; --bs-table-border-color:var(--border); color:var(--text); }
  .table td, .table th { color:var(--text); }
  .table-hover tbody tr:hover td { background:#ffffff05 !important; }

  /* ── Progress ── */
  .progress { background:var(--border); border-radius:3px; }
  .accuracy-bar-high { background:var(--green); }
  .accuracy-bar-mid  { background:var(--yellow); }
  .accuracy-bar-low  { background:var(--red); }

  /* ── Navigation ── */
  .refresh-note { color:var(--muted); font-size:.70rem; }
  .nav-tabs { border-bottom:1px solid var(--border); }
  .nav-tabs .nav-link { color:var(--muted); border:none; border-bottom:2px solid transparent; padding:8px 16px; font-size:.82rem; background:transparent; border-radius:0; transition:color .15s; font-weight:500; }
  .nav-tabs .nav-link:hover { color:var(--text); border-bottom-color:var(--border2); background:transparent; }
  .nav-tabs .nav-link.active { color:var(--text); background:transparent; border-bottom:2px solid var(--accent); }

  /* ── Forms & Buttons ── */
  .form-control { background:var(--bg); border-color:var(--border); color:var(--text); font-size:.80rem; border-radius:8px; }
  .form-control:focus { background:var(--bg); border-color:var(--accent); color:var(--text); box-shadow:0 0 0 2px #38bdf815; }
  .btn { font-size:.80rem; border-radius:8px; }
  .btn-add  { background:#0a2214; color:var(--green); border:1px solid #22c55e30; }
  .btn-add:hover  { background:#0e2c1a; color:var(--green); }
  .btn-del  { background:#2a0a0a; color:#f87171; border:1px solid #f8717130; padding:3px 8px; font-size:.70rem; border-radius:6px; }
  .btn-del:hover  { background:#381010; color:#f87171; }
  .action-btn { width:100%; text-align:left; background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:7px 12px; transition:all .15s; }
  .action-btn:hover { background:var(--surface2); color:#fff; border-color:var(--accent); }

  /* ── Bootstrap text overrides ── */
  .text-muted { color:var(--muted) !important; }
  .text-body  { color:var(--text) !important; }
  .dropdown-menu { background:var(--surface); border-color:var(--border); }
  .dropdown-item { color:var(--text); font-size:.80rem; }
  .dropdown-item:hover { background:var(--bg); color:var(--text); }
  .modal-content { background:var(--surface); color:var(--text); border-color:var(--border); }
  .modal-header, .modal-footer { border-color:var(--border); }

  /* ── Misc components ── */
  .training-chip { font-size:.68rem; color:var(--muted); background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:1px 6px; }
  .health-card { background:var(--bg); border-radius:8px; padding:10px 12px; border:1px solid var(--border); }
  .trend-up   { color:var(--green); }
  .trend-down { color:var(--red); }
  .trend-flat { color:var(--yellow); }

  .event-row { border-left:3px solid var(--border); padding:7px 12px; margin-bottom:6px; background:var(--bg); border-radius:0 8px 8px 0; font-size:.76rem; }
  .event-row.has-outcomes { border-left-color:var(--purple); }
  .outcome-row { padding:7px 10px; margin-bottom:5px; background:var(--bg); border-radius:8px; font-size:.76rem; }
  .outcome-correct   { border-left:3px solid var(--green); }
  .outcome-incorrect { border-left:3px solid var(--red); }
  .outcome-neutral   { border-left:3px solid var(--yellow); }

  .suggestion-item { padding:9px 13px; border-radius:8px; margin-bottom:6px; border:1px solid var(--border); font-size:.80rem; }
  .sug-warn  { border-color:#ef444435; background:#1a0a0a; }
  .sug-ok    { border-color:#22c55e35; background:#0a1a0e; }
  .sug-info  { border-color:#38bdf835; background:#0a141e; }

  /* ── Sub-tabs (Learning) ── */
  .sub-tabs { border-bottom: 1px solid var(--border); margin-bottom: 12px; }
  .sub-tabs .nav-item { margin-bottom: -1px; }
  .sub-tabs .nav-link { color: var(--muted); font-size: .76rem; padding: 5px 14px; border-radius: 6px 6px 0 0; border: 1px solid transparent; background: transparent; }
  .sub-tabs .nav-link.active { color: var(--text); background: var(--surface); border-color: var(--border) var(--border) var(--surface); }
  .sub-tabs .nav-link:hover { color: var(--text); background: var(--surface2); border-color: var(--border); }

  /* ── Theme toggle button ── */
  #theme-toggle { background:none; border:1px solid var(--border); color:var(--muted); border-radius:7px; padding:3px 9px; font-size:.8rem; cursor:pointer; transition:all .15s; line-height:1.4; }
  #theme-toggle:hover { border-color:var(--accent); color:var(--text); }

  /* ── Light theme overrides ── */
  [data-theme="light"] {
    --bg:       #f0f3fa;
    --surface:  #ffffff;
    --surface2: #eaeff8;
    --border:   #dce2f0;
    --border2:  #c8d0e8;
    --text:     #1a1d2e;
    --muted:    #64748b;
    --dim:      #94a3b8;
    --accent:   #0284c7;
    --green:    #16a34a;
    --red:      #dc2626;
    --yellow:   #b45309;
    --purple:   #7c3aed;
    --pink:     #db2777;
  }
  [data-theme="light"] body { background:var(--bg); color:var(--text); }
  [data-theme="light"] .card { background:var(--surface); }
  [data-theme="light"] .ops-card { background:var(--surface2); }
  [data-theme="light"] .health-card { background:var(--surface2); }
  [data-theme="light"] .event-row { background:var(--surface2); }
  [data-theme="light"] .outcome-row { background:var(--surface2); }
  [data-theme="light"] .form-control { background:var(--surface); }
  [data-theme="light"] .form-control:focus { background:var(--surface); }
  [data-theme="light"] .action-btn { background:var(--surface); }
  [data-theme="light"] .action-btn:hover { background:var(--surface2); color:var(--text); }
  [data-theme="light"] .dropdown-menu { background:var(--surface); }
  [data-theme="light"] .dropdown-item:hover { background:var(--surface2); }
  [data-theme="light"] .modal-content { background:var(--surface); }
  [data-theme="light"] .sug-warn { background:#fff5f5; }
  [data-theme="light"] .sug-ok   { background:#f0fff4; }
  [data-theme="light"] .sug-info { background:#f0f9ff; }
  [data-theme="light"] .badge-correct   { background:#dcfce7; color:#15803d; }
  [data-theme="light"] .badge-incorrect { background:#fee2e2; color:#b91c1c; }
  [data-theme="light"] .badge-pending   { background:var(--border); color:var(--muted); }
  [data-theme="light"] .badge-neutral   { background:#fef9c3; color:#92400e; }
  [data-theme="light"] .badge-BUY    { background:#e0f2fe; color:#0369a1; }
  [data-theme="light"] .badge-SELL   { background:#fce7f3; color:#9d174d; }
  [data-theme="light"] .badge-STRONG { background:#fef3c7; color:#92400e; }
  [data-theme="light"] .badge-AI     { background:#ede9fe; color:#5b21b6; }
  [data-theme="light"] .badge-RULE   { background:#dcfce7; color:#15803d; }
  [data-theme="light"] .badge-active   { background:#dcfce7; color:#15803d; }
  [data-theme="light"] .badge-inactive { background:#fee2e2; color:#b91c1c; }
  [data-theme="light"] .sym-tag { background:#e0f2fe; color:#0369a1; }
  [data-theme="light"] .table { --bs-table-color:var(--text); --bs-table-bg:transparent; --bs-table-border-color:var(--border); color:var(--text); }
  [data-theme="light"] .table td, [data-theme="light"] .table th { color:var(--text); border-color:var(--border) !important; }
  [data-theme="light"] .table-hover tbody tr:hover td { background:var(--surface2) !important; }
  [data-theme="light"] .progress { background:var(--border2); }
  [data-theme="light"] ::-webkit-scrollbar-thumb { background:var(--border2); }
</style>
<script>
  (function(){
    var t = localStorage.getItem('tsTheme') || 'dark';
    document.documentElement.setAttribute('data-theme', t);
  })();
</script>
</head>
<body>
<div class="container-fluid py-2 px-3">
  <div class="d-flex justify-content-between align-items-center mb-2">
    <span style="font-size:.95rem;font-weight:700;color:var(--text);letter-spacing:.01em">Trade Signals</span>
    <div class="d-flex align-items-center gap-3">
      <span class="refresh-note" id="last-refresh"></span>
      <button id="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">☀️</button>
    </div>
  </div>

  <!-- Tab navigation -->
  <ul class="nav nav-tabs mb-3" id="mainTabs">
    <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-command">Command</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-overview">Overview</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-signals">Signals</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-ml">ML</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-activity">Learning</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-performance">Performance</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-timeline">Timeline</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-admin">Admin</button></li>
  </ul>

  <div class="tab-content">

  <!-- ── COMMAND CENTER TAB ──────────────────────────────────────────── -->
  <div class="tab-pane fade show active" id="tab-command">

    <!-- System health row -->
    <div class="row g-2 mb-2">
      <div class="col-6 col-md-3">
        <div class="ops-card">
          <div class="ops-label">Dashboard</div>
          <div class="ops-value"><span class="status-dot status-ok"></span>Running</div>
          <div class="ops-meta" id="cc-uptime">port 5002</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="ops-card">
          <div class="ops-label">Bot Status</div>
          <div class="ops-value" id="cc-bot-status"><span class="status-dot status-warn"></span>Checking…</div>
          <div class="ops-meta" id="cc-bot-meta">last activity</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="ops-card">
          <div class="ops-label">FinBERT / Sentiment</div>
          <div class="ops-value" id="cc-finbert-status"><span class="status-dot status-warn"></span>Checking…</div>
          <div class="ops-meta" id="cc-finbert-meta">local AI service</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="ops-card">
          <div class="ops-label">Perf Brake</div>
          <div class="ops-value" id="cc-brake-status">—</div>
          <div class="ops-meta" id="cc-brake-meta">20-trade win rate</div>
        </div>
      </div>
    </div>

    <!-- Last scan summary -->
    <div class="card mb-2 p-3">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <div class="section-head mb-0">Last Scan</div>
        <button class="btn btn-sm" style="background:var(--surface2);color:var(--accent);border:1px solid var(--border)" onclick="refreshCommandCenter()">↻ Refresh</button>
      </div>
      <div class="row g-2" id="cc-scan-summary">
        <div class="col-6 col-md-2"><div class="ops-card"><div class="ops-label">Scanned At</div><div class="ops-value" id="cc-scan-at">—</div></div></div>
        <div class="col-6 col-md-2"><div class="ops-card"><div class="ops-label">Total Symbols</div><div class="ops-value" id="cc-sym-total">—</div></div></div>
        <div class="col-6 col-md-2"><div class="ops-card"><div class="ops-label">Evaluated</div><div class="ops-value" id="cc-sym-scanned" style="color:var(--green)">—</div></div></div>
        <div class="col-6 col-md-2"><div class="ops-card"><div class="ops-label">Skipped (Market)</div><div class="ops-value" id="cc-sym-skipped-mkt" style="color:var(--muted)">—</div></div></div>
        <div class="col-6 col-md-2"><div class="ops-card"><div class="ops-label">Signals Sent</div><div class="ops-value" id="cc-signals-gen" style="color:var(--accent)">—</div></div></div>
        <div class="col-6 col-md-2"><div class="ops-card"><div class="ops-label">Duration</div><div class="ops-value" id="cc-scan-dur">—</div></div></div>
      </div>
    </div>

    <!-- Scan history chart + table -->
    <div class="row g-2 mb-2">
      <div class="col-md-8">
        <div class="card p-3">
          <div class="section-head mb-2">Scan History — Signals per Run</div>
          <canvas id="chart-scan-history" height="100"></canvas>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card p-3 h-100">
          <div class="section-head mb-2">Active Training Job</div>
          <div id="cc-training-job" style="font-size:.82rem;color:var(--muted)">No training running.</div>
        </div>
      </div>
    </div>

    <!-- Recent scan log table -->
    <div class="card p-3">
      <div class="section-head mb-2">Recent Scan Log</div>
      <div class="table-responsive" style="max-height:300px;overflow-y:auto">
        <table class="table table-hover mb-0" style="font-size:.76rem">
          <thead><tr>
            <th>Time (CET)</th><th>Symbols</th><th>Evaluated</th>
            <th>Skipped Market</th><th>Signals</th><th>Duration</th><th>Regime</th>
          </tr></thead>
          <tbody id="cc-scan-log-body"></tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- ── OVERVIEW TAB ─────────────────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-overview">

    <!-- Status strip -->
    <div class="card mb-2 px-3 py-2">
      <div class="d-flex flex-wrap align-items-center gap-3" style="font-size:.76rem">
        <span id="ops-signal-status" style="color:var(--muted)">—</span>
        <span style="color:var(--dim)">·</span>
        <span id="ops-learning-status" style="color:var(--muted)">—</span>
        <span style="color:var(--dim)">·</span>
        <span style="color:var(--muted)">Pending: <span id="ops-pending-work" style="color:var(--text)">—</span></span>
        <span style="color:var(--dim)">·</span>
        <span style="color:var(--muted)">Symbols: <span id="ops-coverage" style="color:var(--text)">—</span></span>
        <span class="ms-auto d-flex gap-3 align-items-center flex-wrap">
          <span id="market-us"></span>
          <span id="market-xetra"></span>
          <span id="market-crypto"></span>
          <span style="color:var(--dim);font-size:.68rem" id="market-times"></span>
        </span>
      </div>
    </div>

    <!-- Key metrics -->
    <div class="row g-2 mb-2">
      <div class="col-6 col-md-3"><div class="card p-2 px-3"><div class="card-title mb-1">Total Signals</div><div class="stat-val" id="s-total">—</div></div></div>
      <div class="col-6 col-md-3"><div class="card p-2 px-3"><div class="card-title mb-1">Today</div><div class="stat-val" id="s-today">—</div></div></div>
      <div class="col-6 col-md-3"><div class="card p-2 px-3"><div class="card-title mb-1">Accuracy</div><div class="stat-val" id="s-acc">—</div></div></div>
      <div class="col-6 col-md-3"><div class="card p-2 px-3"><div class="card-title mb-1">Users</div><div class="stat-val" id="s-users">—</div></div></div>
    </div>
    <!-- Regime badge + attribution -->
    <div class="card mb-2 p-2 px-3 d-flex flex-row align-items-center gap-3 flex-wrap" style="font-size:.8rem">
      <span style="color:var(--muted)">Market Regime:</span>
      <span id="regime-badge" class="badge fs-6 px-3 py-1">—</span>
      <span style="color:var(--dim)">·</span>
      <span id="regime-attribution" style="color:var(--muted)"></span>
    </div>

    <div class="row g-2 mb-2">
      <div class="col-md-6"><div class="card p-3"><canvas id="chart-daily" height="100"></canvas></div></div>
      <div class="col-md-6"><div class="card p-3"><canvas id="chart-symbol" height="100"></canvas></div></div>
    </div>

    <div class="row g-2 mb-2">
      <div class="col-12">
        <div class="card p-3">
          <div class="section-head mb-2">Recent Signals</div>
          <div class="table-responsive">
            <table class="table table-hover mb-0">
              <thead><tr>
                <th>Time</th><th>Symbol</th><th>Action</th><th>Type</th>
                <th>Price</th><th>AI%</th><th>Outcome</th>
              </tr></thead>
              <tbody id="signals-body"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── SIGNALS TAB ──────────────────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-signals">

    <div class="d-flex justify-content-between align-items-center mb-2 flex-wrap gap-2">
      <div class="d-flex gap-2 flex-wrap">
        <select id="sig-filter-action" class="form-control" style="max-width:120px" onchange="refreshSignalsTab()">
          <option value="">All Actions</option>
          <option value="BUY">BUY only</option>
          <option value="SELL">SELL only</option>
        </select>
        <select id="sig-filter-outcome" class="form-control" style="max-width:130px" onchange="refreshSignalsTab()">
          <option value="">All Outcomes</option>
          <option value="pending">Pending</option>
          <option value="correct">Correct</option>
          <option value="incorrect">Incorrect</option>
          <option value="neutral">Neutral</option>
        </select>
        <select id="sig-filter-strength" class="form-control" style="max-width:130px" onchange="refreshSignalsTab()">
          <option value="">All Types</option>
          <option value="STRONG">STRONG</option>
          <option value="AI">AI</option>
          <option value="RULE">RULE</option>
        </select>
      </div>
      <button class="btn btn-sm" style="background:var(--surface2);color:var(--accent);border:1px solid var(--border)" onclick="refreshSignalsTab()">↻ Refresh</button>
    </div>

    <div id="signals-explainability-list"></div>

  </div>

  <!-- ── ML ACCURACY TAB ──────────────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-ml">

    <div class="row g-2 mb-2">
      <div class="col-6 col-md-3"><div class="ops-card"><div class="ops-label">Active Models</div><div class="ops-value" id="ml-active-models">—</div><div class="ops-meta">In learning universe</div></div></div>
      <div class="col-6 col-md-3"><div class="ops-card"><div class="ops-label">Trained</div><div class="ops-value" id="ml-trained-models">—</div><div class="ops-meta">Successful train run</div></div></div>
      <div class="col-6 col-md-3"><div class="ops-card"><div class="ops-label">AI Resolved</div><div class="ops-value" id="ml-resolved-ai">—</div><div class="ops-meta">With AI confidence</div></div></div>
      <div class="col-6 col-md-3"><div class="ops-card"><div class="ops-label">Avg Accuracy</div><div class="ops-value" id="ml-avg-acc">—</div><div class="ops-meta">Cross-symbol mean</div></div></div>
    </div>

    <div class="row g-2 mb-2">
      <div class="col-12">
        <div class="card p-3">
          <div class="d-flex justify-content-between align-items-center mb-1">
            <div class="section-head mb-0">AI Confidence vs Accuracy</div>
            <span style="color:var(--muted);font-size:.72rem">Higher confidence should correlate with higher accuracy</span>
          </div>
          <canvas id="chart-buckets" height="75"></canvas>
        </div>
      </div>
    </div>

    <div class="row g-2 mb-2">
      <div class="col-12">
        <div class="card p-3">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <div class="section-head mb-0">Per-Symbol ML Performance</div>
            <span id="ml-table-note" style="font-size:.70rem;color:var(--muted)"></span>
          </div>
          <div class="table-responsive">
            <table class="table table-hover mb-0" id="ml-table">
              <thead><tr>
                <th>Symbol</th>
                <th title="Historical win rate from bootstrap backtest">Bootstrap WR</th>
                <th title="AI-filtered signals and their live accuracy">AI Signals</th>
                <th title="Rule+AI agreement signals">STRONG</th>
                <th title="Rule engine only signals">Rule-Only</th>
                <th title="Last training run details">Training</th>
              </tr></thead>
              <tbody id="ml-body"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Direction accuracy diagnostic -->
    <div class="row g-2 mb-2">
      <div class="col-12">
        <div class="card p-3">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <div class="section-head mb-0">🧭 Direction Accuracy — BUY vs SELL</div>
            <span style="color:var(--muted);font-size:.72rem">Identifies directional bias. Needs ≥3 resolved signals per direction to be meaningful.</span>
          </div>
          <div class="table-responsive">
            <table class="table table-hover mb-0" id="direction-table">
              <thead><tr>
                <th>Symbol</th>
                <th>BUY accuracy</th>
                <th>SELL accuracy</th>
                <th>Diagnosis</th>
              </tr></thead>
              <tbody id="direction-body"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Walk-forward validation results -->
    <div class="row g-2 mb-2">
      <div class="col-12">
        <div class="card p-3">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <div class="section-head mb-0">🧪 Walk-Forward Validation — Out-of-Sample Grades</div>
            <span style="color:var(--muted);font-size:.72rem">A≥58% · B 52-58% · C 47-52% · D&lt;47% · Trained on 75% of history, tested on last 25%</span>
          </div>
          <div id="wf-empty" class="text-muted small" style="display:none">No walk-forward results yet. Use the Action Center to run a test.</div>
          <div class="table-responsive">
            <table class="table table-hover mb-0" id="wf-table">
              <thead><tr>
                <th>Symbol</th>
                <th>Grade</th>
                <th>Rule Signals</th>
                <th>Rule Win%</th>
                <th>AI Signals</th>
                <th>AI Win%</th>
                <th>Test Bars</th>
                <th>Tested</th>
              </tr></thead>
              <tbody id="wf-body"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div class="row g-2 mb-2">
      <div class="col-md-6">
        <div class="card p-3">
          <div class="section-head">Signal Type Breakdown</div>
          <div style="position:relative;height:150px"><canvas id="chart-strength"></canvas></div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card p-3">
          <div class="section-head">Training Runs by Symbol</div>
          <canvas id="chart-train-runs" height="130"></canvas>
        </div>
      </div>
    </div>

  </div>

  <!-- ── LEARNING ACTIVITY TAB ───────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-activity">

    <!-- Always-visible: Action Center + Snapshot -->
    <div class="row g-2 mb-2">
      <div class="col-md-5">
        <div class="card p-3 h-100">
          <div class="section-head">🎛 Action Center</div>
          <div class="row g-2">
            <div class="col-6"><button class="btn action-btn" onclick="runAction('scan_now')">Scan Market Now</button></div>
            <div class="col-6"><button class="btn action-btn" onclick="runAction('check_outcomes_now')">Check Outcomes Now</button></div>
            <div class="col-6"><button class="btn action-btn" onclick="runAction('retrain_all')">Retrain All Models</button></div>
            <div class="col-6"><button class="btn action-btn" style="border-color:#a78bfa40;color:#a78bfa" onclick="runAction('run_bootstrap')">🚀 Run Bootstrap</button></div>
            <div class="col-6"><button class="btn action-btn" style="border-color:#38bdf840;color:#38bdf8" onclick="runAction('run_walk_forward')">🧪 Walk-Forward Test</button></div>
            <div class="col-12 d-flex gap-2">
              <select id="action-symbol" class="form-control" style="max-width:190px">
                <option value="">Pick symbol</option>
              </select>
              <button class="btn action-btn" onclick="runSymbolRetrain()">Retrain</button>
            </div>
          </div>
          <div id="action-msg" class="small mt-2" style="min-height:1.2em;color:var(--muted)"></div>
        </div>
      </div>
      <div class="col-md-7">
        <div class="card p-3 h-100">
          <div class="section-head">🛰 Operator Snapshot</div>
          <div id="ops-snapshot" style="font-size:.82rem;color:var(--muted);line-height:1.6"></div>
        </div>
      </div>
    </div>

    <!-- Live job progress (hidden unless running) -->
    <div class="row g-2 mb-2" id="job-progress-row" style="display:none">
      <div class="col-12">
        <div class="card p-3" style="border-color:#a78bfa40;background:#1a1a2e">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <div class="section-head mb-0" style="color:#a78bfa">⚙️ Training Job In Progress</div>
            <span id="job-status-badge" class="badge" style="background:#a78bfa22;color:#a78bfa">running</span>
          </div>
          <div id="job-progress-bar-wrap" class="mb-2">
            <div class="progress" style="height:8px">
              <div id="job-progress-bar" class="progress-bar" style="width:0%;background:#a78bfa;transition:width .4s"></div>
            </div>
          </div>
          <div class="d-flex justify-content-between" style="font-size:.78rem">
            <span id="job-progress-label" style="color:#a78bfa">Starting…</span>
            <span id="job-progress-pct" style="color:var(--muted)"></span>
          </div>
        </div>
      </div>
    </div>

    <!-- Sub-navigation -->
    <ul class="nav sub-tabs" id="learning-subtabs" role="tablist">
      <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#lt-health" role="tab">🩺 Health</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#lt-analytics" role="tab">📊 Analytics</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#lt-log" role="tab">📋 Log</button></li>
    </ul>

    <div class="tab-content">

      <!-- ── Health sub-tab ── -->
      <div class="tab-pane fade show active" id="lt-health" role="tabpanel">

        <!-- Job history (compact) -->
        <div class="card p-3 mb-2">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <div class="section-head mb-0">📦 Job History</div>
            <button class="btn btn-sm" style="background:var(--surface2);color:var(--accent);border:1px solid var(--border)" onclick="refreshJobs()">↻ Refresh</button>
          </div>
          <div id="job-history" style="max-height:160px;overflow-y:auto"></div>
        </div>

        <!-- Model health + suggestions inline -->
        <div class="card p-3 mb-2">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <div class="section-head mb-0">🩺 Model Health per Symbol</div>
            <div class="d-flex gap-2 align-items-center flex-wrap">
              <select id="activity-symbol" class="form-control" style="max-width:140px" onchange="refreshActivity()">
                <option value="">All symbols</option>
              </select>
              <select id="activity-days" class="form-control" style="max-width:130px" onchange="refreshActivity()">
                <option value="0">All time</option>
                <option value="7">Last 7 days</option>
                <option value="30">Last 30 days</option>
                <option value="90">Last 90 days</option>
              </select>
              <button class="btn btn-sm" style="background:var(--surface2);color:var(--accent);border:1px solid var(--border)" onclick="refreshActivity()">↻</button>
            </div>
          </div>
          <div id="health-summary-bar" class="d-flex gap-3 mb-2 flex-wrap" style="font-size:.78rem"></div>
        <div id="health-cards" class="row g-2 mb-3"></div>
          <div style="border-top:1px solid var(--border);padding-top:10px">
            <div class="section-head" style="font-size:.74rem;color:var(--muted);margin-bottom:6px">💡 What Needs Attention</div>
            <div id="suggestions"></div>
          </div>
        </div>

      </div><!-- lt-health -->

      <!-- ── Analytics sub-tab ── -->
      <div class="tab-pane fade" id="lt-analytics" role="tabpanel">
        <div class="row g-2 mb-2">
          <div class="col-md-6">
            <div class="card p-3">
              <div class="section-head">🧭 Outcome Resolution</div>
              <div style="position:relative;height:160px"><canvas id="chart-resolution-reasons"></canvas></div>
            </div>
          </div>
          <div class="col-md-6">
            <div class="card p-3">
              <div class="section-head">⏱ Avg Time to Outcome</div>
              <canvas id="chart-resolution-time" height="140"></canvas>
            </div>
          </div>
        </div>
        <div class="row g-2">
          <div class="col-md-6">
            <div class="card p-3">
              <div class="section-head">📏 MFE vs MAE by Symbol</div>
              <canvas id="chart-mfe-mae" height="140"></canvas>
            </div>
          </div>
          <div class="col-md-6">
            <div class="card p-3 h-100">
              <div class="section-head">🧠 Adaptive TP/SL Tuning</div>
              <div id="adaptive-summary" style="font-size:.82rem;color:var(--muted);line-height:1.6"></div>
            </div>
          </div>
        </div>
      </div><!-- lt-analytics -->

      <!-- ── Log sub-tab ── -->
      <div class="tab-pane fade" id="lt-log" role="tabpanel">
        <div class="row g-2">
          <div class="col-md-6">
            <div class="card p-3 h-100">
              <div class="section-head">📋 Learning Cycle Log</div>
              <div id="train-log" style="max-height:520px;overflow-y:auto"></div>
            </div>
          </div>
          <div class="col-md-6">
            <div class="card p-3 h-100">
              <div class="section-head">📬 Recent Outcomes</div>
              <div id="outcome-feed" style="max-height:520px;overflow-y:auto"></div>
            </div>
          </div>
        </div>
      </div><!-- lt-log -->

    </div><!-- inner tab-content -->
  </div>

  <!-- ── PERFORMANCE TAB ───────────────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-performance">
    <div class="row g-3 mb-3">
      <!-- Equity Curve -->
      <div class="col-12">
        <div class="card p-3">
          <div class="d-flex align-items-center justify-content-between mb-2">
            <h6 class="mb-0">📈 Simulated Equity Curve <small class="text-muted">($1,000 per signal)</small></h6>
            <div id="eq-summary" class="text-muted small"></div>
          </div>
          <canvas id="equityChart" height="90"></canvas>
        </div>
      </div>
      <!-- Rolling Win Rate -->
      <div class="col-12">
        <div class="card p-3">
          <h6 class="mb-2">📊 Rolling Win Rate (7-day &amp; 30-day)</h6>
          <canvas id="wrChart" height="80"></canvas>
        </div>
      </div>
    </div>
    <div class="row g-3">
      <!-- By signal strength -->
      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-2">By Signal Strength (last 30 days)</h6>
          <table class="table table-sm mb-0" id="strengthTable">
            <thead><tr><th>Strength</th><th>Total</th><th>Correct</th><th>Win Rate</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <!-- By symbol -->
      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-2">By Symbol (last 30 days)</h6>
          <table class="table table-sm mb-0" id="symPerfTable">
            <thead><tr><th>Symbol</th><th>Total</th><th>Correct</th><th>Win Rate</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- ── EVENT TIMELINE TAB ───────────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-timeline">

    <div class="d-flex justify-content-between align-items-center mb-2 flex-wrap gap-2">
      <div class="d-flex gap-2 flex-wrap">
        <label style="font-size:.78rem;color:var(--muted);align-self:center">Filter:</label>
        <label class="d-flex align-items-center gap-1" style="font-size:.78rem;cursor:pointer"><input type="checkbox" id="tl-type-signal" checked onchange="refreshTimeline()"> Signals</label>
        <label class="d-flex align-items-center gap-1" style="font-size:.78rem;cursor:pointer"><input type="checkbox" id="tl-type-outcome" checked onchange="refreshTimeline()"> Outcomes</label>
        <label class="d-flex align-items-center gap-1" style="font-size:.78rem;cursor:pointer"><input type="checkbox" id="tl-type-train" checked onchange="refreshTimeline()"> Training</label>
        <label class="d-flex align-items-center gap-1" style="font-size:.78rem;cursor:pointer"><input type="checkbox" id="tl-type-scan" checked onchange="refreshTimeline()"> Scans</label>
        <label class="d-flex align-items-center gap-1" style="font-size:.78rem;cursor:pointer"><input type="checkbox" id="tl-type-action" onchange="refreshTimeline()"> Actions</label>
      </div>
      <div class="d-flex gap-2 align-items-center">
        <select id="tl-limit" class="form-control" style="max-width:110px" onchange="refreshTimeline()">
          <option value="50">Last 50</option>
          <option value="100" selected>Last 100</option>
          <option value="200">Last 200</option>
        </select>
        <button class="btn btn-sm" style="background:var(--surface2);color:var(--accent);border:1px solid var(--border)" onclick="refreshTimeline()">↻ Refresh</button>
      </div>
    </div>

    <div id="timeline-feed" style="max-height:76vh;overflow-y:auto"></div>

  </div>

  <!-- ── ADMIN TAB ─────────────────────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-admin">

    <div class="row g-2 mb-3">
      <div class="col-12">
        <div class="card p-3" style="border-color:#a78bfa40">
          <div class="section-head">🚀 ML Bootstrap — Pre-train from 2 Years of Historical Data</div>
          <p class="small mb-2 text-muted">
            Runs the signal engine on 2 years of hourly bars per symbol, labels each signal
            as correct/incorrect from TP/SL outcomes, and trains the ML models.
            Takes ~5–10 minutes. Progress is visible in the Learning tab.
          </p>
          <div class="d-flex gap-3 align-items-center flex-wrap">
            <button class="btn" style="background:var(--surface2);color:var(--purple);border:1px solid var(--border);padding:8px 24px;font-size:.9rem"
              onclick="adminBootstrap()">🚀 Run Bootstrap (All Symbols)</button>
            <div id="bootstrap-msg" class="small text-muted"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- AI Sentiment Settings -->
    <div class="row g-2 mb-3">
      <div class="col-12">
        <div class="card p-3" style="border-color:#38bdf840">
          <div class="section-head">🧠 AI Sentiment Settings</div>
          <p class="small mb-2 text-muted">
            Switch provider without restarting the bot — changes take effect on the next scan.
          </p>
          <div class="row g-3" id="settings-form">

            <div class="col-md-4">
              <label class="small text-muted">Sentiment Provider</label>
              <select id="set-sentiment_provider" class="form-control mt-1">
                <option value="disabled">Disabled (no sentiment)</option>
                <option value="local_finbert">Local FinBERT (Docker sidecar)</option>
                <option value="gemini">Gemini 1.5 Flash (free tier)</option>
                <option value="claude">Claude API (cloud)</option>
              </select>
              <div class="small mt-1 text-muted">
                Gemini: 1,500 free calls/day — switch without restart
              </div>
            </div>

            <div class="col-md-4">
              <label class="small text-muted">News Provider</label>
              <select id="set-news_provider" class="form-control mt-1">
                <option value="disabled">Disabled</option>
                <option value="finnhub">Finnhub (free tier)</option>
              </select>
            </div>

            <div class="col-md-4">
              <label class="small text-muted">Suppress Threshold (0.0–1.0)</label>
              <input type="number" id="set-sentiment_suppress_threshold" class="form-control mt-1"
                step="0.05" min="0" max="1" placeholder="0.35">
              <div class="small mt-1 text-muted">
                Sentiment score above this suppresses conflicting signal
              </div>
            </div>

            <div class="col-md-4">
              <label class="small text-muted">Finnhub API Key</label>
              <input type="text" id="set-finnhub_api_key" class="form-control mt-1"
                placeholder="Free key from finnhub.io">
            </div>

            <div class="col-md-4">
              <label class="small text-muted">Gemini API Key</label>
              <input type="password" id="set-gemini_api_key" class="form-control mt-1"
                placeholder="AIza... (Google AI Studio, free tier)">
            </div>

            <div class="col-md-4">
              <label class="small text-muted">Claude API Key</label>
              <input type="password" id="set-claude_api_key" class="form-control mt-1"
                placeholder="sk-ant-... (for claude provider)">
            </div>

            <div class="col-md-4">
              <label class="small text-muted">FinBERT Service URL</label>
              <input type="text" id="set-sentiment_local_url" class="form-control mt-1"
                placeholder="http://finbert:5001">
              <div class="small mt-1 text-muted">
                Change this when deploying to a different host
              </div>
            </div>

            <div class="col-md-4">
              <label class="small text-muted">News Lookback (hours)</label>
              <input type="number" id="set-news_lookback_hours" class="form-control mt-1"
                step="1" min="1" max="48" placeholder="6">
            </div>

            <div class="col-md-4">
              <label class="small text-muted">Outcome Notifications in DM</label>
              <select id="set-outcome_notify_admin" class="form-control mt-1">
                <option value="false">Disabled (no DM for outcomes)</option>
                <option value="true">Enabled (send to admin DM)</option>
              </select>
              <div class="small mt-1 text-muted">Outcome results are always visible in the dashboard</div>
            </div>

            <div class="col-md-4">
              <label class="small text-muted">Display Currency</label>
              <select id="set-display_currency" class="form-control mt-1">
                <option value="USD">$ USD — US Dollar</option>
                <option value="EUR">€ EUR — Euro</option>
                <option value="GBP">£ GBP — British Pound</option>
                <option value="CHF">Fr CHF — Swiss Franc</option>
              </select>
              <div class="small mt-1 text-muted">Price, TP and SL in Telegram signals will be shown in this currency</div>
            </div>


          </div>
          <div class="d-flex gap-3 align-items-center mt-3">
            <button class="btn" style="background:var(--surface2);color:var(--accent);border:1px solid var(--border);padding:6px 20px"
              onclick="saveSettings()">💾 Save Settings</button>
            <div id="settings-msg" class="small text-muted"></div>
          </div>

          <!-- Per-symbol sentiment status -->
          <div class="mt-4">
            <div class="section-head" style="font-size:.74rem;color:var(--muted)">Current Sentiment Cache</div>
            <div id="sentiment-status" style="font-size:.78rem"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="row g-2 mb-3">
      <div class="col-md-6">
        <div class="card p-3">
          <div class="section-head">⚙️ Watched Symbols</div>
          <p class="small mb-2 text-muted">
            Changes take effect within 5 minutes.
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
              style="z-index:1000;display:none;background:var(--surface);border:1px solid var(--border);
                     border-radius:6px;max-height:220px;overflow-y:auto"></div>
          </div>
          <div id="add-msg" class="small mb-2" style="min-height:1.2em"></div>

          <div id="symbol-list"></div>
        </div>
      </div>

      <div class="col-md-6">
        <div class="card p-3">
          <div class="section-head">📊 Symbol Signal Stats</div>
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

let dailyChart, symbolChart, bucketsChart, strengthChart, resolutionReasonsChart, resolutionTimeChart, mfeMaeChart, trainRunsChart, scanHistoryChart;
let _fxRate = 1.0, _fxSymbol = '$';
async function _loadFx() {
  const fx = await fetch('/api/fx-rate').then(r=>r.json());
  _fxRate = fx.rate || 1.0;
  _fxSymbol = fx.symbol || '$';
}
function fxPrice(usd) { return (_fxRate * usd).toFixed(2); }
_loadFx();

// ── Command Center ────────────────────────────────────────────────────────────
async function refreshCommandCenter() {
  const data = await fetch('/api/system/status').then(r=>r.json());
  const ops  = await fetch('/api/ops').then(r=>r.json());
  const last = data.last_scan;
  const scans = data.recent_scans || [];

  // Bot health: check last signal or training activity
  const botEl = document.getElementById('cc-bot-status');
  const botMeta = document.getElementById('cc-bot-meta');
  if (ops.last_signal) {
    const ageH = (Date.now() - new Date(ops.last_signal.sent_at+'Z')) / 3600000;
    const healthy = ageH < 48;
    botEl.innerHTML = `<span class="status-dot ${healthy?'status-ok':'status-warn'}"></span>${healthy?'Active':'Idle'}`;
    botMeta.textContent = `Last signal: ${formatShortTime(ops.last_signal.sent_at)}`;
  } else if (ops.last_cycle) {
    const ageH = (Date.now() - new Date(ops.last_cycle.checked_at+'Z')) / 3600000;
    botEl.innerHTML = `<span class="status-dot ${ageH<2?'status-ok':'status-warn'}"></span>${ageH<2?'Active':'Checking…'}`;
    botMeta.textContent = `Last ML check: ${formatShortTime(ops.last_cycle.checked_at)}`;
  } else {
    botEl.innerHTML = `<span class="status-dot status-bad"></span>No activity`;
    botMeta.textContent = 'No signals or training logged';
  }

  // FinBERT
  const fbEl = document.getElementById('cc-finbert-status');
  const fbMeta = document.getElementById('cc-finbert-meta');
  if (data.finbert_ok) {
    fbEl.innerHTML = `<span class="status-dot status-ok"></span>Online`;
    fbMeta.textContent = 'FinBERT sentiment service';
  } else {
    fbEl.innerHTML = `<span class="status-dot status-bad"></span>Offline`;
    fbMeta.textContent = 'Check local_finbert setting';
  }

  // Perf brake
  const brakeEl = document.getElementById('cc-brake-status');
  const brakeMeta = document.getElementById('cc-brake-meta');
  if (last && last.perf_brake_engaged) {
    brakeEl.innerHTML = `<span class="status-dot status-bad"></span>ENGAGED`;
    brakeMeta.textContent = 'Halving signal frequency';
  } else {
    brakeEl.innerHTML = `<span class="status-dot status-ok"></span>OFF`;
    brakeMeta.textContent = 'Normal scan frequency';
  }

  // Last scan
  if (last) {
    const atCet = last.scanned_at ? new Date(last.scanned_at+'Z').toLocaleString() : '—';
    document.getElementById('cc-scan-at').textContent = atCet;
    document.getElementById('cc-sym-total').textContent = last.symbols_total ?? '—';
    document.getElementById('cc-sym-scanned').textContent = last.symbols_scanned ?? '—';
    document.getElementById('cc-sym-skipped-mkt').textContent = last.symbols_skipped_market ?? '—';
    document.getElementById('cc-signals-gen').textContent = last.signals_generated ?? '—';
    const dur = last.scan_duration_ms != null ? (last.scan_duration_ms > 1000 ? (last.scan_duration_ms/1000).toFixed(1)+'s' : Math.round(last.scan_duration_ms)+'ms') : '—';
    document.getElementById('cc-scan-dur').textContent = dur;
  }

  // Scan history chart
  const chartData = scans.slice().reverse();
  const labels = chartData.map(s => s.scanned_at ? new Date(s.scanned_at+'Z').toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}) : '');
  const sigVals = chartData.map(s => s.signals_generated ?? 0);
  const evalVals = chartData.map(s => s.symbols_scanned ?? 0);
  if (!scanHistoryChart) {
    scanHistoryChart = new Chart(document.getElementById('chart-scan-history'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {label:'Evaluated', data:evalVals, backgroundColor:'#38bdf830', borderColor:'#38bdf8', borderWidth:1, yAxisID:'y2'},
          {label:'Signals', data:sigVals, backgroundColor:'#4ade80cc', borderColor:'#4ade80', borderWidth:1, yAxisID:'y'},
        ]
      },
      options: {
        plugins: {legend:{labels:{color:'#8b8fa8'}}},
        scales: {
          x: {ticks:{color:'#8b8fa8',maxTicksLimit:15},grid:{color:'#2a2d3a'}},
          y: {position:'left',ticks:{color:'#4ade80',stepSize:1},grid:{color:'#2a2d3a'},title:{display:true,text:'Signals',color:'#4ade80'}},
          y2: {position:'right',ticks:{color:'#38bdf8'},grid:{drawOnChartArea:false},title:{display:true,text:'Evaluated',color:'#38bdf8'}},
        }
      }
    });
  } else {
    scanHistoryChart.data.labels = labels;
    scanHistoryChart.data.datasets[0].data = evalVals;
    scanHistoryChart.data.datasets[1].data = sigVals;
    scanHistoryChart.update();
  }

  // Training job
  const jobEl = document.getElementById('cc-training-job');
  const job = data.running_job;
  if (job) {
    const pct = job.total_symbols > 0 ? Math.round(job.done_symbols / job.total_symbols * 100) : 0;
    const cur = job.current_symbol ? ` — ${job.current_symbol}` : '';
    jobEl.innerHTML = `
      <div style="color:#a78bfa;font-weight:600;margin-bottom:6px">${job.job_type} running${cur}</div>
      <div class="progress mb-2" style="height:8px">
        <div class="progress-bar" style="width:${pct}%;background:#a78bfa;transition:width .4s"></div>
      </div>
      <div style="color:var(--muted);font-size:.76rem">${job.done_symbols}/${job.total_symbols} symbols · ${pct}%</div>
      <div style="color:#555;font-size:.72rem;margin-top:4px">Started: ${formatShortTime(job.started_at)}</div>`;
  } else {
    jobEl.innerHTML = '<div style="color:#555">No training job running.<br><span style="font-size:.76rem">Use Action Center in Learning tab to start one.</span></div>';
  }

  // Scan log table
  const regimeColors = {bull:'#4ade80',bear:'#f87171',volatile:'#fbbf24',unknown:'#555'};
  document.getElementById('cc-scan-log-body').innerHTML = scans.map(s => {
    const atStr = s.scanned_at ? new Date(s.scanned_at+'Z').toLocaleString() : '—';
    const dur = s.scan_duration_ms != null ? (s.scan_duration_ms > 1000 ? (s.scan_duration_ms/1000).toFixed(1)+'s' : Math.round(s.scan_duration_ms)+'ms') : '—';
    const sigColor = s.signals_generated > 0 ? 'var(--accent)' : 'var(--muted)';
    const regimeColor = s.regime ? (regimeColors[s.regime] || '#8b8fa8') : '#555';
    return `<tr>
      <td style="white-space:nowrap;color:var(--muted)">${atStr}</td>
      <td>${s.symbols_total ?? '—'}</td>
      <td style="color:var(--green)">${s.symbols_scanned ?? '—'}</td>
      <td style="color:var(--muted)">${s.symbols_skipped_market ?? '—'}</td>
      <td style="color:${sigColor};font-weight:600">${s.signals_generated ?? '—'}</td>
      <td style="color:var(--muted)">${dur}</td>
      <td style="color:${regimeColor}">${s.regime || '—'}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" class="text-center text-muted">No scan data yet — runs on first scan cycle</td></tr>';
}

function statusChip(ok, okText, warnText) {
  if (ok === 'ok')   return `<span class="status-dot status-ok d-inline-block"></span><span style="color:var(--green)">${okText}</span>`;
  if (ok === 'warn') return `<span class="status-dot status-warn d-inline-block"></span><span style="color:var(--yellow)">${warnText}</span>`;
  return `<span class="status-dot status-bad d-inline-block"></span><span style="color:var(--red)">${warnText}</span>`;
}

async function refreshOps() {
  const ops = await fetch('/api/ops').then(r=>r.json());
  const lastSignal = ops.last_signal;
  const lastCycle = ops.last_cycle;
  const signalHealthy = lastSignal ? ((Date.now() - new Date(lastSignal.sent_at+'Z')) / 3600000 < 24 ? 'ok' : 'warn') : 'bad';
  const learningHealthy = lastCycle ? ((Date.now() - new Date(lastCycle.checked_at+'Z')) / 3600000 < 1 ? 'ok' : 'warn') : 'bad';
  document.getElementById('ops-signal-status').innerHTML = statusChip(signalHealthy, 'Signals Flowing', 'Signals Idle');
  const learningLabel = lastCycle?.retrained ? 'Retrained' : 'Monitoring';
  document.getElementById('ops-learning-status').innerHTML = statusChip(learningHealthy, learningLabel, 'Learning Stale');
  document.getElementById('ops-pending-work').textContent = `${ops.pending_outcomes} / ${ops.pending_actions}`;
  document.getElementById('ops-coverage').textContent = `${ops.active_symbol_count}`;

  const actionSelect = document.getElementById('action-symbol');
  if (actionSelect && actionSelect.options.length <= 1) {
    actionSelect.innerHTML = '<option value="">Pick symbol</option>' + ops.active_symbols.map(sym => `<option value="${sym}">${sym}</option>`).join('');
  }
  const snapshot = [];
  if (ops.last_action) snapshot.push(`Last action: <span style="color:var(--text)">${ops.last_action.action}${ops.last_action.symbol ? ' ' + ops.last_action.symbol : ''}</span> <span class="text-muted">(${ops.last_action.status})</span>`);
  if (ops.last_train) snapshot.push(`Last train: <span style="color:var(--text)">${ops.last_train.symbol}</span> <span class="text-muted">at ${new Date(ops.last_train.trained_at+'Z').toLocaleString()}</span>`);
  if (ops.last_outcome) snapshot.push(`Last outcome: <span style="color:var(--text)">${ops.last_outcome.symbol} ${ops.last_outcome.outcome}</span> <span class="text-muted">via ${ops.last_outcome.resolution_reason || 'n/a'}</span>`);
  snapshot.push(`Tracked symbols: <span style="color:var(--text)">${ops.active_symbol_count}</span>`);
  const snapEl = document.getElementById('ops-snapshot');
  if (snapEl) snapEl.innerHTML = snapshot.map(s => `<div class="mb-1">${s}</div>`).join('');
}

async function runAction(action, symbol='') {
  const msg = document.getElementById('action-msg');
  msg.style.color = '#8b8fa8';
  msg.textContent = 'Queueing action...';
  const payload = {action};
  if (symbol) payload.symbol = symbol;
  const res = await fetch('/api/actions', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload),
  }).then(r=>r.json());
  if (res.ok) {
    msg.style.color = '#4ade80';
    msg.textContent = `${action} queued — bot picks up within 60s`;
    refreshOps();
    refreshActivity();
    setTimeout(refreshJobs, 3000);
  } else {
    msg.style.color = '#f87171';
    msg.textContent = res.error || 'Action failed';
  }
}

function runSymbolRetrain() {
  const symbol = document.getElementById('action-symbol').value;
  if (!symbol) {
    const msg = document.getElementById('action-msg');
    msg.style.color = '#f87171';
    msg.textContent = 'Choose a symbol first';
    return;
  }
  runAction('retrain_symbol', symbol);
}

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
function formatShortTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso+'Z');
  const now = new Date();
  const diffDays = (now - d) / 86400000;
  if (diffDays < 1) return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  if (diffDays < 7) return d.toLocaleDateString([], {weekday:'short'}) + ' ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  return d.toLocaleDateString([], {month:'short', day:'numeric'}) + ' ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}
function accColor(acc) {
  if (acc === null || acc === undefined) return '#8b8fa8';
  if (acc >= 60) return '#4ade80';
  if (acc >= 50) return '#fbbf24';
  return '#f87171';
}
function accBar(acc, total) {
  if (!total) return '<span class="text-muted">no data</span>';
  const cls = acc >= 60 ? 'accuracy-bar-high' : (acc >= 50 ? 'accuracy-bar-mid' : 'accuracy-bar-low');
  return `<div class="d-flex align-items-center gap-2">
    <div class="progress flex-grow-1" style="height:6px">
      <div class="progress-bar ${cls}" style="width:${acc}%"></div>
    </div>
    <span style="color:${accColor(acc)};min-width:42px;text-align:right">${acc}%</span>
    <span class="text-muted" style="font-size:.75rem">(${total})</span>
  </div>`;
}

// ── Market status bar ─────────────────────────────────────────────────────────
function updateMarketStatus() {
  const now = new Date();
  // CET/CEST offset detection
  const cetOffset = new Date(now.toLocaleString('en-US', {timeZone:'Europe/Berlin'})) - now;
  const cetHour = new Date(now.getTime() + cetOffset);
  const cetH = cetHour.getHours() + cetHour.getMinutes() / 60;
  const cetStr = cetHour.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  const isTZ = cetHour.getTimezoneOffset() === 0;

  // ET offset
  const etOffset = new Date(now.toLocaleString('en-US', {timeZone:'America/New_York'})) - now;
  const etHour = new Date(now.getTime() + etOffset);
  const etH = etHour.getHours() + etHour.getMinutes() / 60;
  const etStr = etHour.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});

  const dow = now.getDay(); // 0=Sun, 6=Sat

  // XETRA: Mon-Fri 09:00-17:30 CET
  const xetraOpen = dow >= 1 && dow <= 5 && cetH >= 9 && cetH < 17.5;
  // US: Mon-Fri 09:30-16:00 ET
  const usOpen = dow >= 1 && dow <= 5 && etH >= 9.5 && etH < 16;
  // Crypto: always open

  function mkDot(open, label, hours) {
    const cls = open ? 'status-ok' : 'status-bad';
    const status = open ? 'Open' : 'Closed';
    return `<span class="d-flex align-items-center gap-1"><span class="status-dot ${cls}"></span><span style="color:var(--text)">${label}</span><span style="color:var(--dim)">${status}</span></span>`;
  }

  document.getElementById('market-us').innerHTML    = mkDot(usOpen,    '🇺🇸 US',     '');
  document.getElementById('market-xetra').innerHTML = mkDot(xetraOpen, '🇩🇪 XETRA', '');
  document.getElementById('market-crypto').innerHTML= mkDot(true,      '₿ Crypto',  '');
  document.getElementById('market-times').textContent = `CET ${cetStr}  ·  ET ${etStr}`;
}

// ── Overview ──────────────────────────────────────────────────────────────────
async function refreshOverview() {
  const [stats, signals, regimeData] = await Promise.all([
    fetch('/api/stats').then(r=>r.json()),
    fetch('/api/signals').then(r=>r.json()),
    fetch('/api/regime').then(r=>r.json()).catch(()=>({regime:'unknown',attribution:[]})),
  ]);

  // Regime badge
  const regimeBadge = document.getElementById('regime-badge');
  const regimeColors = {bull:'bg-success', bear:'bg-danger', volatile:'bg-warning text-dark', unknown:'bg-secondary'};
  const regimeIcons  = {bull:'↑', bear:'↓', volatile:'~', unknown:'?'};
  if (regimeBadge) {
    const r = regimeData.regime || 'unknown';
    regimeBadge.className = 'badge fs-6 px-3 py-1 ' + (regimeColors[r]||'bg-secondary');
    regimeBadge.textContent = (regimeIcons[r]||'') + ' ' + r.toUpperCase();
  }
  const regimeAttr = document.getElementById('regime-attribution');
  if (regimeAttr && regimeData.attribution && regimeData.attribution.length > 0) {
    regimeAttr.textContent = regimeData.attribution.map(
      a => `${a.regime}: ${a.win_rate}% WR (${a.total} trades)`
    ).join('  ·  ');
  }

  document.getElementById('s-users').textContent = stats.total_users;
  document.getElementById('s-total').textContent = stats.total_signals;
  document.getElementById('s-today').textContent = stats.today_signals;
  document.getElementById('s-acc').textContent = stats.resolved > 0 ? stats.accuracy+'%' : 'N/A';
  document.getElementById('last-refresh').textContent = 'Updated '+new Date().toLocaleTimeString();
  refreshOps();

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

  document.getElementById('signals-body').innerHTML = signals.map(s=>`<tr>
    <td style="white-space:nowrap;color:var(--muted)">${formatShortTime(s.sent_at)}</td>
    <td><span class="sym-tag">${s.symbol}</span></td>
    <td>${badge(s.action,'badge-'+s.action)}</td>
    <td>${badge(s.strength||'RULE','badge-'+(s.strength||'RULE'))}</td>
    <td style="white-space:nowrap">${s.price?_fxSymbol+fxPrice(s.price):'—'}</td>
    <td style="color:var(--muted)">${s.ai_confidence?(s.ai_confidence*100).toFixed(0)+'%':'—'}</td>
    <td>${outcomeBadge(s.outcome)}</td>
  </tr>`).join('');
}

// ── Signals with Explainability ───────────────────────────────────────────────
let _allSignals = [];

async function refreshSignalsTab() {
  if (!_allSignals.length) {
    _allSignals = await fetch('/api/signals').then(r=>r.json());
  }
  const filterAction   = document.getElementById('sig-filter-action')?.value || '';
  const filterOutcome  = document.getElementById('sig-filter-outcome')?.value || '';
  const filterStrength = document.getElementById('sig-filter-strength')?.value || '';

  let sigs = _allSignals;
  if (filterAction)   sigs = sigs.filter(s => s.action   === filterAction);
  if (filterOutcome)  sigs = sigs.filter(s => s.outcome  === filterOutcome);
  if (filterStrength) sigs = sigs.filter(s => s.strength === filterStrength);

  const container = document.getElementById('signals-explainability-list');
  if (!sigs.length) {
    container.innerHTML = '<div class="text-muted text-center py-4">No signals match the selected filters.</div>';
    return;
  }

  container.innerHTML = sigs.map((s, idx) => {
    const outcomeClasses = {correct:'badge-correct',incorrect:'badge-incorrect',neutral:'badge-neutral',pending:'badge-pending'};
    const outBadge = outcomeBadge(s.outcome);
    const actBadge = badge(s.action, 'badge-'+s.action);
    const typBadge = badge(s.strength||'RULE','badge-'+(s.strength||'RULE'));
    const aiPct = s.ai_confidence ? (s.ai_confidence*100).toFixed(0)+'%' : '—';
    const dur = s.resolution_minutes ? Math.round(s.resolution_minutes)+'m' : '—';
    const regime = s.regime || '—';
    const regColor = {bull:'var(--green)',bear:'var(--red)',volatile:'var(--yellow)',unknown:'var(--muted)'}[regime]||'var(--muted)';
    const price = s.price ? _fxSymbol + fxPrice(s.price) : '—';
    const tp = s.tp ? _fxSymbol + fxPrice(s.tp) : '—';
    const sl = s.sl ? _fxSymbol + fxPrice(s.sl) : '—';

    // Explainability bars
    const reasons = Array.isArray(s.reasons) ? s.reasons : [];
    const hasRuleSignal = reasons.some(r => /EMA|RSI|MACD|volume/i.test(r));
    const hasAI = s.ai_confidence !== null && s.ai_confidence !== undefined;
    const hasSentiment = reasons.some(r => /sentiment/i.test(r));

    const ruleScore  = hasRuleSignal ? Math.min(100, 30 + reasons.filter(r => /EMA|RSI|MACD|volume/i.test(r)).length * 15) : 0;
    const aiScore    = hasAI ? Math.round((s.ai_confidence || 0) * 100) : 0;
    const sentScore  = hasSentiment ? 12 : 0;
    const qualScore  = (s.quality || 0) * 8;

    function scoreBar(label, score, color) {
      if (!score) return '';
      const w = Math.min(score, 100);
      return `<div class="d-flex align-items-center gap-2 mb-1" style="font-size:.74rem">
        <span style="color:var(--muted);min-width:90px">${label}</span>
        <div class="progress flex-grow-1" style="height:5px">
          <div class="progress-bar" style="width:${w}%;background:${color}"></div>
        </div>
        <span style="color:${color};min-width:36px;text-align:right">${score}%</span>
      </div>`;
    }

    const explainBars = `
      ${scoreBar('Rule Engine', ruleScore, 'var(--green)')}
      ${scoreBar('ML Engine', aiScore, 'var(--purple)')}
      ${scoreBar('Sentiment', sentScore, 'var(--accent)')}
      ${scoreBar('Quality Score', qualScore, 'var(--yellow)')}
    `;

    const reasonsList = reasons.length
      ? reasons.map(r => `<div class="mb-1" style="font-size:.76rem;color:var(--text)">• ${r}</div>`).join('')
      : '<div class="text-muted" style="font-size:.76rem">No reasons recorded</div>';

    const mfe = s.max_favorable_pct != null ? `+${Number(s.max_favorable_pct).toFixed(2)}%` : '—';
    const mae = s.max_adverse_pct  != null ? `${Number(s.max_adverse_pct).toFixed(2)}%` : '—';
    const resReason = s.resolution_reason ? s.resolution_reason.replace(/_/g,' ') : '—';

    return `
    <div class="card mb-2" id="sig-card-${idx}">
      <div class="p-3" style="cursor:pointer" onclick="toggleSigDetail(${idx})">
        <div class="d-flex align-items-center gap-2 flex-wrap">
          <span class="sym-tag">${s.symbol}</span>
          ${actBadge} ${typBadge} ${outBadge}
          <span style="color:var(--muted);font-size:.78rem">${formatShortTime(s.sent_at)}</span>
          <span style="color:var(--text);font-size:.80rem">${price}</span>
          <span style="color:var(--muted);font-size:.76rem">AI: ${aiPct}</span>
          <span style="color:${regColor};font-size:.72rem">${regime}</span>
          <span class="ms-auto text-muted" style="font-size:.72rem" id="sig-chevron-${idx}">▼ expand</span>
        </div>
        ${explainBars ? `<div class="mt-2">${explainBars}</div>` : ''}
      </div>
      <div id="sig-detail-${idx}" style="display:none;border-top:1px solid var(--border);padding:14px">
        <div class="row g-3">
          <div class="col-md-5">
            <div style="font-size:.76rem;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em">Signal Reasons</div>
            ${reasonsList}
          </div>
          <div class="col-md-4">
            <div style="font-size:.76rem;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em">Trade Details</div>
            <div style="font-size:.78rem;line-height:1.8">
              Entry: <span style="color:var(--text)">${price}</span><br>
              TP: <span style="color:var(--green)">${tp}</span> ${s.tp_pct ? `(+${Number(s.tp_pct).toFixed(1)}%)` : ''}<br>
              SL: <span style="color:var(--red)">${sl}</span> ${s.sl_pct ? `(${Number(s.sl_pct).toFixed(1)}%)` : ''}<br>
              R:R: <span style="color:var(--yellow)">${s.rr ? '1:'+s.rr : '—'}</span><br>
              RSI: <span style="color:var(--text)">${s.rsi ? Number(s.rsi).toFixed(1) : '—'}</span>
            </div>
          </div>
          <div class="col-md-3">
            <div style="font-size:.76rem;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em">Outcome</div>
            <div style="font-size:.78rem;line-height:1.8">
              Result: ${outBadge}<br>
              Resolution: <span style="color:var(--muted)">${resReason}</span><br>
              Duration: <span style="color:var(--text)">${dur}</span><br>
              MFE: <span style="color:var(--green)">${mfe}</span><br>
              MAE: <span style="color:var(--red)">${mae}</span>
            </div>
          </div>
        </div>
      </div>
    </div>`;
  }).join('');
}

function toggleSigDetail(idx) {
  const detail = document.getElementById(`sig-detail-${idx}`);
  const chevron = document.getElementById(`sig-chevron-${idx}`);
  if (!detail) return;
  const open = detail.style.display !== 'none';
  detail.style.display = open ? 'none' : '';
  if (chevron) chevron.textContent = open ? '▼ expand' : '▲ collapse';
}

// ── ML ────────────────────────────────────────────────────────────────────────
async function refreshML() {
  const ml = await fetch('/api/ml-stats').then(r=>r.json());
  const summary = ml.summary || {};
  document.getElementById('ml-active-models').textContent = summary.active_symbol_count ?? '—';
  document.getElementById('ml-trained-models').textContent = summary.trained_model_count ?? '—';
  document.getElementById('ml-resolved-ai').textContent = summary.ai_resolved_signals ?? '—';
  document.getElementById('ml-avg-acc').textContent = summary.avg_ai_accuracy !== null && summary.avg_ai_accuracy !== undefined ? `${summary.avg_ai_accuracy}%` : '—';

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

  // Per-symbol table (6 columns)
  const rows = Object.entries(ml.per_symbol).map(([sym, d]) => {
    const tr = d.training;
    const trainedAt = tr ? new Date(tr.trained_at+'Z').toLocaleString() : null;
    const samples   = tr ? tr.train_samples : null;
    const bwr = d.bootstrap_win_rate != null
      ? `<span style="color:${d.bootstrap_win_rate>=54?'#4ade80':d.bootstrap_win_rate>=50?'#fbbf24':'#f87171'};font-weight:600">${d.bootstrap_win_rate}%</span>
         <div class="text-muted" style="font-size:.70rem">${d.bootstrap_correct||0}✓ ${d.bootstrap_incorrect||0}✗ · ${d.bootstrap_samples||0} samples</div>`
      : '<span class="text-muted">—</span>';
    const mkCell = (total, acc, minN) => {
      if (!total) return '<span class="text-muted">—</span>';
      if (total < minN) {
        const pct = Math.round(total / minN * 100);
        return `<div class="text-muted" style="font-size:.72rem;margin-bottom:2px">${total} signal${total!==1?'s':''}</div>
                <div class="d-flex align-items-center gap-1">
                  <div class="progress flex-grow-1" style="height:4px"><div class="progress-bar" style="width:${pct}%;background:var(--dim)"></div></div>
                  <span class="text-muted" style="font-size:.70rem">${total}/${minN}</span>
                </div>`;
      }
      return `<div class="text-muted" style="font-size:.72rem;margin-bottom:2px">${total} signal${total!==1?'s':''}</div>
              ${accBar(acc, total)}`;
    };
    const trainCell = trainedAt
      ? `<div class="text-muted" style="font-size:.74rem">${trainedAt}</div>
         <div class="text-muted" style="font-size:.70rem">${samples} samples · ${d.train_runs||0} run${d.train_runs!==1?'s':''}</div>`
      : '<span class="text-muted">—</span>';
    return `<tr>
      <td><span class="sym-tag">${sym}</span></td>
      <td>${bwr}</td>
      <td>${mkCell(d.ai_total, d.ai_accuracy, 5)}</td>
      <td>${mkCell(d.strong_total, d.strong_accuracy, 3)}</td>
      <td>${mkCell(d.rule_total, d.rule_accuracy, 3)}</td>
      <td>${trainCell}</td>
    </tr>`;
  }).join('');
  document.getElementById('ml-body').innerHTML = rows ||
    '<tr><td colspan="6" class="text-center" style="color:#555">No resolved signals yet</td></tr>';
  const totalResolved = Object.values(ml.per_symbol).reduce((s,d) => s + (d.rule_total||0) + (d.ai_total||0) + (d.strong_total||0), 0);
  const noteEl = document.getElementById('ml-table-note');
  if (noteEl) {
    if (totalResolved < 50) noteEl.textContent = `${totalResolved} resolved signals — accuracy stabilises above ~50 per symbol`;
    else noteEl.textContent = '';
  }

  // Direction accuracy breakdown
  const dirData = await fetch('/api/accuracy-direction').then(r=>r.json());
  const dirRows = Object.entries(dirData).sort(([a],[b]) => a.localeCompare(b)).map(([sym, dirs]) => {
    function dirCell(d) {
      if (!d) return '<span class="text-muted">—</span>';
      if (d.correct + d.incorrect < 3) {
        return `<span class="text-muted">${d.total} sig (too few)</span>`;
      }
      const acc = d.accuracy;
      const color = acc >= 55 ? 'var(--green)' : acc >= 45 ? 'var(--yellow)' : 'var(--red)';
      return `<span style="color:${color};font-weight:600">${acc}%</span> <span class="text-muted">(${d.correct}/${d.correct+d.incorrect})</span>`;
    }
    const buy = dirs['BUY'], sell = dirs['SELL'];
    let diagnosis = '';
    const buyAcc = buy && (buy.correct + buy.incorrect) >= 3 ? buy.accuracy : null;
    const sellAcc = sell && (sell.correct + sell.incorrect) >= 3 ? sell.accuracy : null;
    if (buyAcc === null && sellAcc === null) {
      diagnosis = '<span class="text-muted">accumulating data</span>';
    } else if (buyAcc !== null && sellAcc !== null) {
      if (buyAcc < 40 && sellAcc < 40) {
        diagnosis = '<span style="color:var(--red)">⚠️ both directions unreliable</span>';
      } else if (buyAcc >= 50 && sellAcc < 40) {
        diagnosis = '<span style="color:var(--yellow)">BUY works · SELL drags accuracy</span>';
      } else if (sellAcc >= 50 && buyAcc < 40) {
        diagnosis = '<span style="color:var(--yellow)">SELL works · BUY drags accuracy</span>';
      } else if (buyAcc >= 50 && sellAcc >= 50) {
        diagnosis = '<span style="color:var(--green)">✓ both directions profitable</span>';
      } else {
        diagnosis = '<span class="text-muted">marginal</span>';
      }
    } else if (buyAcc !== null) {
      diagnosis = buyAcc >= 50
        ? '<span style="color:var(--green)">BUY profitable</span>'
        : '<span style="color:var(--red)">BUY underperforming</span>';
    } else {
      diagnosis = sellAcc >= 50
        ? '<span style="color:var(--green)">SELL profitable</span>'
        : '<span style="color:var(--red)">SELL underperforming</span>';
    }
    return `<tr>
      <td><span class="sym-tag">${sym}</span></td>
      <td>${dirCell(buy)}</td>
      <td>${dirCell(sell)}</td>
      <td style="font-size:.76rem">${diagnosis}</td>
    </tr>`;
  }).join('');
  document.getElementById('direction-body').innerHTML = dirRows ||
    '<tr><td colspan="4" class="text-center text-muted">No resolved signals yet</td></tr>';

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
      options:{maintainAspectRatio:false, plugins:{legend:{position:'right',labels:{color:'#8b8fa8',boxWidth:12,padding:10}}}}
    });
  } else {
    strengthChart.data.labels=Object.keys(bs);
    strengthChart.data.datasets[0].data=Object.values(bs);
    strengthChart.update();
  }

  const runLabels = Object.keys(ml.per_symbol || {});
  const runValues = runLabels.map(sym => ml.per_symbol[sym].train_runs || 0);
  if (!trainRunsChart) {
    trainRunsChart = new Chart(document.getElementById('chart-train-runs'), {
      type:'bar',
      data:{labels:runLabels, datasets:[{label:'Train runs', data:runValues, backgroundColor:'#a78bfa99', borderColor:'#a78bfa', borderWidth:1}]},
      options:{plugins:{legend:{labels:{color:'#8b8fa8'}}},
        scales:{x:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}},
                y:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}}}}
    });
  } else {
    trainRunsChart.data.labels = runLabels;
    trainRunsChart.data.datasets[0].data = runValues;
    trainRunsChart.update();
  }
  // Walk-forward validation results
  const wfData = await fetch('/api/wf-results').then(r=>r.json());
  const gradeColor = {A:'#4ade80', B:'#86efac', C:'#fbbf24', D:'#f87171', 'N/A':'#555'};
  const gradeTooltip = {A:'Strong edge (≥58%)', B:'Good (52–58%)', C:'Marginal (47–52%)', D:'Below random (<47%)', 'N/A':'Insufficient signals'};
  const wfRows = wfData.map(r => {
    const g = r.grade || 'N/A';
    const gradeBadge = `<span style="font-weight:700;font-size:.95rem;color:${gradeColor[g]||'#555'}" title="${gradeTooltip[g]||''}">${g}</span>`;
    const rwr = r.rule_win_rate != null ? `<span style="color:${r.rule_win_rate>=52?'#4ade80':r.rule_win_rate>=47?'#fbbf24':'#f87171'}">${r.rule_win_rate}%</span>` : '<span class="text-muted">—</span>';
    const awr = r.ai_win_rate != null ? `<span style="color:${r.ai_win_rate>=52?'#4ade80':r.ai_win_rate>=47?'#fbbf24':'#f87171'}">${r.ai_win_rate}%</span>` : '<span class="text-muted">—</span>';
    const tested = r.tested_at ? new Date(r.tested_at+'Z').toLocaleDateString() : '—';
    return `<tr>
      <td><span class="sym-tag">${r.symbol}</span></td>
      <td>${gradeBadge}</td>
      <td class="text-muted">${r.rule_signals ?? '—'}</td>
      <td>${rwr}</td>
      <td class="text-muted">${r.ai_signals ?? '—'}</td>
      <td>${awr}</td>
      <td class="text-muted">${r.test_bars ?? '—'}</td>
      <td class="text-muted">${tested}</td>
    </tr>`;
  }).join('');
  document.getElementById('wf-body').innerHTML = wfRows;
  const wfEmpty = document.getElementById('wf-empty');
  if (wfEmpty) wfEmpty.style.display = wfData.length ? 'none' : '';
  if (!wfData.length) document.getElementById('wf-body').innerHTML =
    '<tr><td colspan="8" class="text-center text-muted">No results yet — run Walk-Forward Test from Action Center</td></tr>';
  // ml-confidence-note removed from DOM — no-op
}

// ── Admin ─────────────────────────────────────────────────────────────────────
// ── Settings ──────────────────────────────────────────────────────────────────
async function loadSettings() {
  const s = await fetch('/api/settings').then(r => r.json());
  const keys = ['sentiment_provider','news_provider','sentiment_suppress_threshold',
                 'finnhub_api_key','gemini_api_key','claude_api_key','sentiment_local_url','news_lookback_hours',
                 'outcome_notify_admin','display_currency'];
  for (const k of keys) {
    const el = document.getElementById('set-' + k);
    if (el && s[k] !== undefined) el.value = s[k];
  }
  // Load sentiment cache
  const cache = await fetch('/api/sentiment').then(r => r.json());
  const labelColor = {positive:'#4ade80', negative:'#f87171', neutral:'#fbbf24', disabled:'#555'};
  document.getElementById('sentiment-status').innerHTML = cache.length
    ? `<div class="d-flex flex-wrap gap-2">${cache.map(c => {
        const color = labelColor[c.label] || '#8b8fa8';
        const age = c.computed_at ? Math.round((Date.now() - new Date(c.computed_at+'Z')) / 60000) + 'm ago' : '—';
        return `<div style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:4px 10px">
          <span class="sym-tag">${c.symbol}</span>
          <span style="color:${color};margin-left:6px">${c.label}</span>
          <span style="color:#555;font-size:.72rem;margin-left:4px">${c.score > 0 ? '+' : ''}${(c.score||0).toFixed(2)} · ${age}</span>
        </div>`;
      }).join('')}</div>`
    : '<span style="color:#555">No sentiment data yet — enable a provider and wait for next refresh cycle.</span>';
}

async function saveSettings() {
  const msg = document.getElementById('settings-msg');
  msg.style.color = '#8b8fa8';
  msg.textContent = 'Saving…';
  const keys = ['sentiment_provider','news_provider','sentiment_suppress_threshold',
                 'finnhub_api_key','gemini_api_key','claude_api_key','sentiment_local_url','news_lookback_hours',
                 'outcome_notify_admin','display_currency'];
  const payload = {};
  for (const k of keys) {
    const el = document.getElementById('set-' + k);
    if (el) payload[k] = el.value;
  }
  const res = await fetch('/api/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  }).then(r => r.json());
  if (res.ok) {
    msg.style.color = '#4ade80';
    msg.textContent = '✓ Saved — takes effect on next scan (no restart needed)';
    _loadFx();
  } else {
    msg.style.color = '#f87171';
    msg.textContent = res.error || 'Save failed';
  }
}

async function adminBootstrap() {
  const msg = document.getElementById('bootstrap-msg');
  msg.style.color = '#8b8fa8';
  msg.textContent = 'Queueing bootstrap…';
  const res = await fetch('/api/actions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'run_bootstrap'}),
  }).then(r => r.json());
  if (res.ok) {
    msg.style.color = '#a78bfa';
    msg.textContent = '✓ Bootstrap started — switch to 🔄 Learning Activity to watch progress';
    setTimeout(refreshJobs, 4000);
  } else {
    msg.style.color = '#f87171';
    msg.textContent = res.error || 'Failed to start bootstrap';
  }
}

async function refreshAdmin() {
  const [syms, stats] = await Promise.all([
    fetch('/api/symbols').then(r=>r.json()),
    fetch('/api/stats').then(r=>r.json()),
  ]);

  const list = document.getElementById('symbol-list');
  const marketLabels = {us:'🇺🇸 US', xetra:'🇩🇪 XETRA', crypto:'₿ Crypto'};
  list.innerHTML = syms.map(s => {
    const statusBadge = s.active
      ? '<span class="badge badge-active">Active</span>'
      : '<span class="badge badge-inactive">Inactive</span>';
    const toggleBtn = s.active
      ? `<button class="btn btn-del" style="background:#1a140a;color:#fbbf24;border-color:#fbbf2430" onclick="removeSymbol('${s.symbol}')" title="Deactivate">Deactivate</button>`
      : `<button class="btn btn-add" style="font-size:.72rem;padding:2px 10px" onclick="restoreSymbol('${s.symbol}')" title="Activate">Activate</button>`;
    const delBtn = `<button class="btn btn-del" onclick="hardDeleteSymbol('${s.symbol}')" title="Permanently delete">✕ Delete</button>`;
    const mkt = s.market || 'us';
    const mktLabel = {us:'🇺🇸 US', xetra:'🇩🇪 XETRA', crypto:'₿ Crypto'}[mkt] || mkt;
    const marketBadge = `<span style="font-size:.70rem;color:var(--dim);border:1px solid var(--border);border-radius:4px;padding:1px 6px">${mktLabel}</span>`;
    const rowOpacity = s.active ? '' : 'opacity:.6;';
    return `<div class="d-flex align-items-center mb-2 p-2 gap-2" style="${rowOpacity}background:var(--bg);border:1px solid var(--border);border-radius:6px;flex-wrap:wrap">
      <span class="sym-tag">${s.symbol}</span>
      ${statusBadge}
      ${marketBadge}
      <span class="ms-auto text-muted" style="font-size:.70rem">${s.added_at.slice(0,10)}</span>
      ${toggleBtn}
      ${delBtn}
    </div>`;
  }).join('') || '<div class="text-muted">No symbols configured yet.</div>';

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
        onmouseover="this.style.background='var(--surface2)'"
        onmouseout="this.style.background=''"
        onclick="selectSymbol('${item.symbol}')">
        <span class="sym-tag" style="min-width:70px">${item.symbol}</span>
        <span style="color:var(--text);font-size:.82rem">${item.name}</span>
        <span class="ms-auto text-muted" style="font-size:.72rem">${item.exchange} · ${item.type}</span>
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
  if (!confirm(`Deactivate ${sym}? It will stop being scanned but can be reactivated.`)) return;
  const res = await fetch(`/api/symbols/${sym}`, {method:'DELETE'}).then(r=>r.json());
  if (res.ok) refreshAdmin();
}

async function hardDeleteSymbol(sym) {
  if (!confirm(`Permanently delete ${sym}? The symbol will be removed from the watchlist entirely. Signal history is kept.`)) return;
  const res = await fetch(`/api/symbols/${sym}/delete`, {method:'DELETE'}).then(r=>r.json());
  if (res.ok) refreshAdmin();
  else alert(res.error || 'Delete failed');
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
  const symbol = document.getElementById('activity-symbol')?.value || '';
  const days = document.getElementById('activity-days')?.value || '0';
  const params = new URLSearchParams();
  if (symbol) params.set('symbol', symbol);
  if (days && days !== '0') params.set('days', days);
  const data = await fetch('/api/ml-activity' + (params.toString() ? `?${params}` : '')).then(r=>r.json());
  const symbolSelect = document.getElementById('activity-symbol');
  if (symbolSelect && symbolSelect.options.length <= 1) {
    const current = data.filters?.symbol || '';
    symbolSelect.innerHTML = '<option value="">All symbols</option>' +
      (data.available_symbols || []).map(sym =>
        `<option value="${sym}" ${sym === current ? 'selected' : ''}>${sym}</option>`
      ).join('');
  }
  const daysSelect = document.getElementById('activity-days');
  if (daysSelect && data.filters) {
    daysSelect.value = String(data.filters.days || 0);
    if (data.filters.symbol && symbolSelect) symbolSelect.value = data.filters.symbol;
  }

  // ── Model health cards with traffic lights ──
  const health = data.symbol_health || {};
  function modelHealthStatus(h) {
    const hasData = h.recent_count >= 3;
    const ageH = h.last_trained ? (Date.now() - new Date(h.last_trained+'Z')) / 3600000 : null;
    if (!h.last_trained && h.total_outcomes === 0) return {dot:'status-warn', label:'No model', color:'var(--yellow)'};
    if (!hasData) return {dot:'status-warn', label:'Accumulating', color:'var(--yellow)'};
    const wr = h.recent_acc;
    const stale = ageH !== null && ageH > 24;
    if (wr >= 52 && !stale) return {dot:'status-ok',   label:'Healthy',    color:'var(--green)'};
    if (wr < 42 || stale)  return {dot:'status-bad',  label:wr<42?'Degraded':'Stale', color:'var(--red)'};
    return {dot:'status-warn', label:'Watch',     color:'var(--yellow)'};
  }
  document.getElementById('health-cards').innerHTML =
    Object.entries(health).map(([sym, h]) => {
      const hasData = h.recent_count > 0;
      const accColor = !hasData ? '#555' : (h.recent_acc >= 60 ? '#4ade80' : (h.recent_acc >= 50 ? '#fbbf24' : '#f87171'));
      const barW = hasData ? Math.min(h.recent_acc, 100) : 0;
      const barCls = h.recent_acc >= 60 ? 'accuracy-bar-high' : (h.recent_acc >= 50 ? 'accuracy-bar-mid' : 'accuracy-bar-low');
      const accLabel = hasData
        ? `<strong style="color:${accColor}">${h.recent_acc}%</strong> <span style="color:#555">(${h.recent_count})</span>`
        : `<span style="color:#555">no live data yet</span>`;
      const hs = modelHealthStatus(h);
      return `<div class="col-6 col-md-4 col-lg-3 col-xl-2">
        <div class="health-card" style="border-left:3px solid ${hs.color}">
          <div class="d-flex justify-content-between align-items-center mb-1">
            <span class="sym-tag">${sym}</span>
            <span class="d-flex align-items-center gap-1" style="font-size:.68rem">
              <span class="status-dot ${hs.dot}"></span>
              <span style="color:${hs.color}">${hs.label}</span>
            </span>
          </div>
          <div class="progress mb-1" style="height:4px">
            <div class="progress-bar ${barCls}" style="width:${barW}%"></div>
          </div>
          <div style="font-size:.74rem;color:#8b8fa8">
            ${accLabel}
            ${hasData ? `<span style="color:#555"> · All: ${h.all_acc}%</span>` : ''}
            ${trendIcon(h.trend)}
          </div>
          <div class="mt-1" style="font-size:.70rem;color:#555">
            ${h.total_outcomes} outcomes · ${h.pending} pending<br>
            ${modelAge(h.last_trained)} · ${hasData ? `${h.recent_successes}✓ ${h.recent_failures}✗` : 'accumulating…'}
          </div>
        </div>
      </div>`;
    }).join('') || '<div class="col-12" style="color:#555">No training data yet — models train automatically once signals accumulate outcomes.</div>';

  // Health summary bar
  const healthEntries = Object.entries(health);
  const nHealthy = healthEntries.filter(([,h]) => modelHealthStatus(h).dot === 'status-ok').length;
  const nWatch   = healthEntries.filter(([,h]) => modelHealthStatus(h).dot === 'status-warn').length;
  const nBad     = healthEntries.filter(([,h]) => modelHealthStatus(h).dot === 'status-bad').length;
  const summaryEl = document.getElementById('health-summary-bar');
  if (summaryEl) {
    summaryEl.innerHTML = `
      <span><span class="status-dot status-ok"></span><span style="color:var(--green)">${nHealthy} Healthy</span></span>
      <span><span class="status-dot status-warn"></span><span style="color:var(--yellow)">${nWatch} Watching</span></span>
      <span><span class="status-dot status-bad"></span><span style="color:var(--red)">${nBad} Degraded/Stale</span></span>
      <span style="color:var(--muted)">${healthEntries.length} total symbols</span>`;
  }

  const reasonMap = {
    tp_hit: 'TP Hit',
    sl_hit: 'SL Hit',
    both_hit_same_candle: 'Both Same Candle',
    timeout_no_hit: 'Timed Out',
    threshold_up: 'Threshold Up',
    threshold_down: 'Threshold Down',
    threshold_flat: 'Threshold Flat',
    unknown: 'Unknown',
  };
  const reasonRows = data.resolution_reason_breakdown || [];
  const reasonLabels = reasonRows.map(r => reasonMap[r.reason] || r.reason);
  const reasonCounts = reasonRows.map(r => r.count);
  const reasonColors = ['#4ade80', '#f87171', '#fbbf24', '#38bdf8', '#a78bfa', '#fb923c', '#94a3b8'];
  if (!resolutionReasonsChart) {
    resolutionReasonsChart = new Chart(document.getElementById('chart-resolution-reasons'), {
      type: 'doughnut',
      data: {
        labels: reasonLabels,
        datasets: [{
          data: reasonCounts,
          backgroundColor: reasonColors.slice(0, Math.max(reasonLabels.length, 1)),
          borderColor: '#1a1d27',
          borderWidth: 3,
        }],
      },
      options: {maintainAspectRatio:false, plugins:{legend:{position:'right',labels:{color:'#8b8fa8',boxWidth:12,padding:10}}}}
    });
  } else {
    resolutionReasonsChart.data.labels = reasonLabels;
    resolutionReasonsChart.data.datasets[0].data = reasonCounts;
    resolutionReasonsChart.update();
  }

  const timeRows = data.resolution_time_by_symbol || [];
  const timeLabels = timeRows.map(r => r.symbol);
  const timeValues = timeRows.map(r => r.avg_resolution_minutes);
  if (!resolutionTimeChart) {
    resolutionTimeChart = new Chart(document.getElementById('chart-resolution-time'), {
      type:'bar',
      data:{labels:timeLabels, datasets:[{
        label:'Minutes',
        data:timeValues,
        backgroundColor:'#38bdf8aa',
        borderColor:'#38bdf8',
        borderWidth:1,
      }]},
      options:{plugins:{legend:{labels:{color:'#8b8fa8'}}},
        scales:{
          x:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}},
          y:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'},title:{display:true,text:'Minutes',color:'#8b8fa8'}}
        }}
    });
  } else {
    resolutionTimeChart.data.labels = timeLabels;
    resolutionTimeChart.data.datasets[0].data = timeValues;
    resolutionTimeChart.update();
  }

  const mfeMaeRows = data.mfe_mae_by_symbol || [];
  const mfeMaeLabels = mfeMaeRows.map(r => r.symbol);
  const mfeVals = mfeMaeRows.map(r => r.avg_favorable_pct ?? 0);
  const maeVals = mfeMaeRows.map(r => r.avg_adverse_pct ?? 0);
  if (!mfeMaeChart) {
    mfeMaeChart = new Chart(document.getElementById('chart-mfe-mae'), {
      type:'bar',
      data:{labels:mfeMaeLabels, datasets:[
        {label:'MFE %', data:mfeVals, backgroundColor:'#4ade80aa', borderColor:'#4ade80', borderWidth:1},
        {label:'MAE %', data:maeVals, backgroundColor:'#f87171aa', borderColor:'#f87171', borderWidth:1},
      ]},
      options:{plugins:{legend:{labels:{color:'#8b8fa8'}}},
        scales:{
          x:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'}},
          y:{ticks:{color:'#8b8fa8'},grid:{color:'#2a2d3a'},title:{display:true,text:'Percent move',color:'#8b8fa8'}}
        }}
    });
  } else {
    mfeMaeChart.data.labels = mfeMaeLabels;
    mfeMaeChart.data.datasets[0].data = mfeVals;
    mfeMaeChart.data.datasets[1].data = maeVals;
    mfeMaeChart.update();
  }

  const adaptiveRows = mfeMaeRows.filter(r => r.avg_favorable_pct !== null && r.avg_adverse_pct !== null);
  document.getElementById('adaptive-summary').innerHTML = adaptiveRows.length
    ? adaptiveRows.map(r => {
        const rr = r.avg_adverse_pct > 0 ? (r.avg_favorable_pct / r.avg_adverse_pct).toFixed(2) : '—';
        return `<div class="mb-2">
          <span class="sym-tag">${r.symbol}</span>
          <span style="color:#e0e0e0"> MFE ${Number(r.avg_favorable_pct).toFixed(2)}% </span>
          <span style="color:#8b8fa8">vs</span>
          <span style="color:#e0e0e0"> MAE ${Number(r.avg_adverse_pct).toFixed(2)}% </span>
          <span style="color:#fbbf24"> · live RR anchor ${rr}</span>
        </div>`;
      }).join('')
    : '<div style="color:#555">Not enough resolved outcome data yet for adaptive TP/SL guidance.</div>';

  // ── Training event log ──
  const trainEvents = data.training_events || [];
  const cycleEvents = data.cycle_events || [];
  const learningEvents = [
    ...trainEvents.map(e => ({kind:'train', at:e.trained_at, ...e})),
    ...cycleEvents.map(e => ({kind:'cycle', at:e.checked_at, ...e})),
  ].sort((a,b) => new Date(b.at+'Z') - new Date(a.at+'Z')).slice(0, 60);

  const triggerLabels = {
    bootstrap: {label:'🚀 bootstrap', color:'#a78bfa'},
    outcomes:  {label:'⚡ outcomes triggered', color:'#a78bfa'},
    time:      {label:'⏱ time-based', color:'#38bdf8'},
    manual:    {label:'🖱 manual', color:'#fbbf24'},
  };
  function triggerChip(e) {
    const t = e.trigger || (e.outcome_samples >= 3 ? 'outcomes' : 'time');
    const cfg = triggerLabels[t] || {label: t, color:'#8b8fa8'};
    return `<span style="color:${cfg.color}">${cfg.label}</span>`;
  }
  function winRateChip(wr) {
    if (wr === null || wr === undefined) return '';
    const color = wr >= 55 ? '#4ade80' : wr >= 50 ? '#fbbf24' : '#f87171';
    return `&nbsp;·&nbsp;<span style="color:${color}">win ${wr}%</span>`;
  }
  function sampleBreakdown(e) {
    if (!e.correct_count && !e.incorrect_count && !e.neutral_count) return '';
    return `&nbsp;·&nbsp;<span style="color:#4ade80">${e.correct_count||0}✓</span>` +
           `&nbsp;<span style="color:#f87171">${e.incorrect_count||0}✗</span>` +
           `&nbsp;<span style="color:#fbbf24">${e.neutral_count||0}~</span>`;
  }

  document.getElementById('train-log').innerHTML = learningEvents.length
    ? learningEvents.map(e => {
        if (e.kind === 'train') {
          const cls = (e.outcome_samples > 0 || e.trigger === 'bootstrap') ? 'has-outcomes' : '';
          return `<div class="event-row ${cls}">
            <div class="d-flex justify-content-between">
              <strong style="color:#e0e0e0">${e.symbol}</strong>
              <span style="color:#555">${new Date(e.trained_at+'Z').toLocaleString()}</span>
            </div>
            <div class="mt-1">
              <span style="color:#4ade80">🧠 retrained</span> &nbsp;·&nbsp;
              ${triggerChip(e)}
              ${winRateChip(e.win_rate)}
              ${sampleBreakdown(e)}
              &nbsp;·&nbsp;<span style="color:#8b8fa8">${e.train_samples} samples</span>
              ${e.outcome_samples > 0 ? `&nbsp;·&nbsp;<span style="color:#fbbf24">${e.outcome_samples} real outcomes blended</span>` : ''}
            </div>
          </div>`;
        }
        const cycleCls = e.retrained ? 'has-outcomes' : '';
        const needed = e.retrain_needed ? '<span style="color:#fbbf24">needs retrain</span>' : '<span style="color:#38bdf8">check only</span>';
        const result = e.retrained ? '<span style="color:#4ade80">retrained</span>' : '<span style="color:#8b8fa8">no retrain</span>';
        return `<div class="event-row ${cycleCls}">
          <div class="d-flex justify-content-between">
            <strong style="color:#e0e0e0">${e.symbol}</strong>
            <span style="color:#555">${new Date(e.checked_at+'Z').toLocaleString()}</span>
          </div>
          <div class="mt-1">
            ${needed} &nbsp;·&nbsp; ${result} &nbsp;·&nbsp;
            <span style="color:#8b8fa8">${e.resolved_outcomes} resolved</span>
            &nbsp;·&nbsp;<span style="color:#a78bfa">${e.new_outcomes} new since last train</span>
            ${e.note ? `&nbsp;·&nbsp;<span style="color:#555">${e.note}</span>` : ''}
          </div>
        </div>`;
      }).join('')
    : '<div style="color:#555;font-size:.85rem">No learning-cycle events yet. Once the bot is running, this log updates every 20 minutes even when it only checks and decides not to retrain.</div>';

  // ── Outcome feed ──
  const outcomes = data.recent_outcomes || [];
  document.getElementById('outcome-feed').innerHTML = outcomes.length
    ? outcomes.map(o => {
        const cls = `outcome-${o.outcome}`;
        const emoji = o.outcome === 'correct' ? '✅' : o.outcome === 'incorrect' ? '❌' : '➖';
        const conf = o.ai_confidence ? ` · AI ${(o.ai_confidence*100).toFixed(0)}%` : '';
        const when = o.outcome_at ? new Date(o.outcome_at+'Z').toLocaleString() : '—';
        const reason = o.resolution_reason ? (reasonMap[o.resolution_reason] || o.resolution_reason) : '—';
        const travel = [];
        if (o.max_favorable_pct !== null && o.max_favorable_pct !== undefined) {
          travel.push(`MFE +${Number(o.max_favorable_pct).toFixed(2)}%`);
        }
        if (o.max_adverse_pct !== null && o.max_adverse_pct !== undefined) {
          travel.push(`MAE ${Number(o.max_adverse_pct).toFixed(2)}%`);
        }
        if (o.resolution_minutes !== null && o.resolution_minutes !== undefined) {
          travel.push(`${Number(o.resolution_minutes).toFixed(0)} min`);
        }
        return `<div class="outcome-row ${cls}">
          <div class="d-flex justify-content-between">
            <span><strong>${o.symbol}</strong> ${badge(o.action,'badge-'+o.action)} ${badge(o.strength||'RULE','badge-'+(o.strength||'RULE'))}</span>
            <span style="color:#555;font-size:.72rem">${when}</span>
          </div>
          <div style="color:#8b8fa8;margin-top:3px">
            ${emoji} <strong style="color:#e0e0e0">${o.outcome.toUpperCase()}</strong>
            · Entry $${o.price ? o.price.toFixed(2) : '—'}${conf}
          </div>
          <div style="color:#555;margin-top:3px;font-size:.74rem">
            ${reason}${travel.length ? ' · ' + travel.join(' · ') : ''}
          </div>
        </div>`;
      }).join('')
    : '<div style="color:#555;font-size:.85rem">No resolved outcomes yet. Outcomes are checked every 30 minutes — signals need at least 1 hour to resolve.</div>';

  // ── Suggestions ──
  const sugs = [];
  for (const [sym, h] of Object.entries(health)) {
    if (h.total_outcomes === 0) {
      sugs.push({cls:'sug-info', icon:'ℹ️',
        text:`<strong>${sym}</strong>: no outcomes resolved yet — model has no validated success/failure feedback yet, so suggestions are still provisional`});
    } else if (h.recent_acc < 45 && h.recent_count >= 5) {
      sugs.push({cls:'sug-warn', icon:'⚠️',
        text:`<strong>${sym}</strong>: recent accuracy ${h.recent_acc}% is below 45% on last ${h.recent_count} validated outcomes (${h.recent_successes} success / ${h.recent_failures} failure) — model may be overfitting or market conditions changed`});
    } else if (h.recent_unclear >= 3) {
      sugs.push({cls:'sug-warn', icon:'🌀',
        text:`<strong>${sym}</strong>: ${h.recent_unclear} of the last ${h.recent_count} validated outcomes were ambiguous or timed out — this suggestion is based on observed label quality, not just a static rule`});
    } else if (h.ambiguous_count >= 3) {
      sugs.push({cls:'sug-info', icon:'↔️',
        text:`<strong>${sym}</strong>: ${h.ambiguous_count} outcomes touched TP and SL in the same candle — consider wider stops or a higher timeframe for clearer labeling`});
    } else if (h.timeout_count >= 3) {
      sugs.push({cls:'sug-info', icon:'⌛',
        text:`<strong>${sym}</strong>: ${h.timeout_count} resolved outcomes timed out without hitting TP/SL — targets may be too far or the outcome window too short`});
    } else if (h.trend === 'declining') {
      sugs.push({cls:'sug-warn', icon:'📉',
        text:`<strong>${sym}</strong>: accuracy declining (recent ${h.recent_acc}% vs all-time ${h.all_acc}%) — more outcomes are needed to retrain`});
    } else if (h.trend === 'improving') {
      sugs.push({cls:'sug-ok', icon:'📈',
        text:`<strong>${sym}</strong>: accuracy improving! Recent ${h.recent_acc}% vs all-time ${h.all_acc}% (${h.recent_successes} validated wins recently) — continuous learning is working`});
    } else if (h.avg_resolution_minutes !== null && h.avg_resolution_minutes > 720) {
      sugs.push({cls:'sug-info', icon:'🕒',
        text:`<strong>${sym}</strong>: outcomes take ${Math.round(h.avg_resolution_minutes)} minutes on average — consider a longer signal horizon or slower scan framing`});
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

// ── Training jobs ─────────────────────────────────────────────────────────────
let _jobPollTimer = null;

async function refreshJobs() {
  const data = await fetch('/api/training-jobs').then(r=>r.json());
  const running = data.running;
  const recent = data.recent || [];

  // Live progress widget
  const row = document.getElementById('job-progress-row');
  if (running) {
    row.style.display = '';
    const pct = running.total_symbols > 0
      ? Math.round(running.done_symbols / running.total_symbols * 100) : 0;
    document.getElementById('job-progress-bar').style.width = pct + '%';
    document.getElementById('job-progress-pct').textContent = `${running.done_symbols}/${running.total_symbols} symbols`;
    const curSym = running.current_symbol ? ` — processing ${running.current_symbol}` : '';
    document.getElementById('job-progress-label').textContent = `${running.job_type} running${curSym}`;
    // Poll fast while running
    if (!_jobPollTimer) {
      _jobPollTimer = setInterval(refreshJobs, 5000);
    }
  } else {
    row.style.display = 'none';
    if (_jobPollTimer) { clearInterval(_jobPollTimer); _jobPollTimer = null; }
  }

  // Job history table
  const jobColors = {done:'#4ade80', failed:'#f87171', running:'#a78bfa'};
  document.getElementById('job-history').innerHTML = recent.length
    ? recent.map(j => {
        const color = jobColors[j.status] || '#8b8fa8';
        const duration = j.completed_at && j.started_at
          ? Math.round((new Date(j.completed_at+'Z') - new Date(j.started_at+'Z')) / 1000)
          : null;
        let summary = '';
        try {
          const s = j.result_summary ? JSON.parse(j.result_summary) : null;
          if (s) summary = ` &nbsp;·&nbsp; <span style="color:#e0e0e0">${s.trained}/${s.total} trained</span> &nbsp;·&nbsp; <span style="color:#fbbf24">avg win ${s.avg_win_rate}%</span>`;
        } catch(e){}
        return `<div class="event-row" style="border-left-color:${color}">
          <div class="d-flex justify-content-between">
            <span><span style="color:${color};font-weight:600">${j.job_type}</span>
              &nbsp;<span style="color:#8b8fa8">${j.done_symbols}/${j.total_symbols} symbols</span>
              ${summary}
            </span>
            <span style="color:#555;font-size:.75rem">${new Date(j.started_at+'Z').toLocaleString()}</span>
          </div>
          <div style="color:#555;margin-top:3px;font-size:.75rem">
            ${j.note || ''}
            ${duration !== null ? `&nbsp;·&nbsp;${duration}s` : ''}
          </div>
        </div>`;
      }).join('')
    : '<div style="color:#555;font-size:.85rem">No training jobs run yet. Click "Run Bootstrap" to pre-train models from 2 years of historical data.</div>';
}

// ── Event Timeline ────────────────────────────────────────────────────────────
async function refreshTimeline() {
  const types = [];
  if (document.getElementById('tl-type-signal')?.checked)  types.push('signal');
  if (document.getElementById('tl-type-outcome')?.checked) types.push('outcome');
  if (document.getElementById('tl-type-train')?.checked)   types.push('train');
  if (document.getElementById('tl-type-scan')?.checked)    types.push('scan');
  if (document.getElementById('tl-type-action')?.checked)  types.push('action');
  const limit = document.getElementById('tl-limit')?.value || '100';
  const params = new URLSearchParams({limit, types: types.join(',')});
  const events = await fetch('/api/timeline?'+params).then(r=>r.json());

  const container = document.getElementById('timeline-feed');
  if (!events.length) {
    container.innerHTML = '<div class="text-muted text-center py-4">No events yet.</div>';
    return;
  }

  const typeConfig = {
    signal:  {icon:'📡', color:'var(--accent)',  label:'Signal'},
    outcome: {icon:'📬', color:'var(--green)',    label:'Outcome'},
    train:   {icon:'🧠', color:'var(--purple)',   label:'Train'},
    cycle:   {icon:'🔄', color:'var(--muted)',    label:'Cycle'},
    action:  {icon:'🎛',  color:'var(--yellow)',  label:'Action'},
    scan:    {icon:'🔍', color:'var(--dim)',       label:'Scan'},
  };

  const reasonMap = {
    tp_hit:'TP Hit', sl_hit:'SL Hit', both_hit_same_candle:'Both Hit',
    timeout_no_hit:'Timed Out', threshold_up:'Price Up', threshold_down:'Price Down',
    threshold_flat:'Flat', unknown:'Unknown',
  };

  container.innerHTML = events.map(e => {
    const cfg = typeConfig[e.type] || {icon:'•', color:'var(--muted)', label: e.type};
    const tsStr = e.ts ? new Date(e.ts+'Z').toLocaleString() : '—';
    let detail = '';

    if (e.type === 'signal') {
      const actColor = e.action === 'BUY' ? 'var(--accent)' : 'var(--pink)';
      const outcomeEmoji = {correct:'✅',incorrect:'❌',neutral:'➖',pending:'⏳'}[e.outcome]||'';
      detail = `<span style="color:${actColor};font-weight:600">${e.action}</span>
        <span class="sym-tag ms-1">${e.symbol}</span>
        <span class="ms-2" style="color:var(--muted);font-size:.72rem">${e.strength||'RULE'}</span>
        ${e.ai_confidence ? `<span class="ms-2" style="color:var(--purple);font-size:.72rem">AI ${(e.ai_confidence*100).toFixed(0)}%</span>` : ''}
        ${outcomeEmoji ? `<span class="ms-2">${outcomeEmoji}</span>` : ''}
        ${e.regime ? `<span class="ms-2" style="color:var(--muted);font-size:.72rem">${e.regime}</span>` : ''}`;
    } else if (e.type === 'outcome') {
      const outcomeColor = {correct:'var(--green)',incorrect:'var(--red)',neutral:'var(--yellow)'}[e.outcome]||'var(--muted)';
      const reason = reasonMap[e.resolution_reason] || (e.resolution_reason||'');
      detail = `<span class="sym-tag">${e.symbol}</span>
        <span style="color:${outcomeColor};font-weight:600;margin-left:8px">${(e.outcome||'').toUpperCase()}</span>
        <span style="color:var(--muted);font-size:.72rem;margin-left:8px">${reason}</span>
        ${e.max_favorable_pct != null ? `<span style="color:var(--green);font-size:.72rem;margin-left:8px">MFE +${Number(e.max_favorable_pct).toFixed(2)}%</span>` : ''}
        ${e.resolution_minutes != null ? `<span style="color:var(--dim);font-size:.72rem;margin-left:8px">${Math.round(e.resolution_minutes)}min</span>` : ''}`;
    } else if (e.type === 'train') {
      const wrColor = e.win_rate >= 55 ? 'var(--green)' : e.win_rate >= 50 ? 'var(--yellow)' : 'var(--red)';
      const triggerLabels = {bootstrap:'🚀',outcomes:'⚡',time:'⏱',manual:'🖱'};
      detail = `${triggerLabels[e.trigger]||'🔄'} <span class="sym-tag">${e.symbol}</span>
        <span style="color:var(--muted);font-size:.72rem;margin-left:8px">${e.train_samples} samples</span>
        ${e.win_rate != null ? `<span style="color:${wrColor};font-size:.72rem;margin-left:8px">WR ${e.win_rate}%</span>` : ''}
        ${e.outcome_samples ? `<span style="color:var(--yellow);font-size:.72rem;margin-left:8px">${e.outcome_samples} outcomes blended</span>` : ''}`;
    } else if (e.type === 'scan') {
      const sigColor = e.signals_generated > 0 ? 'var(--green)' : 'var(--muted)';
      const dur = e.scan_duration_ms != null ? (e.scan_duration_ms > 1000 ? (e.scan_duration_ms/1000).toFixed(1)+'s' : Math.round(e.scan_duration_ms)+'ms') : '';
      detail = `<span style="color:var(--muted)">${e.symbols_scanned||0}/${e.symbols_total||0} symbols</span>
        <span style="color:${sigColor};font-weight:600;margin-left:8px">${e.signals_generated||0} signals</span>
        ${e.symbols_skipped_market ? `<span style="color:var(--dim);font-size:.72rem;margin-left:8px">${e.symbols_skipped_market} skipped</span>` : ''}
        ${dur ? `<span style="color:var(--dim);font-size:.72rem;margin-left:8px">${dur}</span>` : ''}
        ${e.regime ? `<span style="color:var(--muted);font-size:.72rem;margin-left:8px">${e.regime}</span>` : ''}`;
    } else if (e.type === 'action') {
      const statusColor = {completed:'var(--green)',failed:'var(--red)',pending:'var(--yellow)'}[e.status]||'var(--muted)';
      detail = `<span style="color:var(--text)">${e.action}</span>
        ${e.symbol ? `<span class="sym-tag ms-1">${e.symbol}</span>` : ''}
        <span style="color:${statusColor};font-size:.72rem;margin-left:8px">${e.status}</span>
        ${e.result_note ? `<span style="color:var(--dim);font-size:.72rem;margin-left:8px">${e.result_note.slice(0,60)}</span>` : ''}`;
    }

    return `<div class="event-row mb-1 d-flex align-items-start gap-2" style="border-left-color:${cfg.color}">
      <span style="font-size:.88rem;flex-shrink:0">${cfg.icon}</span>
      <div style="flex:1;min-width:0">
        <span style="color:${cfg.color};font-size:.70rem;text-transform:uppercase;letter-spacing:.07em;font-weight:600">${cfg.label}</span>
        <span style="margin-left:8px;font-size:.78rem">${detail}</span>
      </div>
      <span style="color:var(--dim);font-size:.70rem;flex-shrink:0;white-space:nowrap">${tsStr}</span>
    </div>`;
  }).join('');
}

// ── Theme ─────────────────────────────────────────────────────────────────────
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('tsTheme', next);
  document.getElementById('theme-toggle').textContent = next === 'dark' ? '☀️' : '🌙';
}
(function initThemeIcon(){
  const t = document.documentElement.getAttribute('data-theme') || 'dark';
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.textContent = t === 'dark' ? '☀️' : '🌙';
})();

// ── Init ──────────────────────────────────────────────────────────────────────
refreshCommandCenter();
refreshOverview();
updateMarketStatus();
setInterval(refreshCommandCenter, 60000);
setInterval(refreshOverview, 30000);
setInterval(refreshOps, 30000);
setInterval(updateMarketStatus, 60000);

document.querySelectorAll('[data-bs-target="#tab-command"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshCommandCenter())
);
document.querySelectorAll('[data-bs-target="#tab-overview"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshOverview())
);
document.querySelectorAll('[data-bs-target="#tab-signals"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => { _allSignals = []; refreshSignalsTab(); })
);
document.querySelectorAll('[data-bs-target="#tab-timeline"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshTimeline())
);
document.querySelectorAll('[data-bs-target="#tab-ml"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshML())
);
document.querySelectorAll('[data-bs-target="#tab-activity"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => { refreshActivity(); refreshJobs(); })
);
document.querySelectorAll('[data-bs-target="#lt-analytics"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => {
    [resolutionReasonsChart, resolutionTimeChart, mfeMaeChart].forEach(c => c && c.resize());
  })
);
document.querySelectorAll('[data-bs-target="#tab-admin"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => { refreshAdmin(); loadSettings(); })
);
document.querySelectorAll('[data-bs-target="#tab-performance"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshPerformance())
);

// ── Performance tab ──────────────────────────────────────────────────────────
let equityChartInst = null, wrChartInst = null;

function wrColor(wr) {
  if (wr >= 55) return '#4ade80';
  if (wr >= 47) return '#facc15';
  return '#f87171';
}

async function refreshPerformance() {
  const [perf, eq] = await Promise.all([
    fetch('/api/performance?days=90').then(r => r.json()),
    fetch('/api/equity-curve?size=1000').then(r => r.json()),
  ]);

  // Equity curve
  const eqLabels = eq.curve.map(p => p.date);
  const eqValues = eq.curve.map(p => p.equity);
  const ddValues = eq.curve.map(p => -p.drawdown_pct);

  const eqCtx = document.getElementById('equityChart').getContext('2d');
  if (equityChartInst) equityChartInst.destroy();
  equityChartInst = new Chart(eqCtx, {
    type: 'line',
    data: {
      labels: eqLabels,
      datasets: [
        {
          label: 'P&L ($)',
          data: eqValues,
          borderColor: '#818cf8',
          backgroundColor: 'rgba(129,140,248,0.08)',
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          yAxisID: 'y',
        },
        {
          label: 'Drawdown (%)',
          data: ddValues,
          borderColor: '#f87171',
          backgroundColor: 'rgba(248,113,113,0.06)',
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          yAxisID: 'y2',
        },
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#94a3b8', boxWidth: 12 } } },
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 10 }, grid: { color: '#1e293b' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' }, title: { display: true, text: 'P&L ($)', color: '#64748b' } },
        y2: { position: 'right', ticks: { color: '#f87171' }, grid: { drawOnChartArea: false }, title: { display: true, text: 'Drawdown (%)', color: '#f87171' } },
      }
    }
  });

  const sign = eq.current >= 0 ? '+' : '';
  const ddCls = eq.max_drawdown_pct >= 5 ? 'text-danger' : 'text-success';
  document.getElementById('eq-summary').innerHTML =
    `P&L: <strong>${sign}$${eq.current.toFixed(0)}</strong> &nbsp;|&nbsp; ` +
    `Trades: ${eq.total_trades} &nbsp;|&nbsp; ` +
    `WR: ${eq.win_rate}% &nbsp;|&nbsp; ` +
    `<span class="${ddCls}">Max DD: ${eq.max_drawdown_pct.toFixed(1)}%</span>`;

  // Rolling win rate chart
  const daily = perf.daily;
  const wrCtx = document.getElementById('wrChart').getContext('2d');
  if (wrChartInst) wrChartInst.destroy();
  wrChartInst = new Chart(wrCtx, {
    type: 'line',
    data: {
      labels: daily.map(d => d.date),
      datasets: [
        {
          label: '7-day WR',
          data: daily.map(d => d.win_rate_7d),
          borderColor: '#818cf8',
          tension: 0.3,
          pointRadius: 0,
          fill: false,
        },
        {
          label: '30-day WR',
          data: daily.map(d => d.win_rate_30d),
          borderColor: '#34d399',
          tension: 0.3,
          pointRadius: 0,
          borderDash: [4, 3],
          fill: false,
        },
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#94a3b8', boxWidth: 12 } },
        annotation: {
          annotations: {
            breakeven: {
              type: 'line', yMin: 37.5, yMax: 37.5,
              borderColor: 'rgba(248,113,113,0.5)', borderWidth: 1,
              borderDash: [4, 4],
              label: { content: 'breakeven', enabled: true, color: '#f87171', font: { size: 9 } }
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 10 }, grid: { color: '#1e293b' } },
        y: { min: 0, max: 100, ticks: { color: '#94a3b8', callback: v => v+'%' }, grid: { color: '#1e293b' } }
      }
    }
  });

  // By strength table
  const stBody = document.querySelector('#strengthTable tbody');
  stBody.innerHTML = '';
  const strOrder = ['STRONG', 'RULE', 'AI'];
  strOrder.forEach(s => {
    const d = (perf.by_strength || {})[s];
    if (!d) return;
    const wr = d.win_rate;
    stBody.innerHTML += `<tr>
      <td><span class="badge" style="background:${s==='STRONG'?'#4ade80':s==='RULE'?'#818cf8':'#f59e0b'};color:#0f172a">${s}</span></td>
      <td>${d.total}</td><td>${d.correct}</td>
      <td><strong style="color:${wrColor(wr)}">${wr}%</strong></td>
    </tr>`;
  });
  if (!stBody.innerHTML) stBody.innerHTML = '<tr><td colspan="4" class="text-muted text-center">No resolved signals yet</td></tr>';

  // By symbol table
  const symBody = document.querySelector('#symPerfTable tbody');
  symBody.innerHTML = '';
  const syms = Object.entries(perf.by_symbol || {}).sort((a, b) => b[1].total - a[1].total);
  syms.forEach(([sym, d]) => {
    const wr = d.win_rate;
    symBody.innerHTML += `<tr>
      <td><strong>${sym}</strong></td>
      <td>${d.total}</td><td>${d.correct}</td>
      <td><strong style="color:${wrColor(wr)}">${wr}%</strong></td>
    </tr>`;
  });
  if (!symBody.innerHTML) symBody.innerHTML = '<tr><td colspan="4" class="text-muted text-center">No data yet</td></tr>';
}
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
    return jsonify(get_recent_signals(100, include_all_fields=True))


@app.route("/api/ml-stats")
def api_ml_stats():
    return jsonify(get_ml_accuracy_stats())


@app.route("/api/ml-activity")
def api_ml_activity():
    symbol = (request.args.get("symbol") or "").strip().upper() or None
    days_raw = (request.args.get("days") or "").strip()
    try:
        days = int(days_raw) if days_raw else None
    except ValueError:
        days = None
    return jsonify(get_ml_activity_log(symbol=symbol, days=days))


@app.route("/api/ops")
def api_ops():
    return jsonify(get_ops_snapshot())


@app.route("/api/actions", methods=["POST"])
def api_actions():
    data = request.get_json(force=True)
    action = (data.get("action") or "").strip()
    symbol = (data.get("symbol") or "").strip().upper() or None
    allowed = {"scan_now", "check_outcomes_now", "retrain_all", "retrain_symbol", "run_bootstrap", "run_walk_forward"}
    if action not in allowed:
        return jsonify({"ok": False, "error": "Unsupported action"}), 400
    if action == "retrain_symbol" and not symbol:
        return jsonify({"ok": False, "error": "Symbol required"}), 400
    create_action_request(action, symbol=symbol, requested_by="dashboard")
    return jsonify({"ok": True})


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


@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    return jsonify(get_all_settings())


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    data = request.get_json(force=True)
    allowed_keys = {
        "sentiment_provider", "sentiment_local_url", "claude_api_key", "gemini_api_key",
        "sentiment_suppress_threshold", "news_provider", "finnhub_api_key",
        "outcome_notify_admin", "news_lookback_hours",
        "display_currency",
    }
    for key, value in data.items():
        if key in allowed_keys:
            set_setting(key, str(value).strip())
    return jsonify({"ok": True})


@app.route("/api/sentiment")
def api_sentiment():
    return jsonify(get_all_sentiment_cache())


@app.route("/api/training-jobs")
def api_training_jobs():
    return jsonify({
        "running": get_running_training_job(),
        "recent": get_recent_training_jobs(10),
    })


@app.route("/api/symbols/<symbol>", methods=["DELETE"])
def api_symbols_remove(symbol):
    removed = remove_symbol(symbol.upper())
    if removed:
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Symbol not found"}), 404


@app.route("/api/symbols/<symbol>/delete", methods=["DELETE"])
def api_symbols_delete(symbol):
    deleted = delete_symbol(symbol.upper())
    if deleted:
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Symbol not found"}), 404


@app.route("/api/fx-rate")
def api_fx_rate():
    currency = get_all_settings().get("display_currency", "USD").upper()
    _symbols = {"USD": "$", "EUR": "€", "GBP": "£", "CHF": "Fr"}
    symbol = _symbols.get(currency, currency + " ")
    if currency == "USD":
        return jsonify({"currency": currency, "symbol": symbol, "rate": 1.0})
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"USD{currency}=X")
        rate = ticker.fast_info.last_price
        if rate and rate > 0:
            return jsonify({"currency": currency, "symbol": symbol, "rate": round(rate, 6)})
    except Exception:
        pass
    return jsonify({"currency": currency, "symbol": symbol, "rate": 1.0})


@app.route("/api/accuracy-direction")
def api_accuracy_direction():
    return jsonify(get_accuracy_by_direction())


@app.route("/api/wf-results")
def api_wf_results():
    return jsonify(get_wf_results())


@app.route("/api/performance")
def api_performance():
    days = int(request.args.get("days", 30))
    return jsonify(get_rolling_performance(days=days))


@app.route("/api/equity-curve")
def api_equity_curve():
    trade_size = float(request.args.get("size", 1000))
    return jsonify(get_equity_curve(trade_size=trade_size))


@app.route("/api/regime")
def api_regime():
    try:
        from bot import get_market_regime
        from ml_signals import get_predict_market_context
        spy_df, vix_df = get_predict_market_context()
        regime = get_market_regime(spy_df, vix_df)
    except Exception:
        regime = "unknown"
    attribution = get_regime_attribution(days=90)
    return jsonify({"regime": regime, "attribution": attribution})


@app.route("/api/symbols/<symbol>/market", methods=["POST"])
def api_symbols_set_market(symbol):
    data = request.get_json(force=True)
    market = (data.get("market") or "").strip().lower() or None
    if market and market not in ("us", "xetra", "crypto"):
        return jsonify({"ok": False, "error": "market must be us, xetra, or crypto"}), 400
    set_symbol_market(symbol.upper(), market)
    return jsonify({"ok": True})


@app.route("/api/system/status")
def api_system_status():
    """Command Center: scan history, container health, last scan outcome."""
    import urllib.request as _ur
    last_scan = get_last_scan()
    recent_scans = get_recent_scans(50)
    running_job = get_running_training_job()

    # FinBERT health check
    settings = get_all_settings()
    finbert_ok = False
    if settings.get("sentiment_provider") == "local_finbert":
        try:
            url = settings.get("sentiment_local_url", "http://finbert:5001").rstrip("/") + "/health"
            req = _ur.Request(url)
            with _ur.urlopen(req, timeout=2) as r:
                finbert_ok = r.status == 200
        except Exception:
            finbert_ok = False

    return jsonify({
        "last_scan": last_scan,
        "recent_scans": recent_scans,
        "running_job": running_job,
        "finbert_ok": finbert_ok,
        "perf_brake": False,  # reading from module state not possible cross-process; placeholder
    })


@app.route("/api/pipeline/<symbol>")
def api_pipeline_trace(symbol):
    limit = int(request.args.get("limit", 20))
    return jsonify(get_pipeline_trace(symbol.upper(), limit=limit))


@app.route("/api/scan-log")
def api_scan_log():
    limit = int(request.args.get("limit", 50))
    return jsonify(get_recent_scans(limit))


@app.route("/api/timeline")
def api_timeline():
    limit = int(request.args.get("limit", 100))
    types_raw = request.args.get("types", "")
    types = [t.strip() for t in types_raw.split(",") if t.strip()] or None
    return jsonify(get_event_timeline(limit=limit, event_types=types))


if __name__ == "__main__":
    init_db()
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "5000"))
    app.run(host=host, port=port)
