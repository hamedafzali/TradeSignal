import os
import urllib.request
import urllib.parse
import json as _json
from flask import Flask, jsonify, render_template_string, request
from database import (
    init_db, get_stats, get_recent_signals,
    get_ml_accuracy_stats, get_ml_activity_log,
    get_all_symbols_with_status, add_symbol, remove_symbol,
    create_action_request, get_ops_snapshot,
    get_running_training_job, get_recent_training_jobs,
    get_all_settings, set_setting, get_all_sentiment_cache,
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
  .ops-card { background:#0f1117; border:1px solid #2a2d3a; border-radius:10px; padding:14px 16px; min-height:112px; }
  .ops-label { color:#8b8fa8; font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; }
  .ops-value { color:#ffffff; font-size:1.4rem; font-weight:700; margin-top:4px; }
  .ops-meta { color:#555; font-size:.75rem; margin-top:8px; line-height:1.5; }
  .action-btn { width:100%; text-align:left; background:#0f1117; color:#e0e0e0; border:1px solid #2a2d3a; }
  .action-btn:hover { background:#141824; color:#fff; border-color:#38bdf8; }
  .status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
  .status-ok { background:#4ade80; }
  .status-warn { background:#fbbf24; }
  .status-bad { background:#f87171; }
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
      <div class="col-md-3"><div class="ops-card"><div class="ops-label">Signal Engine</div><div class="ops-value" id="ops-signal-status">—</div><div class="ops-meta" id="ops-last-signal">No signal activity yet</div></div></div>
      <div class="col-md-3"><div class="ops-card"><div class="ops-label">Learning Engine</div><div class="ops-value" id="ops-learning-status">—</div><div class="ops-meta" id="ops-last-learning">No learning cycles yet</div></div></div>
      <div class="col-md-3"><div class="ops-card"><div class="ops-label">Pending Work</div><div class="ops-value" id="ops-pending-work">—</div><div class="ops-meta" id="ops-pending-meta">No queued actions</div></div></div>
      <div class="col-md-3"><div class="ops-card"><div class="ops-label">Coverage</div><div class="ops-value" id="ops-coverage">—</div><div class="ops-meta" id="ops-coverage-meta">No watched symbols yet</div></div></div>
    </div>

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
      <div class="col-6 col-md-3"><div class="ops-card"><div class="ops-label">Active Models</div><div class="ops-value" id="ml-active-models">—</div><div class="ops-meta">Configured symbols in the learning universe</div></div></div>
      <div class="col-6 col-md-3"><div class="ops-card"><div class="ops-label">Trained Models</div><div class="ops-value" id="ml-trained-models">—</div><div class="ops-meta">Models with at least one successful training run</div></div></div>
      <div class="col-6 col-md-3"><div class="ops-card"><div class="ops-label">AI Resolved Signals</div><div class="ops-value" id="ml-resolved-ai">—</div><div class="ops-meta">Resolved outcomes where AI contributed confidence</div></div></div>
      <div class="col-6 col-md-3"><div class="ops-card"><div class="ops-label">Avg AI Accuracy</div><div class="ops-value" id="ml-avg-acc">—</div><div class="ops-meta">Cross-symbol mean of available AI accuracy</div></div></div>
    </div>

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
          <h6 class="mb-3">Training Runs by Symbol</h6>
          <canvas id="chart-train-runs" height="200"></canvas>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-1">How Training Works</h6>
          <p class="small mt-2" style="color:#8b8fa8; line-height:1.7">
            <strong style="color:#4ade80">Every 10 minutes</strong>, the bot checks each symbol for new resolved outcomes.<br>
            As soon as <strong style="color:#fbbf24">3 new outcomes</strong> arrive since the last training run,
            the model retrains immediately — no waiting.<br>
            As a fallback, it retrains every <strong style="color:#38bdf8">2 hours</strong> even without new outcomes.<br>
            <strong style="color:#fbbf24">Confirmed outcomes</strong> are injected back at
            <strong style="color:#a78bfa">3× weight</strong> so the model learns from its real mistakes.<br>
            <strong style="color:#38bdf8">Gradient Boosting</strong> runs two classifiers — one for BUY, one for SELL —
            each outputting a probability from 0 to 1.<br>
            A signal fires when confidence exceeds <strong style="color:#fbbf24">65%</strong>.
            STRONG signals require both the rule engine and AI to agree.
          </p>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h6 class="mb-3">Confidence Notes</h6>
          <div style="color:#8b8fa8;font-size:.84rem;line-height:1.7">
            AI confidence is a model probability, not a guarantee. If it looks low, the better fix is more resolved symbol-specific outcomes and better calibration, not artificially inflating the number.
            <div class="mt-2" id="ml-confidence-note">Confidence details will appear once data is available.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── LEARNING ACTIVITY TAB ───────────────────────────────────────── -->
  <div class="tab-pane fade" id="tab-activity">

    <div class="row g-3 mb-4">
      <div class="col-md-5">
        <div class="card p-3 h-100">
          <h6 class="mb-3">🎛 Action Center</h6>
          <div class="row g-2">
            <div class="col-md-6"><button class="btn action-btn" onclick="runAction('scan_now')">Scan Market Now</button></div>
            <div class="col-md-6"><button class="btn action-btn" onclick="runAction('check_outcomes_now')">Check Outcomes Now</button></div>
            <div class="col-md-6"><button class="btn action-btn" onclick="runAction('retrain_all')">Retrain All Models</button></div>
            <div class="col-md-6"><button class="btn action-btn" style="border-color:#a78bfa40;color:#a78bfa" onclick="runAction('run_bootstrap')">🚀 Run Bootstrap</button></div>
            <div class="col-12 d-flex gap-2">
              <select id="action-symbol" class="form-control" style="max-width:200px">
                <option value="">Pick symbol</option>
              </select>
              <button class="btn action-btn" onclick="runSymbolRetrain()">Retrain Symbol</button>
            </div>
          </div>
          <div id="action-msg" class="small mt-3" style="min-height:1.2em;color:#8b8fa8"></div>
        </div>
      </div>
      <div class="col-md-7">
        <div class="card p-3 h-100">
          <h6 class="mb-3">🛰 Operator Snapshot</h6>
          <div id="ops-snapshot" style="font-size:.84rem;color:#8b8fa8;line-height:1.7"></div>
        </div>
      </div>
    </div>

    <!-- Live bootstrap job progress -->
    <div class="row g-3 mb-4" id="job-progress-row" style="display:none!important">
      <div class="col-12">
        <div class="card p-3" style="border-color:#a78bfa40;background:#1a1a2e">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <h6 class="mb-0" style="color:#a78bfa">⚙️ Training Job In Progress</h6>
            <span id="job-status-badge" class="badge" style="background:#a78bfa22;color:#a78bfa">running</span>
          </div>
          <div id="job-progress-bar-wrap" class="mb-2">
            <div class="progress" style="height:10px;background:#2a2d3a">
              <div id="job-progress-bar" class="progress-bar" style="width:0%;background:#a78bfa;transition:width .4s"></div>
            </div>
          </div>
          <div class="d-flex justify-content-between" style="font-size:.8rem">
            <span id="job-progress-label" style="color:#a78bfa">Starting…</span>
            <span id="job-progress-pct" style="color:#8b8fa8"></span>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent training jobs -->
    <div class="row g-3 mb-4">
      <div class="col-12">
        <div class="card p-3">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h6 class="mb-0">📦 Bootstrap & Training Job History</h6>
            <button class="btn btn-sm" style="background:#1a2d3a;color:#38bdf8;border:1px solid #2a2d3a" onclick="refreshJobs()">↻ Refresh</button>
          </div>
          <div id="job-history"></div>
        </div>
      </div>
    </div>

    <!-- Model health cards per symbol -->
    <div class="row g-3 mb-4">
      <div class="col-12">
        <div class="card p-3">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h6 class="mb-0">🩺 Model Health per Symbol</h6>
            <div class="d-flex gap-2 align-items-center">
              <select id="activity-symbol" class="form-control" style="max-width:150px" onchange="refreshActivity()">
                <option value="">All symbols</option>
              </select>
              <select id="activity-days" class="form-control" style="max-width:150px" onchange="refreshActivity()">
                <option value="0">All time</option>
                <option value="7">Last 7 days</option>
                <option value="30">Last 30 days</option>
                <option value="90">Last 90 days</option>
              </select>
              <button class="btn btn-sm" style="background:#1a2d3a;color:#38bdf8;border:1px solid #2a2d3a"
                onclick="refreshActivity()">↻ Refresh</button>
            </div>
          </div>
          <div id="health-cards" class="row g-3"></div>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-3">🧭 Outcome Resolution Reasons</h6>
          <canvas id="chart-resolution-reasons" height="200"></canvas>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-3">⏱ Average Time to Outcome</h6>
          <canvas id="chart-resolution-time" height="200"></canvas>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-6">
        <div class="card p-3">
          <h6 class="mb-3">📏 MFE vs MAE by Symbol</h6>
          <canvas id="chart-mfe-mae" height="200"></canvas>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h6 class="mb-3">🧠 Adaptive TP/SL Tuning</h6>
          <div id="adaptive-summary" style="font-size:.84rem;color:#8b8fa8;line-height:1.7"></div>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-4">
      <!-- Training event log -->
      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h6 class="mb-3">📋 Learning Cycle Log</h6>
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
      <div class="col-12">
        <div class="card p-3" style="border-color:#a78bfa40">
          <h6 class="mb-3">🚀 ML Bootstrap — Pre-train from 2 Years of Historical Data</h6>
          <p class="small mb-3" style="color:#8b8fa8">
            Runs the signal engine on 2 years of hourly bars per symbol, labels each signal
            as correct/incorrect from TP/SL outcomes, and trains the ML models.
            Takes ~5–10 minutes. Progress is visible in the 🔄 Learning Activity tab.
          </p>
          <div class="d-flex gap-3 align-items-center flex-wrap">
            <button class="btn" style="background:#2a1a4a;color:#a78bfa;border:1px solid #a78bfa40;padding:8px 24px;font-size:.9rem"
              onclick="adminBootstrap()">🚀 Run Bootstrap (All Symbols)</button>
            <div id="bootstrap-msg" class="small" style="color:#8b8fa8"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- AI Sentiment Settings -->
    <div class="row g-3 mb-4">
      <div class="col-12">
        <div class="card p-3" style="border-color:#38bdf840">
          <h6 class="mb-3">🧠 AI Sentiment Settings</h6>
          <p class="small mb-3" style="color:#8b8fa8">
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
              <div class="small mt-1" style="color:#555">
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
              <div class="small mt-1" style="color:#555">
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
              <div class="small mt-1" style="color:#555">
                Change this when deploying to a different host
              </div>
            </div>

            <div class="col-md-4">
              <label class="small text-muted">News Lookback (hours)</label>
              <input type="number" id="set-news_lookback_hours" class="form-control mt-1"
                step="1" min="1" max="48" placeholder="6">
            </div>

          </div>
          <div class="d-flex gap-3 align-items-center mt-3">
            <button class="btn" style="background:#1a3a4a;color:#38bdf8;border:1px solid #38bdf840;padding:6px 20px"
              onclick="saveSettings()">💾 Save Settings</button>
            <div id="settings-msg" class="small" style="color:#8b8fa8"></div>
          </div>

          <!-- Per-symbol sentiment status -->
          <div class="mt-4">
            <h6 class="mb-2" style="font-size:.85rem;color:#8b8fa8">Current Sentiment Cache</h6>
            <div id="sentiment-status" style="font-size:.8rem"></div>
          </div>
        </div>
      </div>
    </div>

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

let dailyChart, symbolChart, bucketsChart, strengthChart, resolutionReasonsChart, resolutionTimeChart, mfeMaeChart, trainRunsChart;

function statusChip(ok, warnText, badText) {
  if (ok === 'ok') return `<span class="status-dot status-ok"></span>${warnText}`;
  if (ok === 'warn') return `<span class="status-dot status-warn"></span>${badText}`;
  return `<span class="status-dot status-bad"></span>${badText}`;
}

async function refreshOps() {
  const ops = await fetch('/api/ops').then(r=>r.json());
  const lastSignal = ops.last_signal;
  const lastCycle = ops.last_cycle;
  const signalHealthy = lastSignal ? ((Date.now() - new Date(lastSignal.sent_at+'Z')) / 3600000 < 24 ? 'ok' : 'warn') : 'bad';
  const learningHealthy = lastCycle ? ((Date.now() - new Date(lastCycle.checked_at+'Z')) / 3600000 < 1 ? 'ok' : 'warn') : 'bad';
  document.getElementById('ops-signal-status').innerHTML = statusChip(signalHealthy, 'Flowing', 'Idle');
  document.getElementById('ops-learning-status').innerHTML = statusChip(learningHealthy, 'Checking', 'Stale');
  document.getElementById('ops-pending-work').textContent = `${ops.pending_outcomes} / ${ops.pending_actions}`;
  document.getElementById('ops-coverage').textContent = `${ops.active_symbol_count}`;
  document.getElementById('ops-last-signal').textContent = lastSignal
    ? `${lastSignal.symbol} ${lastSignal.action} · ${new Date(lastSignal.sent_at+'Z').toLocaleString()}`
    : 'No signal activity yet';
  document.getElementById('ops-last-learning').textContent = lastCycle
    ? `${lastCycle.symbol} · ${lastCycle.retrained ? 'retrained' : 'checked'} · ${new Date(lastCycle.checked_at+'Z').toLocaleString()}`
    : 'No learning cycles yet';
  document.getElementById('ops-pending-meta').textContent = `${ops.pending_outcomes} pending outcomes · ${ops.pending_actions} queued actions`;
  document.getElementById('ops-coverage-meta').textContent = `${ops.active_symbols.join(', ') || 'No watched symbols'}`;

  const actionSelect = document.getElementById('action-symbol');
  if (actionSelect && actionSelect.options.length <= 1) {
    actionSelect.innerHTML = '<option value="">Pick symbol</option>' + ops.active_symbols.map(sym => `<option value="${sym}">${sym}</option>`).join('');
  }
  const snapshot = [];
  if (ops.last_action) snapshot.push(`Last action: ${ops.last_action.action}${ops.last_action.symbol ? ' ' + ops.last_action.symbol : ''} (${ops.last_action.status})`);
  if (ops.last_train) snapshot.push(`Last train: ${ops.last_train.symbol} at ${new Date(ops.last_train.trained_at+'Z').toLocaleString()}`);
  if (ops.last_outcome) snapshot.push(`Last outcome: ${ops.last_outcome.symbol} ${ops.last_outcome.outcome} via ${ops.last_outcome.resolution_reason || 'n/a'}`);
  snapshot.push(`Tracked symbols: ${ops.active_symbol_count}`);
  document.getElementById('ops-snapshot').innerHTML = snapshot.map(s => `<div class="mb-2">${s}</div>`).join('');
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
  document.getElementById('ml-confidence-note').textContent =
    (summary.ai_resolved_signals || 0) > 0
      ? `Current AI threshold is 65%. Low-looking confidence usually means limited resolved training data, not necessarily a broken model.`
      : 'No resolved AI-backed signals yet. Confidence quality improves after the bot accumulates validated outcomes.';
}

// ── Admin ─────────────────────────────────────────────────────────────────────
// ── Settings ──────────────────────────────────────────────────────────────────
async function loadSettings() {
  const s = await fetch('/api/settings').then(r => r.json());
  const keys = ['sentiment_provider','news_provider','sentiment_suppress_threshold',
                 'finnhub_api_key','gemini_api_key','claude_api_key','sentiment_local_url','news_lookback_hours'];
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
        return `<div style="background:#0f1117;border:1px solid #2a2d3a;border-radius:6px;padding:4px 10px">
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
                 'finnhub_api_key','gemini_api_key','claude_api_key','sentiment_local_url','news_lookback_hours'];
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
            <span>🕒 Trained ${modelAge(h.last_trained)} · Checked ${modelAge(h.last_checked)}</span><br>
            <span>✅ ${h.recent_successes} success · ❌ ${h.recent_failures} failure</span>
          </div>
        </div>
      </div>`;
    }).join('') || '<div class="col-12" style="color:#555">No training data yet — models train automatically once signals accumulate outcomes.</div>';

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
      options: {plugins:{legend:{labels:{color:'#8b8fa8'}}}}
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

// ── Init ──────────────────────────────────────────────────────────────────────
refreshOverview();
setInterval(refreshOverview, 30000);
setInterval(refreshOps, 30000);

document.querySelectorAll('[data-bs-target="#tab-ml"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => refreshML())
);
document.querySelectorAll('[data-bs-target="#tab-activity"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => { refreshActivity(); refreshJobs(); })
);
document.querySelectorAll('[data-bs-target="#tab-admin"]').forEach(el =>
  el.addEventListener('shown.bs.tab', () => { refreshAdmin(); loadSettings(); })
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
    allowed = {"scan_now", "check_outcomes_now", "retrain_all", "retrain_symbol", "run_bootstrap"}
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
        "news_lookback_hours",
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
    return jsonify({"ok": False, "error": "Symbol not found or already inactive"}), 404


if __name__ == "__main__":
    init_db()
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "5000"))
    app.run(host=host, port=port)
