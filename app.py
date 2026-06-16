"""
ATS Resume Scorer — FastAPI Application
========================================
POST /score       — JSON body {jd_text, resume_text}
POST /score-files — multipart form with optional .docx uploads
GET  /            — HTML frontend
"""

import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nlp_engine import score_resume

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ATS Resume Scorer",
    description="Keyword extraction, TF-IDF cosine similarity, and ATS gap analysis.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    jd_text:     str
    resume_text: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/score")
async def score_json(req: ScoreRequest):
    try:
        result = score_resume(req.jd_text, req.resume_text)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


@app.post("/score-files")
async def score_files(
    jd_text:     str         = Form(default=""),
    resume_text: str         = Form(default=""),
    jd_file:     Optional[UploadFile] = File(default=None),
    resume_file: Optional[UploadFile] = File(default=None),
):
    jd_bytes  = await jd_file.read()     if jd_file     else None
    res_bytes = await resume_file.read() if resume_file  else None

    try:
        result = score_resume(
            jd_text=jd_text, resume_text=resume_text,
            jd_file=jd_bytes,     resume_file=res_bytes,
            jd_filename=jd_file.filename     if jd_file     else "",
            resume_filename=resume_file.filename if resume_file else "",
        )
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── HTML Frontend ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ATS Resume Scorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --ink:      #1a1a1a;
  --paper:    #f5f0e8;
  --cream:    #ede8dc;
  --rule:     #d4c9b0;
  --accent:   #c8410a;
  --green:    #1a6b3a;
  --amber:    #b86c00;
  --blue:     #1a4a8a;
  --muted:    #6b6358;
  --serif:    'Instrument Serif', Georgia, serif;
  --sans:     'Instrument Sans', sans-serif;
  --mono:     'JetBrains Mono', monospace;
}

html, body {
  background: var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  min-height: 100vh;
}

/* Ruled paper texture */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: repeating-linear-gradient(
    transparent, transparent 27px,
    rgba(180,160,120,0.12) 27px, rgba(180,160,120,0.12) 28px
  );
  pointer-events: none;
  z-index: 0;
}

/* Red margin line */
body::after {
  content: '';
  position: fixed;
  left: 72px;
  top: 0; bottom: 0;
  width: 1px;
  background: rgba(200, 65, 10, 0.15);
  pointer-events: none;
  z-index: 0;
}

.wrapper {
  position: relative;
  z-index: 1;
  max-width: 1160px;
  margin: 0 auto;
  padding: 48px 32px 80px 96px;
}

/* ── Header ── */
header {
  margin-bottom: 48px;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 20px;
}

.eyebrow {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 3px;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 10px;
}

h1 {
  font-family: var(--serif);
  font-size: clamp(2.2rem, 5vw, 3.2rem);
  font-weight: 400;
  letter-spacing: -0.5px;
  line-height: 1.1;
  margin-bottom: 10px;
}

h1 em {
  font-style: italic;
  color: var(--accent);
}

.tagline {
  font-size: 13px;
  color: var(--muted);
  font-family: var(--mono);
}

/* ── Two-column input grid ── */
.input-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

@media (max-width: 768px) { .input-grid { grid-template-columns: 1fr; } }

.field-block { display: flex; flex-direction: column; }

.field-label {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.field-label .dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
  display: inline-block;
}

textarea {
  flex: 1;
  background: rgba(255,255,255,0.6);
  border: 1px solid var(--rule);
  border-bottom: 2px solid var(--ink);
  padding: 16px;
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.7;
  color: var(--ink);
  resize: vertical;
  min-height: 280px;
  outline: none;
  transition: border-color 0.2s, background 0.2s;
}

textarea:focus {
  border-color: var(--accent);
  background: rgba(255,255,255,0.85);
}

textarea::placeholder { color: var(--muted); opacity: 0.6; }

/* Upload row */
.upload-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 28px;
}

.upload-box {
  border: 1px dashed var(--rule);
  padding: 12px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  transition: border-color 0.2s;
  background: rgba(255,255,255,0.4);
}

.upload-box:hover { border-color: var(--accent); }

.upload-box input[type="file"] { display: none; }

.upload-icon { font-size: 1.1rem; }

.upload-text {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
}

.upload-text strong { color: var(--ink); font-weight: 500; }

/* Submit button */
.submit-wrap { display: flex; gap: 12px; align-items: center; margin-bottom: 40px; }

#submit-btn {
  background: var(--ink);
  color: var(--paper);
  border: none;
  padding: 14px 36px;
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 1px;
  text-transform: uppercase;
  cursor: pointer;
  transition: background 0.2s, transform 0.1s;
}

#submit-btn:hover  { background: var(--accent); }
#submit-btn:active { transform: scale(0.97); }
#submit-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.clear-btn {
  background: none;
  border: 1px solid var(--rule);
  padding: 14px 20px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  cursor: pointer;
  transition: border-color 0.2s, color 0.2s;
}
.clear-btn:hover { border-color: var(--ink); color: var(--ink); }

/* Spinner */
.spinner {
  display: inline-block;
  width: 12px; height: 12px;
  border: 2px solid rgba(245,240,232,0.4);
  border-top-color: var(--paper);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  vertical-align: middle;
  margin-right: 6px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Results section ── */
#results { display: none; animation: fadeUp 0.4s ease; }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

.results-header {
  border-top: 2px solid var(--ink);
  border-bottom: 1px solid var(--rule);
  padding: 16px 0;
  margin-bottom: 28px;
  display: flex;
  align-items: baseline;
  gap: 16px;
}

.results-title {
  font-family: var(--serif);
  font-size: 1.6rem;
  font-weight: 400;
}

/* Score hero */
.score-hero {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 32px;
  align-items: center;
  background: var(--ink);
  color: var(--paper);
  padding: 32px;
  margin-bottom: 28px;
}

.score-circle {
  width: 120px; height: 120px;
  border: 3px solid rgba(245,240,232,0.3);
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  position: relative;
}

.score-number {
  font-family: var(--serif);
  font-size: 2.4rem;
  line-height: 1;
  font-weight: 400;
}

.score-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  color: rgba(245,240,232,0.5);
  text-transform: uppercase;
  margin-top: 3px;
}

.score-grade {
  position: absolute;
  top: -8px; right: -8px;
  width: 28px; height: 28px;
  background: var(--accent);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  color: white;
}

.score-meta { flex: 1; }

.verdict-text {
  font-family: var(--serif);
  font-size: 1.3rem;
  font-style: italic;
  margin-bottom: 20px;
  color: rgba(245,240,232,0.9);
}

.score-bars { display: flex; flex-direction: column; gap: 10px; }

.bar-row {
  display: grid;
  grid-template-columns: 140px 1fr 48px;
  align-items: center;
  gap: 12px;
}

.bar-name {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 1px;
  color: rgba(245,240,232,0.6);
}

.bar-track {
  height: 3px;
  background: rgba(245,240,232,0.15);
}

.bar-fill {
  height: 100%;
  background: var(--paper);
  transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

.bar-pct {
  font-family: var(--mono);
  font-size: 10px;
  color: rgba(245,240,232,0.7);
  text-align: right;
}

/* Sub-metric cards */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--rule);
  border: 1px solid var(--rule);
  margin-bottom: 28px;
}

@media (max-width: 640px) { .metrics-row { grid-template-columns: 1fr 1fr; } }

.metric-card {
  background: var(--paper);
  padding: 16px;
  text-align: center;
}

.metric-value {
  font-family: var(--serif);
  font-size: 1.6rem;
  font-weight: 400;
  margin-bottom: 4px;
}

.metric-value.good   { color: var(--green); }
.metric-value.mid    { color: var(--amber); }
.metric-value.bad    { color: var(--accent); }

.metric-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
}

/* Keywords panels */
.kw-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 28px;
}

@media (max-width: 640px) { .kw-grid { grid-template-columns: 1fr; } }

.kw-panel {
  border: 1px solid var(--rule);
  background: rgba(255,255,255,0.4);
}

.kw-panel-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--rule);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.kw-panel-title {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 2px;
  text-transform: uppercase;
}

.kw-panel-title.matched { color: var(--green); }
.kw-panel-title.missing { color: var(--accent); }

.kw-count {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 2px;
}
.kw-count.matched { background: rgba(26,107,58,0.1); color: var(--green); }
.kw-count.missing { background: rgba(200,65,10,0.1);  color: var(--accent); }

.kw-list {
  padding: 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-height: 220px;
  overflow-y: auto;
}

.kw-tag {
  font-family: var(--mono);
  font-size: 11px;
  padding: 4px 10px;
  border: 1px solid;
}

.kw-tag.matched {
  border-color: rgba(26,107,58,0.3);
  color: var(--green);
  background: rgba(26,107,58,0.05);
}
.kw-tag.missing {
  border-color: rgba(200,65,10,0.3);
  color: var(--accent);
  background: rgba(200,65,10,0.05);
}

/* Category breakdown */
.categories-section { margin-bottom: 28px; }

.section-label {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--rule);
  padding-bottom: 8px;
  margin-bottom: 14px;
}

.cat-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.cat-row {
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 16px;
  align-items: center;
}

.cat-name {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
}

.cat-tags { display: flex; flex-wrap: wrap; gap: 4px; }

.cat-tag {
  font-family: var(--mono);
  font-size: 10px;
  padding: 2px 8px;
  background: rgba(200,65,10,0.08);
  border: 1px solid rgba(200,65,10,0.2);
  color: var(--accent);
}

/* Tips */
.tips-section {
  background: rgba(26,74,138,0.05);
  border: 1px solid rgba(26,74,138,0.2);
  border-left: 3px solid var(--blue);
  padding: 20px;
  margin-bottom: 28px;
}

.tips-title {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--blue);
  margin-bottom: 12px;
}

.tip-item {
  font-family: var(--sans);
  font-size: 13px;
  color: var(--ink);
  line-height: 1.6;
  padding: 6px 0;
  border-bottom: 1px solid rgba(26,74,138,0.1);
  display: flex;
  gap: 10px;
}

.tip-item:last-child { border-bottom: none; }

.tip-num {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--blue);
  flex-shrink: 0;
  margin-top: 2px;
}

/* Struct section */
.struct-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--rule);
  border: 1px solid var(--rule);
  margin-bottom: 28px;
}

@media (max-width: 640px) { .struct-grid { grid-template-columns: 1fr 1fr; } }

.struct-item {
  background: var(--paper);
  padding: 12px 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.struct-icon { font-size: 14px; }

.struct-name {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
}

.struct-name.ok  { color: var(--green); }
.struct-name.nok { color: var(--accent); }

/* Error */
.error-banner {
  background: rgba(200,65,10,0.08);
  border: 1px solid rgba(200,65,10,0.3);
  border-left: 3px solid var(--accent);
  padding: 16px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--accent);
  margin-bottom: 24px;
}

/* Example filler */
.example-btn {
  font-family: var(--mono);
  font-size: 11px;
  background: none;
  border: 1px solid var(--rule);
  color: var(--muted);
  padding: 6px 12px;
  cursor: pointer;
  transition: all 0.15s;
}
.example-btn:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<div class="wrapper">

  <header>
    <div class="eyebrow">NLP · TF-IDF · Cosine Similarity · spaCy</div>
    <h1>Resume <em>ATS</em> Scorer</h1>
    <p class="tagline">Keyword extraction + semantic similarity + structural gap analysis</p>
  </header>

  <!-- Upload row -->
  <div class="upload-row">
    <label class="upload-box" for="jd-file">
      <input type="file" id="jd-file" accept=".docx,.txt">
      <span class="upload-icon">📄</span>
      <div class="upload-text">
        <strong id="jd-file-label">Upload Job Description</strong><br>
        .docx or .txt — optional
      </div>
    </label>
    <label class="upload-box" for="resume-file">
      <input type="file" id="resume-file" accept=".docx,.txt">
      <span class="upload-icon">📋</span>
      <div class="upload-text">
        <strong id="resume-file-label">Upload Resume</strong><br>
        .docx or .txt — optional
      </div>
    </label>
  </div>

  <!-- Text areas -->
  <div style="margin-bottom:8px;display:flex;justify-content:flex-end;gap:8px">
    <button class="example-btn" onclick="fillExample()">Load example ↓</button>
  </div>
  <div class="input-grid">
    <div class="field-block">
      <div class="field-label"><span class="dot"></span> Job Description</div>
      <textarea id="jd-input" placeholder="Paste the full job description here…

Tip: include the requirements, qualifications, and responsibilities sections for best results."></textarea>
    </div>
    <div class="field-block">
      <div class="field-label"><span class="dot"></span> Your Resume</div>
      <textarea id="resume-input" placeholder="Paste your resume text here…

Tip: include your skills section, work experience, and education for accurate scoring."></textarea>
    </div>
  </div>

  <div class="submit-wrap">
    <button id="submit-btn" onclick="submitScore()">Analyze Resume</button>
    <button class="clear-btn" onclick="clearAll()">Clear</button>
    <span id="word-counts" style="font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:8px"></span>
  </div>

  <!-- Error -->
  <div id="error-box" class="error-banner" style="display:none"></div>

  <!-- Results -->
  <div id="results">

    <div class="results-header">
      <div class="results-title">Analysis Results</div>
      <span id="result-meta" style="font-family:var(--mono);font-size:11px;color:var(--muted)"></span>
    </div>

    <!-- Score hero -->
    <div class="score-hero">
      <div class="score-circle">
        <div class="score-number" id="r-score">—</div>
        <div class="score-label">ATS Score</div>
        <div class="score-grade" id="r-grade">—</div>
      </div>
      <div class="score-meta">
        <div class="verdict-text" id="r-verdict">—</div>
        <div class="score-bars">
          <div class="bar-row">
            <div class="bar-name">Keyword Match</div>
            <div class="bar-track"><div class="bar-fill" id="bar-kw" style="width:0%"></div></div>
            <div class="bar-pct" id="pct-kw">0%</div>
          </div>
          <div class="bar-row">
            <div class="bar-name">Semantic Similarity</div>
            <div class="bar-track"><div class="bar-fill" id="bar-cos" style="width:0%"></div></div>
            <div class="bar-pct" id="pct-cos">0%</div>
          </div>
          <div class="bar-row">
            <div class="bar-name">Resume Structure</div>
            <div class="bar-track"><div class="bar-fill" id="bar-str" style="width:0%"></div></div>
            <div class="bar-pct" id="pct-str">0%</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Sub metrics -->
    <div class="metrics-row">
      <div class="metric-card">
        <div class="metric-value" id="m-kw">—</div>
        <div class="metric-label">Keywords Matched</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" id="m-miss">—</div>
        <div class="metric-label">Keywords Missing</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" id="m-jd-words">—</div>
        <div class="metric-label">JD Word Count</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" id="m-res-words">—</div>
        <div class="metric-label">Resume Word Count</div>
      </div>
    </div>

    <!-- Keyword panels -->
    <div class="kw-grid">
      <div class="kw-panel">
        <div class="kw-panel-header">
          <div class="kw-panel-title matched">✓ Matched Keywords</div>
          <div class="kw-count matched" id="count-matched">0</div>
        </div>
        <div class="kw-list" id="matched-list"></div>
      </div>
      <div class="kw-panel">
        <div class="kw-panel-header">
          <div class="kw-panel-title missing">✗ Missing Keywords</div>
          <div class="kw-count missing" id="count-missing">0</div>
        </div>
        <div class="kw-list" id="missing-list"></div>
      </div>
    </div>

    <!-- Missing by category -->
    <div class="categories-section" id="cat-section" style="display:none">
      <div class="section-label">Missing Keywords by Category</div>
      <div class="cat-grid" id="cat-grid"></div>
    </div>

    <!-- Resume structure check -->
    <div style="margin-bottom:28px">
      <div class="section-label">Resume Structure Checklist</div>
      <div class="struct-grid" id="struct-grid"></div>
    </div>

    <!-- Tips -->
    <div class="tips-section" id="tips-section" style="display:none">
      <div class="tips-title">⚡ Actionable Improvements</div>
      <div id="tips-list"></div>
    </div>

  </div><!-- /results -->
</div><!-- /wrapper -->

<script>
const $ = id => document.getElementById(id);

// File upload label updates
['jd-file','resume-file'].forEach(id => {
  $(id).addEventListener('change', function() {
    const labelId = id === 'jd-file' ? 'jd-file-label' : 'resume-file-label';
    $(labelId).textContent = this.files[0] ? this.files[0].name : (id === 'jd-file' ? 'Upload Job Description' : 'Upload Resume');
  });
});

// Word count live update
function updateWordCounts() {
  const jdW  = ($('jd-input').value.trim().match(/\S+/g) || []).length;
  const resW = ($('resume-input').value.trim().match(/\S+/g) || []).length;
  if (jdW || resW) {
    $('word-counts').textContent = `JD: ${jdW} words  ·  Resume: ${resW} words`;
  }
}
$('jd-input').addEventListener('input', updateWordCounts);
$('resume-input').addEventListener('input', updateWordCounts);

async function submitScore() {
  const jdText  = $('jd-input').value.trim();
  const resText = $('resume-input').value.trim();
  const jdFile  = $('jd-file').files[0];
  const resFile = $('resume-file').files[0];

  if (!jdText && !jdFile) {
    showError('Please provide a job description (paste text or upload a .docx file).');
    return;
  }
  if (!resText && !resFile) {
    showError('Please provide your resume (paste text or upload a .docx file).');
    return;
  }

  $('submit-btn').disabled = true;
  $('submit-btn').innerHTML = '<span class="spinner"></span> Analyzing…';
  $('error-box').style.display = 'none';
  $('results').style.display = 'none';

  try {
    let data;
    if (jdFile || resFile) {
      const form = new FormData();
      form.append('jd_text', jdText);
      form.append('resume_text', resText);
      if (jdFile)  form.append('jd_file', jdFile);
      if (resFile) form.append('resume_file', resFile);
      const r = await fetch('/score-files', { method: 'POST', body: form });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
      data = await r.json();
    } else {
      const r = await fetch('/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jd_text: jdText, resume_text: resText }),
      });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
      data = await r.json();
    }
    renderResults(data);
  } catch (err) {
    showError(err.message || 'Unexpected error. Is the server running?');
  } finally {
    $('submit-btn').disabled = false;
    $('submit-btn').textContent = 'Analyze Resume';
  }
}

function scoreColor(v) {
  if (v >= 70) return 'good';
  if (v >= 45) return 'mid';
  return 'bad';
}

function renderResults(d) {
  // Hero score
  $('r-score').textContent = d.ats_score;
  $('r-grade').textContent = d.grade;
  $('r-verdict').textContent = d.verdict;

  // Bars (animate after small delay so CSS transition fires)
  setTimeout(() => {
    setBar('bar-kw',  'pct-kw',  d.keyword_match);
    setBar('bar-cos', 'pct-cos', d.cosine_sim);
    setBar('bar-str', 'pct-str', d.struct_score);
  }, 80);

  // Metrics
  setMetric('m-kw',       d.matched_keywords.length, d.matched_keywords.length >= d.jd_keywords.length * 0.7 ? 'good' : d.matched_keywords.length >= d.jd_keywords.length * 0.4 ? 'mid' : 'bad');
  setMetric('m-miss',     d.missing_keywords.length, d.missing_keywords.length === 0 ? 'good' : d.missing_keywords.length < 5 ? 'mid' : 'bad');
  setMetric('m-jd-words', d.jd_word_count, '');
  setMetric('m-res-words',d.resume_word_count, d.resume_word_count >= 300 ? 'good' : d.resume_word_count >= 150 ? 'mid' : 'bad');

  // Meta line
  $('result-meta').textContent = `${d.jd_keywords.length} keywords extracted from JD`;

  // Keyword tags
  renderTags('matched-list', d.matched_keywords, 'matched');
  renderTags('missing-list', d.missing_keywords, 'missing');
  $('count-matched').textContent = d.matched_keywords.length;
  $('count-missing').textContent = d.missing_keywords.length;

  // Categories
  const catGrid = $('cat-grid');
  catGrid.innerHTML = '';
  const cats = d.categories || {};
  const catEntries = Object.entries(cats).filter(([,v]) => v.length > 0);
  if (catEntries.length > 0) {
    $('cat-section').style.display = 'block';
    catEntries.forEach(([cat, kws]) => {
      const row = document.createElement('div');
      row.className = 'cat-row';
      row.innerHTML = `
        <div class="cat-name">${cat}</div>
        <div class="cat-tags">${kws.map(k => `<span class="cat-tag">${k}</span>`).join('')}</div>`;
      catGrid.appendChild(row);
    });
  } else {
    $('cat-section').style.display = 'none';
  }

  // Struct checklist
  const sg = $('struct-grid');
  sg.innerHTML = '';
  const icons = {
    email: '✉', phone: '📞', linkedin: '🔗', education: '🎓',
    experience: '💼', projects: '🔧', skills_section: '⚡', achievements: '📈'
  };
  const labels = {
    email: 'Email', phone: 'Phone', linkedin: 'LinkedIn', education: 'Education',
    experience: 'Experience', projects: 'Projects', skills_section: 'Skills Section', achievements: 'Metrics/Results'
  };
  Object.entries(d.struct_indicators || {}).forEach(([key, ok]) => {
    const div = document.createElement('div');
    div.className = 'struct-item';
    div.innerHTML = `
      <span class="struct-icon">${ok ? icons[key] || '✓' : '○'}</span>
      <span class="struct-name ${ok ? 'ok' : 'nok'}">${labels[key] || key}${ok ? '' : ' ✗'}</span>`;
    sg.appendChild(div);
  });

  // Tips
  if (d.tips && d.tips.length > 0) {
    $('tips-section').style.display = 'block';
    $('tips-list').innerHTML = d.tips.map((tip, i) =>
      `<div class="tip-item"><span class="tip-num">${String(i+1).padStart(2,'0')}</span><span>${tip}</span></div>`
    ).join('');
  } else {
    $('tips-section').style.display = 'none';
  }

  $('results').style.display = 'block';
  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setBar(barId, pctId, value) {
  $(barId).style.width = Math.min(value, 100) + '%';
  $(pctId).textContent = value.toFixed(1) + '%';
}

function setMetric(id, value, cls) {
  const el = $(id);
  el.textContent = value;
  el.className = 'metric-value' + (cls ? ' ' + cls : '');
}

function renderTags(containerId, keywords, cls) {
  $(containerId).innerHTML = keywords.length
    ? keywords.map(k => `<span class="kw-tag ${cls}">${k}</span>`).join('')
    : `<span style="font-family:var(--mono);font-size:11px;color:var(--muted);padding:4px">None</span>`;
}

function showError(msg) {
  $('error-box').textContent = '⚠ ' + msg;
  $('error-box').style.display = 'block';
}

function clearAll() {
  $('jd-input').value = '';
  $('resume-input').value = '';
  $('jd-file').value = '';
  $('resume-file').value = '';
  $('jd-file-label').textContent = 'Upload Job Description';
  $('resume-file-label').textContent = 'Upload Resume';
  $('results').style.display = 'none';
  $('error-box').style.display = 'none';
  $('word-counts').textContent = '';
}

function fillExample() {
  $('jd-input').value = `Senior Python Engineer — ML Platform

We are seeking a Senior Python Engineer to join our Machine Learning Platform team.

Requirements:
• 5+ years of Python development experience
• Strong expertise in machine learning frameworks: TensorFlow or PyTorch
• Experience building and deploying REST APIs with FastAPI or Django
• Proficiency in SQL (PostgreSQL) and NoSQL (MongoDB, Redis)
• Hands-on experience with Docker, Kubernetes, and AWS or GCP
• Knowledge of scikit-learn, pandas, numpy for data engineering
• Experience with NLP and natural language processing pipelines
• CI/CD experience with GitHub Actions or Jenkins
• Familiarity with Apache Kafka or Apache Airflow for data pipelines
• Strong communication and leadership skills
• Agile/Scrum methodology experience preferred

Nice to have: Spark, Elasticsearch, MLflow, Hugging Face`;

  $('resume-input').value = `Alex Chen  |  alex@email.com  |  +1 (555) 987-6543  |  linkedin.com/in/alexchen

SUMMARY
Senior Software Engineer with 6 years building scalable Python systems and ML pipelines.

EXPERIENCE
ML Platform Engineer — DataCo Inc. (2021–Present)
• Built end-to-end machine learning pipelines using TensorFlow and scikit-learn, improving model accuracy by 18%
• Developed high-throughput REST APIs using FastAPI handling 500K+ daily requests
• Reduced PostgreSQL query latency by 45% through indexing and query optimization
• Containerized services using Docker and orchestrated with Kubernetes on AWS ECS
• Implemented CI/CD pipelines with GitHub Actions, cutting deployment time by 60%
• Processed real-time events using Apache Kafka and scheduled ETL jobs in Apache Airflow

Software Engineer — StartupXYZ (2018–2021)
• Developed Django REST APIs integrated with MongoDB and Redis caching layer
• Built NLP text classification pipelines using spaCy and Hugging Face transformers
• Collaborated in Agile/Scrum teams with 2-week sprint cycles

EDUCATION
M.S. Computer Science — Stanford University, 2018

SKILLS
Python, FastAPI, Django, TensorFlow, PyTorch, scikit-learn, pandas, numpy, SQL, PostgreSQL,
MongoDB, Redis, Docker, Kubernetes, AWS, GCP, Kafka, Airflow, NLP, GitHub Actions, Jenkins, Agile`;

  updateWordCounts();
}

// Enter to submit (Ctrl+Enter in textarea)
document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') submitScore();
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def frontend():
    return HTML
