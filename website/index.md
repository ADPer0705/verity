---
layout: home
title: Verity — Forensic Deepfake Examination Workbench
description: Multi-detector deepfake examination for single-investigator workstations and air-gapped forensic labs.

hero:
  name: Verity
  text: Forensic Deepfake Workbench
  tagline: Inspect images, videos, and audio across 58 isolated detector architectures, 5 deterministic signal analyzers, and SHA-256 chain-of-custody evidence bundles. Built for single-investigator, air-gapped labs.
  image:
    src: /logo.svg
    alt: Verity forensic mark
  actions:
    - theme: brand
      text: Quickstart & Setup
      link: /quickstart
    - theme: alt
      text: Architecture Overview
      link: /architecture/overview
    - theme: alt
      text: All 58 Detectors
      link: /reference/detector-catalog

features:
  - icon: 📦
    title: Standalone Single Binary
    details: One checksummed Linux executable bundles the CLI, FastAPI server, React Web Console, and detector specs — storing runtime data cleanly via platformdirs.
  - icon: ⚖️
    title: No Collapsed Verdict
    details: Every detector's verdict, confidence, reasoning, and visual exhibits remain separate. Panel disagreement is treated as a first-class investigative finding.
  - icon: 🧪
    title: 58 Isolated uv Runtimes
    details: Conflicting PyTorch, CUDA, timm, and transformers graphs run out-of-process in per-detector uv virtual environments so one model never breaks another.
  - icon: 🔒
    title: 100% Air-Gap & USB Transfer
    details: Provision once online or stage a verified USB transfer kit (`verity airgap export/import`) and seal `offline=true` for zero-network examinations.
  - icon: 🔍
    title: Deep Explainability + Forensics
    details: Pairs Grad-CAM, native attention maps, face crops, and frame timelines with five model-free metadata, compression, noise, temporal, and audio analyzers.
  - icon: 🛡️
    title: Chain-of-Custody Bundles
    details: Content-addressed SHA-256 ingestion (`0o444` working copies) plus deterministic `.zip` dossiers carrying `MANIFEST.json` and `SHA256SUMS.txt`.
---

<div class="vy-home-wrap">

<div class="vy-metrics-grid">
  <div class="vy-metric-card">
    <div class="vy-metric-val">58</div>
    <div class="vy-metric-label">Detector Architectures</div>
    <div class="vy-metric-sub">Images, face swaps, reenactment, lip-sync &amp; cloned voice</div>
  </div>
  <div class="vy-metric-card">
    <div class="vy-metric-val">39</div>
    <div class="vy-metric-label">Evidence-Grade Models</div>
    <div class="vy-metric-sub">Auto-selected only with verified trained heads</div>
  </div>
  <div class="vy-metric-card">
    <div class="vy-metric-val">5</div>
    <div class="vy-metric-label">Model-Free Analyzers</div>
    <div class="vy-metric-sub">Metadata, ELA/DCT, noise/FFT, temporal &amp; audio</div>
  </div>
  <div class="vy-metric-card">
    <div class="vy-metric-val">28 / 6.3 GB</div>
    <div class="vy-metric-label">HF Weight Mirror</div>
    <div class="vy-metric-sub">Attributed on ADPer/verity-weights + origin URLs</div>
  </div>
</div>

<div class="vy-install-card">
  <div class="vy-install-header">
    <div class="vy-install-dots">
      <span></span><span></span><span></span>
      <strong style="margin-left: 0.45rem; font-size: 0.875rem;">One-Command Air-Gap Lab Setup (Linux x86_64)</strong>
    </div>
    <span class="vy-install-badge">v0.6.0 · glibc 2.35+ · platformdirs</span>
  </div>
  <div class="vy-install-body">

```bash
# 1. Install the SHA-256 verified standalone binary into ~/.local/bin/verity
curl -fsSL https://raw.githubusercontent.com/ADPer0705/verity/main/install.sh | bash

# 2. Bootstrap uv + Pythons, build 39 detector envs, pull & verify weights, seal offline
verity setup --trained --airgap-ready

# 3. Disconnect network permanently and launch the Investigator Web Console
verity doctor --verify && verity serve
```

  </div>
</div>

<div class="vy-pathways">
  <div class="vy-pathway-card">
    <span class="vy-pathway-tag">Architecture</span>
    <h3>Pipeline &amp; Engineering Invariants</h3>
    <p>How deterministic ingestion, model-free analyzers, trust-tier selection, isolated execution, and chain-of-custody bundles fit together.</p>
    <div class="vy-pathway-links">
      <a href="/architecture/overview">Architecture Overview →</a>
      <a href="/architecture/pipeline">Examination Pipeline →</a>
      <a href="/architecture/forensics">Model-Free Forensics →</a>
      <a href="/architecture/security-and-integrity">Security &amp; Chain of Custody →</a>
    </div>
  </div>

  <div class="vy-pathway-card">
    <span class="vy-pathway-tag">Reference</span>
    <h3>58 Detectors &amp; Visual Exhibits</h3>
    <p>Field guide to every ported architecture, trust tier, caveat, Grad-CAM / attention map contract, and archived evidence format.</p>
    <div class="vy-pathway-links">
      <a href="/reference/detector-catalog">Detector Catalog (All 58 Ports) →</a>
      <a href="/reference/explainability-artifacts">Explainability &amp; Formats →</a>
      <a href="/architecture/detector-engine">Shared Detector Engine →</a>
      <a href="/architecture/weights-and-checkpoints">Weights &amp; Attribution →</a>
    </div>
  </div>

  <div class="vy-pathway-card">
    <span class="vy-pathway-tag">Operations</span>
    <h3>Air-Gap Deployment &amp; Releases</h3>
    <p>Workstation installation, offline USB air-gap kit export/import, Docker Compose fallback, and release verification runbooks.</p>
    <div class="vy-pathway-links">
      <a href="/operations/deployment-and-airgap">Deployment &amp; Air-Gap Operations →</a>
      <a href="/operations/releasing-and-publishing">Releasing &amp; Extending Verity →</a>
      <a href="/architecture/web-api-and-console">Web API &amp; Investigator Console →</a>
      <a href="https://github.com/ADPer0705/verity/releases/tag/v0.6.0">Download v0.6.0 Release →</a>
    </div>
  </div>
</div>

<div class="vy-principle-banner">
  <strong>Investigative Principle:</strong> Detector predictions are statistical indicators — never standalone proof. Verity preserves every detector's verdict, auxiliary cues, Grad-CAM/attention visualizations, and model-free signal measurements side by side so a qualified examiner evaluates the complete record without hiding conflict behind an opaque score.
</div>

</div>
