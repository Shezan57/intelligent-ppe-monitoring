# Intelligent PPE Compliance Monitoring System

## What This Is
An Intelligent Personal Protective Equipment (PPE) Compliance Monitoring System. It's a real-time, highly accurate microservice-based AI pipeline designed for construction and industrial sites. It overcomes "Alert Fatigue" by combining fast bounding box detection with dense semantic verification.

## Core Value
Economically viable, real-time safety monitoring that eliminates alert fatigue and the absence detection paradox via deduplicated, accurate daily reporting.

## Requirements

### Validated

- ✓ [Dataset Merging] — 29,053-image unified dataset with a 5-class schema.
- ✓ [YOLO26m Baseline] — Reached 89.7% mAP@50 and 91.8% no-helmet detection AP.
- ✓ [Sentry-Judge Pipeline] — Asynchronous decoupling of YOLO (Sentry) and SAM 3 (Judge).
- ✓ [Sentry 5-Path Triage] — 30 FPS bounding box processing with a 5-minute cooldown per worker.
- ✓ [Agentic Reporter] — LLM-generated daily OSHA summary PDFs.

### Active

- [ ] [Cross-Camera Tracking] — Resolve worker tracking ID loss when moving between camera zones to prevent queue flooding.
- [ ] [Sentry-Judge Load Balancing] — Optimize architecture to perfectly balance Sentry's queue payload against Judge's processing limits.
- [ ] [Empirical Ablation Studies] — Benchmark YOLO11m vs YOLO26m, and test custom IoU tracking vs ByteTrack/BoT-SORT.

### Out of Scope

- [Real-time manual SMS Spams] — Supervisors should not be spammed with false-alarms. Only end-of-day PDF summaries are triggered unless critical.

## Context
- **Tech Ecosystem:** Python backend, FastAPI, YOLOv11, SAM 3, React Frontend.
- **Constraints / Motivation:** Object detectors struggle with *absence detection* (e.g., confirming a helmet is missing is very hard). High false alarms cause site managers to mute alerts. SAM 3 runs at < 1 FPS which normally blocks live video tracking if run synchronously.

## Constraints
- **Performance:** Sentry MUST sustain 30 FPS tracking on the video feed.
- **Budget / Hardware:** The solution must be economically viable to deploy on a single edge device (e.g., T4 GPU) rather than requiring enterprise clusters.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Decouple detection and verification | Solves the latency of SAM (Judge) slowing down the live stream (Sentry). | ✓ Good |
| 5-minute tracking cooldown | Prevents flooding the Judge queue for the same worker. | ✓ Good |
| Single Daily OSHA Report | Eliminates 'alert fatigue' from management via LLM summarization. | ✓ Good |

## Evolution
This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-10 after initialization*
