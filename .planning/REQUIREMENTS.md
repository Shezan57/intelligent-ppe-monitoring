# Requirements: Intelligent PPE Compliance Monitoring System

**Defined:** 2026-04-10
**Core Value:** Economically viable, real-time safety monitoring that eliminates alert fatigue and the absence detection paradox via deduplicated, accurate daily reporting.

## v1 Requirements

### Tracking & Deduplication (Phase 1)
- [ ] **TRK-01**: Sentry must preserve unique worker tracking IDs across different camera angles/zones.
- [ ] **TRK-02**: Tracking must block duplicate SAM validation submissions within the 5-minute cooldown regardless of camera zone movement.

### System Optimization (Phase 2)
- [ ] **OPT-01**: The Judge queue must not exceed memory bounds when saturated with frames.
- [ ] **OPT-02**: Measure and document optimal thread/worker distribution between Sentry output and Judge ingestion to hit 30FPS stream capacity.

### Benchmarks (Phase 3)
- [ ] **BEN-01**: Execute ablation tests validating metrics between YOLO11m and YOLO26m processing speeds.
- [ ] **BEN-02**: Document tracking fidelity comparing custom IoU algorithms versus mainstream trackers like ByteTrack.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Immediate SMS Spam | Contradicts primary goal of preventing alert fatigue. End-of-day reports exist to replace this. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRK-01 | Phase 1 | Pending |
| TRK-02 | Phase 1 | Pending |
| OPT-01 | Phase 2 | Pending |
| OPT-02 | Phase 2 | Pending |
| BEN-01 | Phase 3 | Pending |
| BEN-02 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 6 total
- Mapped to phases: 6
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-10*
*Last updated: 2026-04-10 after initial definition*
