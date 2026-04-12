# ROADMAP

## Milestones

- 🚧 **v1.0 MVP** - Phases 1-3 (in progress)

## Phases

### 🚧 v1.0 MVP (In Progress)

**Milestone Goal:** Real-time safety monitoring without alert fatigue.

#### Phase 1: Worker Tracking Optimization
**Goal**: Resolve cross-camera tracking resets. Decouple worker ID from a specific camera zone so if they migrate between cameras, they are still under the 5-minute cooldown.
**Depends on**: None
**Requirements**: [TRK-01, TRK-02]
**Success Criteria**:
  1. A worker tracked in Camera A retains their ID in Camera B.
  2. Cooldown timer persists across cameras.
**Plans**: 1 plan

Plans:
- [ ] 01-01: Implement spatial ID persistence layer mapping.

#### Phase 2: Queue Balancing Architecture
**Goal**: Profile and tune standard multi-threading logic balancing the Sentry and Judge queues.
**Depends on**: Phase 1
**Requirements**: [OPT-01, OPT-02]
**Success Criteria**:
  1. Queue size avoids OOM errors under heavy load.
  2. Sustained 30FPS output from Sentry.
**Plans**: 1 plan

Plans:
- [ ] 02-01: Tune consumer thread limits and frame dropping logic for Judge async queue.

#### Phase 3: Empirical Testing & Profiling
**Goal**: Benchmark metrics (YOLO11m vs YOLO26m, IoU vs ByteTrack).
**Depends on**: Phase 2
**Requirements**: [BEN-01, BEN-02]
**Success Criteria**:
  1. Detailed documentation of mAP vs real-world reliability.
  2. Ablation tables proving architectural merits.
**Plans**: 1 plan

Plans:
- [ ] 03-01: Build and execute automated colab notebook tracking test.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Worker Tracking Optimization | v1.0 | 0/1 | Not started | - |
| 2. Queue Balancing Architecture | v1.0 | 0/1 | Not started | - |
| 3. Empirical Testing & Profiling | v1.0 | 0/1 | Not started | - |
