# Context: Phase 1 (Worker Tracking Optimization)

## Domain
Resolve cross-camera tracking resets. Decouple worker ID from a specific camera zone so if they migrate between cameras, they are still under the 5-minute cooldown.

## Decisions

### 1. Spatial Relationship & Persistence
- **Decision**: Store tracked embeddings, RE-ID metadata, or global coordinates in a centralized fast-access store (like an isolated lightweight Redis cache or the existing SQLite database) rather than an in-memory array per stream.
- **Rationale**: Ensures that Worker ID 4 entering Camera A maps correctly to that identical worker migrating into Camera B.

### 2. Deduplication Logic Update
- **Decision**: Alter the Sentry cooldown tracker to query the unified database/store rather than the bounded local stream memory.
- **Rationale**: Prevents a worker triggering a new alarm the second they enter a new camera's line of sight, keeping alignment with the "Zero Alert Fatigue" core value.

## Deferred Ideas
- 3D tracking map rendering for site supervisors (Out of Scope for v1).

## Canonical Refs
- `.planning/ROADMAP.md` (Phase 1 goals and success criteria)
- `.planning/PROJECT.md` (Core Value alignment)
- `d:\SHEZAN\AI\intelligent-ppe-monitoring\thesis_update_draft.md` (Background research context)
