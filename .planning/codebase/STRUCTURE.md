# Structure

## Directory Layout
Top-level breakdown is fairly standard for a full-stack Python/Node app:
- `backend/`: FastAPI Application
  - `backend/api/routes`: REST endpoints organized by domain (`detection`, `history`, `upload`, `video`).
  - `backend/services`: Heavy ML/AI Logic (`sentry.py`, `judge.py`, `hybrid_detector.py`, `stream_processor.py`, `violation_tracker.py`, `email_service.py`).
  - `backend/models`: SQLAlchemy definitions.
  - `backend/database`: DB connections.
- `frontend/`: React + Vite UI
  - `frontend/src/api`: Axios clients.
  - `frontend/src/components`: React components.
  - `frontend/src/styles`: CSS/Tailwind definitions.
- `.agent/`, `.planning/`: Workflow and context directories for the GSD Agent framework.

## Naming Conventions
- **Python**: Variables and functions are generally `snake_case`, classes are `PascalCase`. Files are `snake_case` (e.g. `hybrid_detector.py`).
- **Javascript/React**: Files are mostly `PascalCase` when they return components (`App.jsx`), variables and functions are `camelCase`. `jsx` extensions are explicitly used.
