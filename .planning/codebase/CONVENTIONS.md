# Code Conventions

## Code Style
- **Type Hinting**: Python files use type hinting extensively (especially with FastAPI and Pydantic).
- **Asynchronous Code**: Strong use of `async`/`await` in FastAPI endpoints and backend services for I/O operations (like `async_sam_verifier.py`).
- **Data Validation**: Heavily relies on Pydantic models to parse incoming requests.

## Error Handling
- **API Errors**: FastApi `HTTPException` raises are common for returning error codes.

## General Patterns
- Service injection or service classes wrapping specific ML functionalities (Yolo, Sam).
