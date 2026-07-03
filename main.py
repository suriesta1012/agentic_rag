"""
Entry point — runs the FastAPI application.

    python main.py

or

    uvicorn api.app:app --reload
"""

import uvicorn
from api.app import app  # noqa: F401  (imported so uvicorn can find it)

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
