import os
import shutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routes import router as api_router
from backend.config import FRONTEND_DIR, IS_VERCEL, DATA_SEGMENTS_PATH, BASE_DATA_SEGMENTS, METADATA_PATH, BASE_METADATA_PATH, MODELS_DIR

def ensure_serverless_artifacts():
    if IS_VERCEL:
        try:
            os.makedirs(os.path.dirname(DATA_SEGMENTS_PATH), exist_ok=True)
            os.makedirs(MODELS_DIR, exist_ok=True)
            if not os.path.exists(DATA_SEGMENTS_PATH) and os.path.exists(BASE_DATA_SEGMENTS):
                shutil.copy(BASE_DATA_SEGMENTS, DATA_SEGMENTS_PATH)
            if not os.path.exists(METADATA_PATH) and os.path.exists(BASE_METADATA_PATH):
                shutil.copy(BASE_METADATA_PATH, METADATA_PATH)
        except Exception as err:
            print(f"Serverless artifact initialization notice: {err}")

ensure_serverless_artifacts()

app = FastAPI(
    title="High-Yield Customer Segmentation Engine API",
    description="Production-Grade Multi-Algorithm Clustering & Customer Persona Platform",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix="/api")

# Mount Static Frontend assets if directory exists
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
@app.get("/{page_name:path}")
def serve_index(page_name: str = ""):
    if page_name:
        file_path = os.path.join(FRONTEND_DIR, page_name)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
            
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to Customer Segmentation API. Visit /docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
