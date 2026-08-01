from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT
from app.retrieval import load_corpus
from app.routers import chat, repo

app = FastAPI(title="HyperscaleES Explorer")
app.include_router(repo.router)
app.include_router(chat.router)


@app.on_event("startup")
def startup() -> None:
    load_corpus()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Mounted last so it only catches paths the API routers above didn't claim.
app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
