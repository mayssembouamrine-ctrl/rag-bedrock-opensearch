from fastapi import FastAPI
from api.routes.retrieve import router as retrieve_router
from api.routes.chat import router as chat_router

app = FastAPI(
    title="RAG API",
    description="API de recherche sémantique et génération RAG avec Bedrock",
    version="1.0.0"
)

app.include_router(retrieve_router)
app.include_router(chat_router)

@app.get("/")
async def root():
    return {"message": "RAG API is running ✅"}

