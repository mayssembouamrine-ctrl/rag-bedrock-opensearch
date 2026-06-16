from fastapi import FastAPI
from api.routes.retrieve import router

app = FastAPI(
    title="RAG API",
    description="API de recherche sémantique RAG avec Bedrock et OpenSearch",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "RAG API is running ✅"}
