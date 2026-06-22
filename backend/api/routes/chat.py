from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from api.routes.retrieve import indexer, chunks
from ingestion.generation import ResponseGenerator

router = APIRouter()
generator = ResponseGenerator()

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint de génération de réponse RAG"""

    if not chunks:
        raise HTTPException(status_code=404, detail="Aucun document indexé")

    # Étape 1 : Retrieval (réutilise RAG-36)
    results = indexer.search(request.query, k=request.top_k)

    context_chunks = []
    sources = []
    for r in results:
        context_chunks.append({
            "content": r["content"],
            "source": r["metadata"].get("source", "inconnu"),
            "page": r["metadata"].get("page", 0)
        })
        sources.append(f"{r['metadata'].get('source')} - page {r['metadata'].get('page')}")

    # Étape 2 : Generation (Bedrock Claude)
    answer = generator.generate(request.query, context_chunks)

    return ChatResponse(
        answer=answer,
        sources=list(set(sources))
    )

