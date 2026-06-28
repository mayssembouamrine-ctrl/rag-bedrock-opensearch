from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from api.routes.retrieve import indexer, chunks
from ingestion.generation import ResponseGenerator

router = APIRouter()
generator = ResponseGenerator()

SIMILARITY_THRESHOLD = 0.6  # Au-delà de ce score, on considère le chunk non pertinent

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceDetail(BaseModel):
    source: str
    page: int
    extract: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDetail]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint de génération de réponse RAG avec filtrage anti-hallucination"""

    if not chunks:
        raise HTTPException(status_code=404, detail="Aucun document indexé")

    # Étape 1 : Retrieval (réutilise RAG-36)
    results = indexer.search(request.query, k=request.top_k)

    # 🛡️ Filtrage anti-hallucination : on ne garde que les chunks pertinents
    relevant_results = [r for r in results if r["score"] < SIMILARITY_THRESHOLD]

    # Si aucun chunk n'est suffisamment pertinent → réponse directe sans appeler le LLM
    if not relevant_results:
        return ChatResponse(
            answer="Je ne sais pas. Aucune information pertinente n'a été trouvée dans la base de connaissances pour répondre à cette question.",
            sources=[]
        )

    context_chunks = []
    sources_seen = {}
    for r in relevant_results:
        context_chunks.append({
            "content": r["content"],
            "source": r["metadata"].get("source", "inconnu"),
            "page": r["metadata"].get("page", 0)
        })
        key = f"{r['metadata'].get('source')}-{r['metadata'].get('page')}"
        if key not in sources_seen:
            sources_seen[key] = SourceDetail(
                source=r["metadata"].get("source", "inconnu"),
                page=r["metadata"].get("page", 0),
                extract=r["content"]
            )

    # Étape 2 : Generation (Bedrock Claude)
    answer = generator.generate(request.query, context_chunks)

    return ChatResponse(
        answer=answer,
        sources=list(sources_seen.values())
    )
