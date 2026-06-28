from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import tiktoken
from api.routes.retrieve import indexer, chunks
from ingestion.generation import ResponseGenerator

router = APIRouter()
generator = ResponseGenerator()

SIMILARITY_THRESHOLD = 0.6   # Seuil anti-hallucination (RAG-41)
MAX_TOP_K = 5                 # Plafond dur sur le nombre de chunks (RAG-42)
MAX_CONTEXT_TOKENS = 4000     # Limite de sécurité pour le contexte envoyé à Claude (RAG-42)

encoding = tiktoken.get_encoding("cl100k_base")  # Encodage standard compatible Claude/GPT

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

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
    """Endpoint de génération de réponse RAG avec protections anti-hallucination et anti-dépassement de tokens"""

    if not chunks:
        raise HTTPException(status_code=404, detail="Aucun document indexé")

    # 🛡️ RAG-42 : on plafonne top_k, peu importe ce que demande le client
    safe_top_k = min(request.top_k, MAX_TOP_K)

    # Étape 1 : Retrieval (réutilise RAG-36)
    results = indexer.search(request.query, k=safe_top_k)

    # 🛡️ RAG-41 : filtrage anti-hallucination
    relevant_results = [r for r in results if r["score"] < SIMILARITY_THRESHOLD]

    if not relevant_results:
        return ChatResponse(
            answer="Je ne sais pas. Aucune information pertinente n'a été trouvée dans la base de connaissances pour répondre à cette question.",
            sources=[]
        )

    # 🛡️ RAG-42 : on ajoute les chunks un par un tant qu'on reste sous la limite de tokens
    context_chunks = []
    sources_seen = {}
    total_tokens = 0

    for r in relevant_results:
        chunk_text = r["content"]
        chunk_tokens = count_tokens(chunk_text)

        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            print(f"⚠️ Limite de tokens atteinte ({total_tokens}/{MAX_CONTEXT_TOKENS}) — chunk ignoré")
            break

        total_tokens += chunk_tokens
        context_chunks.append({
            "content": chunk_text,
            "source": r["metadata"].get("source", "inconnu"),
            "page": r["metadata"].get("page", 0)
        })

        key = f"{r['metadata'].get('source')}-{r['metadata'].get('page')}"
        if key not in sources_seen:
            sources_seen[key] = SourceDetail(
                source=r["metadata"].get("source", "inconnu"),
                page=r["metadata"].get("page", 0),
                extract=chunk_text
            )

    print(f"✅ Contexte final : {total_tokens} tokens sur {len(context_chunks)} chunks")

    # Étape 2 : Generation (Bedrock Claude)
    answer = generator.generate(request.query, context_chunks)

    return ChatResponse(
        answer=answer,
        sources=list(sources_seen.values())
    )

