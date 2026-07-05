from fastapi import APIRouter, HTTPException
from api.models.schemas import RetrieveRequest, RetrieveResponse, DocumentResult
from ingestion.pdf_ressources import PDFProcessor
from ingestion.opensearch_indexer import OpenSearchIndexer

router = APIRouter()

# Initialisation du pipeline au démarrage
processor = PDFProcessor()  # RAG-11 : utilise les valeurs optimisées (500/50)
indexer = OpenSearchIndexer()

# Charger et indexer les documents
chunks = processor.process_folder("data")
if chunks:
    indexer.index_chunks(chunks)

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """Endpoint de recherche sémantique via OpenSearch Serverless"""

    if not chunks:
        raise HTTPException(status_code=404, detail="Aucun document indexé")

    results = indexer.search(request.query, k=request.top_k)

    documents = []
    for r in results:
        documents.append(DocumentResult(
            rank=r["rank"],
            score=round(r["score"], 4),
            content=r["content"],
            source=r["metadata"].get("source", "inconnu"),
            page=r["metadata"].get("page", 0)
        ))

    return RetrieveResponse(
        query=request.query,
        results=documents,
        total=len(documents)
    )
