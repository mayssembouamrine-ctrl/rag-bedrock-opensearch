from fastapi import APIRouter, HTTPException
from api.models.schemas import RetrieveRequest, RetrieveResponse, DocumentResult
from ingestion.pdf_ressources import PDFProcessor
from ingestion.embeddings import EmbeddingsIndexer

router = APIRouter()

# Initialisation du pipeline au démarrage
processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
indexer = EmbeddingsIndexer()

# Charger et indexer les documents
chunks = processor.process_folder("data")
if chunks:
    indexer.index_chunks(chunks)

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """Endpoint de recherche sémantique"""
    
    if not chunks:
        raise HTTPException(status_code=404, detail="Aucun document indexé")
    
    # Recherche k-NN
    results = indexer.search(request.query, k=request.top_k)
    
    # Formatage des résultats
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