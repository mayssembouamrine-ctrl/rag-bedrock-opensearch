from fastapi import APIRouter, HTTPException
from api.models.schemas import RetrieveRequest, RetrieveResponse, DocumentResult
from ingestion.pdf_ressources import PDFProcessor
from ingestion.opensearch_indexer import OpenSearchIndexer
from ingestion.reranker import Reranker

router = APIRouter()

processor = PDFProcessor()
indexer = OpenSearchIndexer()
reranker = Reranker(vector_weight=0.6, bm25_weight=0.4)

chunks = processor.process_folder("data")
if chunks:
    indexer.index_chunks(chunks)

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    if not chunks:
        raise HTTPException(status_code=404, detail="Aucun document indexé")

    # Étape 1 : Recherche vectorielle k-NN
    vector_results = indexer.search(request.query, k=request.top_k)

    # Étape 2 : Recherche hybride RRF (RAG-21)
    hybrid_results = reranker.hybrid_search(request.query, vector_results, chunks)

    # Étape 3 : Re-ranking final (RAG-12)
    reranked = reranker.rerank(request.query, hybrid_results)

    documents = []
    for r in reranked:
        documents.append(DocumentResult(
            rank=r["rank"],
            score=round(r.get("combined_score", r.get("score", 0)), 4),
            content=r["content"],
            source=r["metadata"].get("source", "inconnu"),
            page=r["metadata"].get("page", 0)
        ))

    return RetrieveResponse(
        query=request.query,
        results=documents,
        total=len(documents)
    )
