from ingestion.query_expander import QueryExpander
from fastapi import APIRouter, HTTPException
from api.models.schemas import RetrieveRequest, RetrieveResponse, DocumentResult
from ingestion.pdf_ressources import PDFProcessor
from ingestion.opensearch_indexer import OpenSearchIndexer
from ingestion.reranker import Reranker

router = APIRouter()

processor = PDFProcessor()
indexer = OpenSearchIndexer()
reranker = Reranker(vector_weight=0.6, bm25_weight=0.4)
expander = QueryExpander()
chunks = processor.process_folder("data")
if chunks:
    indexer.index_chunks(chunks)

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    if not chunks:
        raise HTTPException(status_code=404, detail="Aucun document indexé")

    # RAG-28 : Expansion de la requête si elle est trop courte
    expanded_query = expander.expand(request.query)

    vector_results = indexer.search(expanded_query, k=request.top_k)
    hybrid_results = reranker.hybrid_search(expanded_query, vector_results, chunks)
    reranked = reranker.rerank(expanded_query, hybrid_results)

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
