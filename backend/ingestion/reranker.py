from rank_bm25 import BM25Okapi

class Reranker:
    """
    RAG-12 : Re-ranking vectoriel + BM25
    RAG-21 : Recherche hybride avec RRF (Reciprocal Rank Fusion)
    """
    def __init__(self, vector_weight=0.6, bm25_weight=0.4, rrf_k=60):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def rerank(self, query: str, results: list) -> list:
        """Re-classe les résultats par score combiné vectoriel + BM25"""
        if not results:
            return results

        corpus = [r["content"].lower().split() for r in results]
        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        bm25_min = min(bm25_scores)
        bm25_max = max(bm25_scores)
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
        bm25_normalized = [(s - bm25_min) / bm25_range for s in bm25_scores]

        vector_scores = [r["score"] for r in results]
        vector_max = max(vector_scores) if max(vector_scores) > 0 else 1
        vector_normalized = [1 - (s / vector_max) for s in vector_scores]

        combined = []
        for i, r in enumerate(results):
            final_score = (
                self.vector_weight * vector_normalized[i] +
                self.bm25_weight * bm25_normalized[i]
            )
            combined.append({**r, "combined_score": round(final_score, 4)})

        combined.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, r in enumerate(combined):
            r["rank"] = i + 1

        print(f"Re-ranking effectue sur {len(combined)} resultats")
        return combined

    def hybrid_search(self, query: str, vector_results: list, all_chunks: list) -> list:
        """
        RAG-21 : Recherche hybride vectorielle + BM25 avec RRF
        - vector_results : résultats de la recherche k-NN OpenSearch
        - all_chunks : tous les chunks en mémoire pour la recherche BM25
        """
        if not all_chunks:
            return vector_results

        # Recherche BM25 sur tous les chunks
        corpus = [c.page_content.lower().split() for c in all_chunks]
        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        # Trier les chunks par score BM25 (top-K)
        k = len(vector_results)
        bm25_ranked = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # RRF : score = sum(1 / (rrf_k + rank))
        rrf_scores = {}

        # Scores depuis les résultats vectoriels
        for rank, r in enumerate(vector_results):
            key = r["content"][:50]
            rrf_scores[key] = rrf_scores.get(key, {"score": 0, "data": r})
            rrf_scores[key]["score"] += 1 / (self.rrf_k + rank + 1)

        # Scores depuis BM25
        for rank, (idx, _) in enumerate(bm25_ranked):
            chunk = all_chunks[idx]
            key = chunk.page_content[:50]
            if key not in rrf_scores:
                rrf_scores[key] = {
                    "score": 0,
                    "data": {
                        "rank": 0,
                        "score": 0,
                        "content": chunk.page_content[:200],
                        "metadata": chunk.metadata
                    }
                }
            rrf_scores[key]["score"] += 1 / (self.rrf_k + rank + 1)

        # Trier par score RRF final
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:k]

        final = []
        for i, item in enumerate(sorted_results):
            r = item["data"].copy()
            r["rank"] = i + 1
            r["combined_score"] = round(item["score"], 4)
            final.append(r)

        print(f"Recherche hybride RRF effectuee : {len(final)} resultats")
        return final
