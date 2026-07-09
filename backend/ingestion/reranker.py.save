from rank_bm25 import BM25Okapi

class Reranker:
    """
    RAG-12 : Re-ranking des résultats par score combiné vectoriel + BM25
    - Score vectoriel : distance k-NN (plus petit = meilleur)
    - Score BM25 : fréquence des mots-clés de la question dans le chunk
    - Score final : combinaison pondérée des deux
    """

    def __init__(self, vector_weight=0.6, bm25_weight=0.4):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def rerank(self, query: str, results: list) -> list:
        """Re-classe les résultats par score combiné vectoriel + BM25"""

        if not results:
            return results

        # Prépare les textes pour BM25
        corpus = [r["content"].lower().split() for r in results]
        bm25 = BM25Okapi(corpus)

        # Score BM25 pour la question
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        # Normalise les scores BM25 entre 0 et 1
        bm25_min = min(bm25_scores)
        bm25_max = max(bm25_scores)
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
        bm25_normalized = [(s - bm25_min) / bm25_range for s in bm25_scores]
        # Score vectoriel normalisé (on inverse car plus petit = meilleur)
        vector_scores = [r["score"] for r in results]
        vector_max = max(vector_scores) if max(vector_scores) > 0 else 1
        vector_normalized = [1 - (s / vector_max) for s in vector_scores]

        # Score combiné final
        combined = []
        for i, r in enumerate(results):
            final_score = (
                self.vector_weight * vector_normalized[i] +
                self.bm25_weight * bm25_normalized[i]
            )
            combined.append({**r, "combined_score": round(final_score, 4)})

        # Trier par score combiné décroissant (plus grand = meilleur)
        combined.sort(key=lambda x: x["combined_score"], reverse=True)

        # Réassigner les rangs
        for i, r in enumerate(combined):
            r["rank"] = i + 1

        print(f"✅ Re-ranking effectué sur {len(combined)} résultats")
        return combined
