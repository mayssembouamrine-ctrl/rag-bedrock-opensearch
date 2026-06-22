import boto3
import json
import numpy as np

class BedrockEmbeddingsIndexer:
    def __init__(self, model_id="amazon.titan-embed-text-v2:0", region="eu-north-1"):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.vectors = []
        self.chunks = []

    def embed_text(self, text: str) -> np.ndarray:
        """Génère un embedding via Bedrock Titan"""
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body
        )
        result = json.loads(response["body"].read())
        return np.array(result["embedding"], dtype=np.float32)

    def index_chunks(self, chunks: list):
        """Indexe les chunks en générant leurs embeddings"""
        print(f"⏳ Génération des embeddings Bedrock pour {len(chunks)} chunks...")
        self.vectors = []
        for chunk in chunks:
            vec = self.embed_text(chunk.page_content)
            self.vectors.append(vec)
        self.chunks = chunks
        print(f"✅ {len(self.vectors)} vecteurs indexés (Bedrock Titan)")

    def search(self, query: str, k: int = 3) -> list:
        """Recherche k-NN par similarité cosinus"""
        print(f"\n🔍 Recherche : '{query}'")
        query_vec = self.embed_text(query)

        # Calcul de distance cosinus pour chaque vecteur indexé
        scores = []
        for i, vec in enumerate(self.vectors):
            cos_sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            distance = 1 - cos_sim  # plus petit = plus proche
            scores.append((distance, i))

        scores.sort(key=lambda x: x[0])
        top_k = scores[:k]

        results = []
        for rank, (distance, idx) in enumerate(top_k):
            results.append({
                "rank": rank + 1,
                "score": float(distance),
                "content": self.chunks[idx].page_content[:200],
                "metadata": self.chunks[idx].metadata
            })
        return results

