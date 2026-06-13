import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingsIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("⏳ Chargement du modèle d'embeddings...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        print("✅ Modèle chargé !")

    def generate_embeddings(self, chunks: list) -> np.ndarray:
        """Convertit les chunks en vecteurs"""
        texts = [chunk.page_content for chunk in chunks]
        print(f"⏳ Génération des embeddings pour {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        print(f"✅ Embeddings générés : shape {embeddings.shape}")
        return embeddings

    def index_chunks(self, chunks: list):
        """Indexe les chunks dans FAISS"""
        embeddings = self.generate_embeddings(chunks)
        
        # Créer l'index FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.chunks = chunks
        
        print(f"✅ {self.index.ntotal} vecteurs indexés dans FAISS")

    def search(self, query: str, k: int = 3) -> list:
        """Recherche k-NN : trouve les chunks les plus proches"""
        print(f"\n🔍 Recherche : '{query}'")
        
        # Convertir la question en vecteur
        query_vector = self.model.encode([query], convert_to_numpy=True)
        
        # Chercher les k plus proches
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "rank": i + 1,
                "score": float(distances[0][i]),
                "content": self.chunks[idx].page_content[:200],
                "metadata": self.chunks[idx].metadata
            })
        
        return results
    