from ingestion.pdf_ressources import PDFProcessor
from ingestion.embeddings import EmbeddingsIndexer

def main():
    # Étape 1 : Extraction et chunking (RAG-33)
    processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    chunks = processor.process_folder("data")

    if not chunks:
        print("❌ Aucun chunk trouvé")
        return

    # Étape 2 : Embeddings et indexation (RAG-34)
    indexer = EmbeddingsIndexer()
    indexer.index_chunks(chunks)

    # Étape 3 : Test de recherche k-NN
    query = "Qu'est-ce que le format PDF ?"
    results = indexer.search(query, k=3)

    print("\n--- Résultats de la recherche ---")
    for result in results:
        print(f"\n📌 Résultat {result['rank']}")
        print(f"   Score    : {result['score']:.4f}")
        print(f"   Contenu  : {result['content']}...")
        print(f"   Metadata : {result['metadata']}")

if __name__ == "__main__":
    main()
    