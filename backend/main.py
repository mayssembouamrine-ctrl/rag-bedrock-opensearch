from ingestion.pdf_ressources import PDFProcessor

def main():
    processor = PDFProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Traiter tous les PDFs du dossier data/
    chunks = processor.process_folder("data")
    
    # Afficher un aperçu
    if chunks:
        print("\n--- Aperçu du premier chunk ---")
        print(f"Contenu : {chunks[0].page_content[:200]}...")
        print(f"Métadonnées : {chunks[0].metadata}")

if __name__ == "__main__":
    main()
