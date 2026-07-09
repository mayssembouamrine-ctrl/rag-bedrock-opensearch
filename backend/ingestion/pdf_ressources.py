import os
import re
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    # RAG-11 : Configuration optimisée après tests comparatifs
    # Configs testées : (500/50), (1000/200), (1500/300)
    # Résultat : 500/50 donne plus de chunks précis
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def preprocess_text(self, text: str) -> str:
        """RAG-22 : Nettoyage et normalisation du texte avant chunking"""
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = text.strip()
        return text

    def split_into_chunks(self, pages: list) -> list:
        """Découpe le texte en chunks avec métadonnées par page"""
        all_chunks = []
        for page_num, (text, source) in enumerate(pages):
            chunks = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[{"source": source, "page": page_num + 1}]
            )
            all_chunks.extend(chunks)
        return all_chunks

    def process_pdf(self, pdf_path: str) -> list:
        """Pipeline complet : extraction + preprocessing + chunking"""
        print(f"📄 Traitement de : {pdf_path}")
        reader = PdfReader(pdf_path)
        pages = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_text = self.preprocess_text(page_text)
                pages.append((page_text, pdf_path))
        print(f"✅ Pages extraites : {len(pages)}")
        chunks = self.split_into_chunks(pages)
        print(f"✅ Chunks créés : {len(chunks)} chunks")
        return chunks

    def process_folder(self, folder_path: str) -> list:
        """Traite tous les PDFs d'un dossier"""
        all_chunks = []
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        if not pdf_files:
            print("⚠️ Aucun PDF trouvé dans le dossier")
            return []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)
        print(f"\n🎉 Total : {len(all_chunks)} chunks extraits de {len(pdf_files)} PDFs")
        return all_chunks
