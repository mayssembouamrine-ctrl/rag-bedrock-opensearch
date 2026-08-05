# RAG System — AWS Bedrock + OpenSearch 🤖

Système de Retrieval-Augmented Generation (RAG) permettant d'interroger des documents PDF en langage naturel, avec des réponses générées par Claude Sonnet (AWS Bedrock).

## Architecture

```
PDF → Extraction → Chunking → Embeddings (Bedrock Titan) → OpenSearch Serverless
                                                                      ↓
User → React UI → FastAPI /chat → Query Expansion → Hybrid Search (k-NN + BM25 + RRF)
                                                                      ↓
                                              Claude Sonnet → Réponse + Sources
```

## Stack technique

| Composant | Technologie |
|-----------|------------|
| Infrastructure | AWS CDK (TypeScript) |
| Embeddings | AWS Bedrock Titan Embed v2 |
| Index vectoriel | Amazon OpenSearch Serverless |
| Génération | AWS Bedrock Claude Sonnet |
| Backend | FastAPI + Python |
| Frontend | React + Vite |
| Déploiement | EC2 (eu-north-1) |

## Prérequis

- Python 3.11+
- Node.js 18+
- AWS CLI configuré
- Compte AWS avec accès Bedrock

## Installation

### 1. Cloner le repo
```bash
git clone https://github.com/mayssembouamrine-ctrl/rag-bedrock-opensearch.git
cd rag-bedrock-opensearch
```

### 2. Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Frontend
```bash
cd frontend
npm install
```

## Configuration AWS

```bash
aws configure
# AWS Access Key ID: [votre clé]
# AWS Secret Access Key: [votre clé secrète]
# Default region: eu-north-1
```

## Lancement

### Backend (EC2 ou local avec credentials AWS)
```bash
cd backend
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm run dev
# Ouvrir http://localhost:5173
```

## Structure du projet

```
rag-bedrock-opensearch/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py          # Endpoint /chat
│   │   │   └── retrieve.py      # Endpoint /retrieve
│   │   └── models/
│   │       └── schemas.py
│   ├── ingestion/
│   │   ├── pdf_ressources.py    # Extraction + chunking
│   │   ├── opensearch_indexer.py # Indexation OpenSearch
│   │   ├── embeddings_bedrock.py # Embeddings Bedrock
│   │   ├── generation.py        # Génération Claude
│   │   ├── reranker.py          # Re-ranking BM25 + RRF
│   │   └── query_expander.py    # Expansion de requêtes
│   └── data/                    # Dossier des PDFs
├── frontend/
│   └── src/
│       ├── App.jsx              # Interface chat React
│       └── App.css
└── lib/
    └── rag-bedrock-opensearch-stack.ts  # CDK Stack
```

## Fonctionnalités clés

- ✅ Recherche hybride (k-NN + BM25 + RRF)
- ✅ Re-ranking multi-critères
- ✅ Expansion automatique des requêtes courtes
- ✅ Anti-hallucination (seuil de similarité + vérification de cohérence)
- ✅ Protection contre le dépassement de tokens (tiktoken)
- ✅ Indexation parallèle (ThreadPoolExecutor)
- ✅ Sources cliquables avec extraits originaux

## Auteur

**Mayssem Bouamrine** — Stage Data Science & IA  
Smartovate | Encadrant : Abdelkhalek Bakkari