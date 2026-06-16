from pydantic import BaseModel
from typing import List

class RetrieveRequest(BaseModel):
    """Structure de la requête entrante"""
    query: str          # La question de l'utilisateur
    top_k: int = 3      # Nombre de documents à retourner

class DocumentResult(BaseModel):
    """Structure d'un document retourné"""
    rank: int           # Position dans les résultats
    score: float        # Score de similarité
    content: str        # Contenu du chunk
    source: str         # Nom du fichier source
    page: int           # Numéro de page

class RetrieveResponse(BaseModel):
    """Structure de la réponse"""
    query: str
    results: List[DocumentResult]
    total: int