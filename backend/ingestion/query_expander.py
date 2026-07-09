import boto3
import json

class QueryExpander:
    """
    RAG-28 : Expansion de requête pour améliorer les résultats sur les requêtes courtes
    Si la requête fait moins de 5 mots, on la enrichit avec le LLM
    """
    def __init__(self, model_id="eu.anthropic.claude-sonnet-4-5-20250929-v1:0", region="eu-north-1"):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.min_words = 5  # Seuil : requêtes < 5 mots seront enrichies

    def expand(self, query: str) -> str:
        """Enrichit une requête courte avec le LLM"""
        words = query.strip().split()

        # Si la requête est déjà assez longue, on la retourne telle quelle
        if len(words) >= self.min_words:
            print(f"Query suffisamment longue ({len(words)} mots) — pas d'expansion")
            return query

        print(f"Query courte ({len(words)} mots) — expansion en cours...")

        prompt = f"""Tu es un assistant de recherche documentaire.
L'utilisateur a posé cette courte requête : "{query}"

Reformule cette requête en une question complète et détaillée (15-25 mots maximum) 
pour améliorer la recherche dans une base documentaire.
Réponds UNIQUEMENT avec la question reformulée, sans explication ni ponctuation supplémentaire."""

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read())
        expanded = result["content"][0]["text"].strip()
        print(f"Query originale : '{query}'")
        print(f"Query enrichie  : '{expanded}'")
        return expanded
