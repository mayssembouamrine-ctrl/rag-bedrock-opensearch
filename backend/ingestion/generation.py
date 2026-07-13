import boto3
import json

class ResponseGenerator:
    def __init__(self, model_id="eu.anthropic.claude-sonnet-4-5-20250929-v1:0", region="eu-north-1"):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def build_prompt(self, query: str, context_chunks: list) -> str:
        context_text = "\n\n".join([
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['content']}"
            for chunk in context_chunks
        ])
        prompt = f"""Tu es un assistant strict qui répond UNIQUEMENT à partir du contexte fourni ci-dessous.

RÈGLES :
- Réponds en français en te basant UNIQUEMENT sur le contexte fourni ci-dessous.
- Si l'information est clairement présente dans le contexte, réponds directement et précisément.
- Si l'information n'est PAS dans le contexte, dis : "Je ne sais pas, cette information n'est pas présente dans les documents disponibles."
- Ne fais aucune supposition au-delà du contexte.

CONTEXTE:
{context_text}

QUESTION: {query}

RÉPONSE:"""
        return prompt

    def verify_answer(self, answer: str, context_chunks: list) -> bool:
        """RAG-29 : Vérifie que la réponse est cohérente avec le contexte"""
        if "je ne sais pas" in answer.lower():
            return True
        full_context = " ".join([c["content"].lower() for c in context_chunks])
        answer_words = [w.lower() for w in answer.split() if len(w) > 4]
        if not answer_words:
            return True
        words_in_context = sum(1 for w in answer_words if w in full_context)
        ratio = words_in_context / len(answer_words)
        print(f"Verification coherence : {ratio:.0%} des mots dans le contexte")
        return ratio >= 0.3

    def generate(self, query: str, context_chunks: list) -> str:
        prompt = self.build_prompt(query, context_chunks)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        result = json.loads(response["body"].read())
        answer = result["content"][0]["text"]

        # RAG-29 : Vérification de cohérence
        if not self.verify_answer(answer, context_chunks):
            print("Reponse non coherente — remplacement par message standard")
            answer = "Je ne sais pas, cette information n'est pas suffisamment présente dans les documents disponibles."

        return answer

