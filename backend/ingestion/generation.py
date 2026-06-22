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

        prompt = f"""Tu es un assistant qui répond UNIQUEMENT à partir du contexte fourni ci-dessous.
Si la réponse n'est pas dans le contexte, dis que tu ne sais pas.

CONTEXTE:
{context_text}

QUESTION: {query}

RÉPONSE:"""
        return prompt

    def generate(self, query: str, context_chunks: list) -> str:
        prompt = self.build_prompt(query, context_chunks)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read())
        answer = result["content"][0]["text"]
        return answer
