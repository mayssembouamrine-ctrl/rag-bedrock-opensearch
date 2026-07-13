import time
import boto3
import json
from concurrent.futures import ThreadPoolExecutor
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

class OpenSearchIndexer:
    def __init__(self, host="autakszsi4wqpra9op02.eu-north-1.aoss.amazonaws.com", region="eu-north-1", index_name="rag-chunks", embedding_model_id="amazon.titan-embed-text-v2:0"):
        self.index_name = index_name
        self.embedding_model_id = embedding_model_id
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, "aoss", session_token=credentials.token)
        self.client = OpenSearch(hosts=[{"host": host, "port": 443}], http_auth=awsauth, use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection, timeout=30)

    def embed_text(self, text: str):
        body = json.dumps({"inputText": text})
        response = self.bedrock_client.invoke_model(modelId=self.embedding_model_id, body=body)
        result = json.loads(response["body"].read())
        return result["embedding"]

    def create_index_if_not_exists(self, dimension=1024):
        if self.client.indices.exists(index=self.index_name):
            print("Index existant conserve")
            return
        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "embedding": {"type": "knn_vector", "dimension": dimension},
                    "content": {"type": "text"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"}
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=index_body)
        print("Index cree")
        time.sleep(15)
        print("Index pret")

    def index_chunks(self, chunks: list):
        self.create_index_if_not_exists()
        print(f"Indexation parallele de {len(chunks)} chunks...")

        def index_one(chunk):
            embedding = self.embed_text(chunk.page_content)
            doc = {
                "embedding": embedding,
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "inconnu"),
                "page": chunk.metadata.get("page", 0)
            }
            self.client.index(index=self.index_name, body=doc)

        # RAG-30 : Parallélisation avec 3 threads simultanés
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(index_one, chunks)

        print(f"{len(chunks)} chunks indexes en parallele dans OpenSearch Serverless") 
    def search(self, query: str, k: int = 3):
        query_vector = self.embed_text(query)
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }
        response = self.client.search(index=self.index_name, body=search_body)
        results = []
        for rank, hit in enumerate(response["hits"]["hits"]):
            results.append({
                "rank": rank + 1,
                "score": 1 - hit["_score"],
                "content": hit["_source"]["content"][:200],
                "metadata": {
                    "source": hit["_source"]["source"],
                    "page": hit["_source"]["page"]
                }
            })
        return results
