from sentence_transformers import SentenceTransformer
import pandas as pd
import pinecone
from sklearn.metrics.pairwise import cosine_similarity


pinecone.init(api_key ="3a31c149-9146-4b7b-9c60-7415e1bc5c9c", environment = "asia-southeast1-gcp-free")
pinecone.create_index("pharma-index",768, "cosine")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#loading data
df = pd.read_csv('hazwaste.csv')

sentences = df['Description'].values

embeddings  = model.encode(sentences)

index = pinecone.Index(index_name="pharma-index")

for i, embedding in enumerate(embeddings):
    index.upsert(items={str(i): embedding})


#print(df.columns)
#print(embeddings)
query = "quenching bath from oilbath with cyanides"

query_embedding = model.encode([query])[0]
results = index.query(queries =[query_embedding], top_k=5)

for result in results.results:
    print(result.id)
#similarities = cosine_similarity([query_embedding], embeddings)
#most_similar_index = similarities.argmax()
#print(sentences[most_similar_index])




