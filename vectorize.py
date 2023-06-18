from sentence_transformers import SentenceTransformer
import pandas as pd
import pinecone
from pinecone.core.client.exceptions import NotFoundException
from sklearn.metrics.pairwise import cosine_similarity


pinecone.init(api_key ="8dd7f9d3-2492-4506-b050-aae9af21066b", environment = "asia-southeast1-gcp-free")
pinecone.delete_index("example-index")
pinecone.create_index(name="pharma-index",metric="cosine")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#loading data
df = pd.read_csv('hazwaste.csv')

sentences = df['Description'].values

embeddings  = model.encode(sentences)

#index = pinecone.Index(index_name="pharma-index")

#for i, embedding in enumerate(embeddings):
 #   index.upsert(items={str(i): embedding})


#print(df.columns)
#print(embeddings)
#query = "quenching bath from oilbath with cyanides"

#query_embedding = model.encode([query])[0]
#results = index.query(queries =[query_embedding], top_k=5)

#for result in results.results:
#    print(result.id)
#similarities = cosine_similarity([query_embedding], embeddings)
#most_similar_index = similarities.argmax()
#print(sentences[most_similar_index])




