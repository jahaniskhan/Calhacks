import pandas as pd
import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Initialize Pinecone
pinecone.init(api_key="67abdf79-1c2d-4c83-a449-3af343b6b576", environment="us-west1-gcp-free")
print("Pinecone initialized")
pinecone.delete_index("pharma-index")


# Create the index
index_name = "pharma-index"
pinecone.create_index(index_name, dimension=768, metric="cosine")
print("Index created")
time.sleep(30)

# Load the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load the data
df = pd.read_csv('hazwaste.csv')
print("Data loaded")
sentences = df['Description'].values
epa_numbers = df['EPA'].values

# Encode sentences
embeddings = model.encode(sentences)
embeddings = np.array(embeddings)  # Convert to numpy array
print("Sentences encoded")

# Define the sample documents with embeddings and metadata
sample_docs = []
for i, embedding in enumerate(embeddings):
    sample_doc = {
        "id": str(epa_numbers[i]),
        "values": embedding.tolist()
    }
    sample_docs.append(sample_doc)

# Upsert the sample documents into the Pinecone index
index = pinecone.Index(index_name)
index.upsert(sample_docs, namespace="example_namespace")
print("Items upserted to index")

# Execute the query
query = "sludges from electroplating operations"
query_embedding = model.encode([query])[0].tolist()  # Convert query_embedding to a list

results = index.query(queries=[query_embedding], top_k=5)
print("Query executed")
for result in results.results:
    print(result.id)

similarities = cosine_similarity([query_embedding], embeddings)
most_similar_index = similarities.argmax()
print(sentences[most_similar_index])
