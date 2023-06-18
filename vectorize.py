import pandas as pd
import openai
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

openai.api_key = "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"

def get_gpt_response(input:str):
    completion = openai.ChatCompletion.create(
        model="gpt-4",  # Replace with the actual model ID of GPT-4
        messages=[
            {"role": "system", "content": "You are a knowledgable assistant that specializes in medical, hazardous and pharmaceutical waste compliance."},
            {"role": "user", "content": input},
        ],
    )
    return completion.choices[0].message['content']


# Execute the query
# query = get_gpt_response("What should I query?")
# #query = "sludges from electroplating operations"
# query_embedding = model.encode([query])[0].tolist()  # Convert query_embedding to a list

# results = index.query(queries=[query_embedding], top_k=5)
# print("Query executed")
# for result in results.results:
#     print(result.id)

# similarities = cosine_similarity([query_embedding], embeddings)
# most_similar_index = similarities.argmax()
# print(sentences[most_similar_index])
# Execute the query
query = get_gpt_response("What should I query?")
query_embedding = model.encode([query])[0].tolist()  # Convert query_embedding to a list

results = index.query(queries=[query_embedding], top_k=5)

# Use GPT-4 to generate a response based on the query results
response = get_gpt_response("The top results are: " + ", ".join([result.id for result in results.results]))
print(response)

similarities = cosine_similarity([query_embedding], embeddings)
most_similar_index = similarities.argmax()
most_similar_sentence = sentences[most_similar_index]

# Use GPT-4 to generate a response based on the most similar sentence
response = get_gpt_response("The most similar sentence to the query is: " + most_similar_sentence)
print(response)

