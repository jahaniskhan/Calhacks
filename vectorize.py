from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#loading data
df = pd.read_csv('hazwaste.csv')

sentences = df['Description'].values

embeddings  = model.encode(sentences)
print(df.columns)
print(embeddings)
query = "quenching bath from oilbath with syanides"
query_embedding = model.encode([query])[0]
similarities = cosine_similarity([query_embedding], embeddings)
most_similar_index = similarities.argmax()
print(sentences[most_similar_index])




