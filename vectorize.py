from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#loading data
df = pd.read_csv('HazWasteSheet.csv')

sentences = df['Title'].values

embeddings  = model.encode(sentences)


