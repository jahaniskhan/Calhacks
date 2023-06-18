from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#loading data
df = pd.read_csv('HazardWasteSheet.csv')

sentences = df['Description'].values

embeddings  = model.encode(sentences)
print(df.columns)



