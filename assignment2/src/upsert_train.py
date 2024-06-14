import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import os
from dotenv import load_dotenv
import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone import Pinecone

env_path = Path('..') / 'assignment2' / '.env'
load_dotenv(dotenv_path=env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_DATABASE_NAME = os.getenv("PINECONE_DATABASE_NAME")
EMBED_PATH = 'data/DM2023_training_docs_and_labels_embedded.tsv'

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_DATABASE_NAME)

def upsert_batch(rows):
    vectors = []
    for _, row in rows.iterrows():
        metadata = {
            'filename': row['Filename'],
            'categories': row['Categories']
        }
        embeddings = [float(x) for x in row['Embeddings'].strip('[]').split(',')]
        vectors.append((row['Filename'], embeddings, metadata))
    index.upsert(vectors, namespace='default')

def main(start_row=0, batch_size=200, max_workers=8):
    df = pd.read_csv(EMBED_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df.columns = ['Filename', 'Categories', 'Embeddings']
    df = df.iloc[start_row:]

    for i in tqdm.tqdm(range(0, df.shape[0], batch_size)):
        batch = df.iloc[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(upsert_batch, batch)]
            for future in as_completed(futures):
                future.result()

if __name__ == '__main__':
    start_row = 0
    main(start_row)