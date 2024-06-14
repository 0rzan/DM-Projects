import pandas as pd
import tqdm
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

env_path = Path('..') / 'assignment2' / '.env'

load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_DATABASE_NAME = os.getenv("PINECONE_DATABASE_NAME")

DATA_PATH = 'data/DM2023_training_docs_and_labels.tsv'

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_DATABASE_NAME)

def embed_and_prepare(row):
    embedding = OpenAIEmbeddings().embed_query(row['Description'])
    metadata = {
        'filename': row['Filename'],
        'description': row['Description'],
        'categories': row['Categories']
    }
    return (row['Filename'], embedding, metadata)

def upsert_batch(batch):
    index.upsert(batch, namespace='default')

def main(start_row=0):
    df = pd.read_csv(DATA_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df.columns = ['Filename', 'Description', 'Categories']
    df['Categories'] = df['Categories'].str.split(',')

    df = df.iloc[start_row:]

    insert_counter = 0
    batch_size = 250

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = deque()
        batch = []

        for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            future = executor.submit(embed_and_prepare, row)
            futures.append(future)
            if len(futures) >= batch_size:
                while futures and len(batch) < batch_size:
                    batch.append(futures.popleft().result())
                upsert_batch(batch)
                insert_counter += len(batch)
                batch = []
                if insert_counter >= 500:
                    time.sleep(20)
                    insert_counter = 0

        while futures:
            batch.append(futures.popleft().result())
            if len(batch) >= batch_size:
                upsert_batch(batch)
                insert_counter += len(batch)
                batch = []
                if insert_counter >= 500:
                    time.sleep(20)
                    insert_counter = 0

        if batch:
            upsert_batch(batch)
            insert_counter += len(batch)
            if insert_counter >= 500:
                time.sleep(20)

if __name__ == '__main__':
    start_row = 96950
    main(start_row=start_row)