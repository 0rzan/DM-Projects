import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pathlib import Path
import os
from dotenv import load_dotenv
import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

env_path = Path('..') / 'assignment2' / '.env'

load_dotenv(dotenv_path=env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_DATABASE_NAME = os.getenv("PINECONE_DATABASE_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_DATABASE_NAME)

TEST_PATH = 'data/DM2023_test_docs_embedded.tsv'
LABELS_TEST_PATH = 'data/DM2023_test_labels.tsv'

def query_by_vec(vec):
    response = index.query(namespace='default', vector=vec, include_metadata=True, top_k=14)
    return response

def process_row(row):
    embeddings = [float(x) for x in row['Embeddings'].strip('[]').split(',')]
    res = query_by_vec(embeddings)
    categories = [match['metadata']['categories'] for match in res['matches']]
    if not categories:
        return None
    return [row['Filename'], categories]

def main(start_row=0, batch_size=200, max_workers=8):
    df = pd.read_csv(TEST_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df.columns = ['Filename', 'Description', 'Categories', 'Embeddings']
    df = df.iloc[start_row:]


    with open(LABELS_TEST_PATH, mode='a', newline='', encoding='ISO-8859-1') as file:
        writer = csv.writer(file, delimiter='\t')

        for i in tqdm.tqdm(range(0, df.shape[0], batch_size)):
            batch = df.iloc[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {executor.submit(process_row, row): row for _, row in batch.iterrows()}
                for future in as_completed(future_to_row):
                    result = future.result()
                    if result is not None:
                        writer.writerow(result)

if __name__ == '__main__':
    start_row = 1
    main(start_row)