import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import os
from dotenv import load_dotenv
import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

env_path = Path('..') / 'assignment2' / '.env'
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEST_PATH = 'data/DM2023_test_docs.tsv'
EMBED_PATH = 'data/DM2023_test_docs_embedded.tsv'

embeddings = OpenAIEmbeddings()

def embed_text(text):
    return embeddings.embed_query(text)

def process_row(row):
    vec = embed_text(row['Description'])
    return [row['Filename'], row['Description'], row['Categories'], vec]

def main(start_row=0, batch_size=100, max_workers=4):
    df = pd.read_csv(TEST_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df.columns = ['Filename', 'Description', 'Categories']
    df = df.iloc[start_row:]

    with open(EMBED_PATH, mode='a', newline='', encoding='ISO-8859-1') as file:
        writer = csv.writer(file, delimiter='\t')
        
        if file.tell() == 0:
            writer.writerow(['Filename', 'Description', 'Categories', 'Embeddings'])

        for i in tqdm.tqdm(range(0, df.shape[0], batch_size)):
            batch = df.iloc[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {executor.submit(process_row, row): row for _, row in batch.iterrows()}
                for future in as_completed(future_to_row):
                    result = future.result()
                    writer.writerow(result)

if __name__ == '__main__':
    start_row = 0
    main(start_row)