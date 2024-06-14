import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pathlib import Path
import os
from dotenv import load_dotenv
import tqdm
import ast

env_path = Path('..') / 'assignment2' / '.env'

load_dotenv(dotenv_path=env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_DATABASE_NAME = os.getenv("PINECONE_DATABASE_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_DATABASE_NAME)

TEST_PATH = 'data/DM2023_test_docs.tsv'
EMBED_PATH = 'data/DM2023_test_docs_embedded.tsv'

embeddings = OpenAIEmbeddings()

def query_from_pinecone(vec):
    response = index.query(namespace='default', vector=[vec], top_k=10, include_metadata=True)

    return response

def main():
    df2 = pd.read_csv(TEST_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df2.columns = ['Filename', 'Description', 'Categories']
    
    df = pd.read_csv(EMBED_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df.columns = ['Filename', 'Description', 'Categories', 'Embeddings']
    df = df.drop_duplicates(subset='Filename', keep='first')
    df = df.set_index('Filename').reindex(df2['Filename']).reset_index()

    chuj = 0
    for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]): 
        chuj += 1
        embed = ast.literal_eval(row['Embeddings'])
        res = query_from_pinecone(embed)
        for match in res['matches']:
            print(match['metadata']['categories'])
            # print(match['metadata']['description'])
            print(match['score'])
        # if chuj == 100:
        break

if __name__ == '__main__':
    main()