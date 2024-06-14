import pandas as pd

EMBED_PATH = 'data/DM2023_test_docs_embedded.tsv'
TEST_PATH = 'data/DM2023_test_docs.tsv'

def main():
    df2 = pd.read_csv(TEST_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df2.columns = ['Filename', 'Description', 'Categories']
    # print first 5 rows of df
    print(df2.head())
    print(df2.tail())
    
    df = pd.read_csv(EMBED_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df.columns = ['Filename', 'Description', 'Categories', 'Embeddings']
    df = df.drop_duplicates(subset='Filename', keep='first')
    df = df.set_index('Filename').reindex(df2['Filename']).reset_index()
    # print length of df
    # print(len(df))
    # print last 5 rows of df
    print(df.head())
    print(df.tail())
    # duplicates_df = df[df.duplicated('Filename', keep=False)]
    # print(duplicates_df)


if __name__ == '__main__':
    main()