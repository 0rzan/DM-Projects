import numpy as np
import pandas as pd
import ast
from collections import Counter
import csv
import tqdm

LABELS_TEST_PATH = 'data/DM2023_train_labels.tsv'
data = pd.read_csv(LABELS_TEST_PATH, sep='\t', header=None, names=['id', 'target', 'similar'], nrows=100)

# LABELS_TEST_PATH = 'data/DM2023_test_labels.tsv'
# data = pd.read_csv(LABELS_TEST_PATH, sep='\t', header=None, names=['id', 'similar'])

RESULT_PATH = 'data/DM2023_test_labels_result.tsv'
TEST_PATH = 'data/DM2023_test_docs.tsv'

# data['target'] = data['target'].apply(ast.literal_eval)
data['similar'] = data['similar'].apply(ast.literal_eval)

start_weight = 0.9
end_weight = 0.83

def guess_labels(data, difference_threshold=2.0):
    # Sort data by scores in descending order
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    
    selected_labels = []
    for i in range(len(sorted_data) - 1):
        label, score = sorted_data[i]
        next_score = sorted_data[i + 1][1]
        selected_labels.append(label)
        
        # Check the difference between current score and next score
        if score - next_score > difference_threshold:
            break
    
    # Add the last label if the list ends without breaking
    if len(sorted_data) > 0:
        selected_labels.append(sorted_data[-1][0])
    
    if len(selected_labels) == 0:
        selected_labels.append(sorted_data[0][0])

    if len(selected_labels) > 5:
        selected_labels = selected_labels[:5]

    return selected_labels

def save_labels(ids, data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for id, labels in zip(ids, data):
            # transform labels [K.3, F.m] into 'K.3,F.m'
            labels = ','.join(labels)
            res = [id, labels]
            writer.writerow(res)

def clean_up():
    df2 = pd.read_csv(TEST_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df2.columns = ['Filename', 'Description', 'Categories']
    
    df = pd.read_csv(RESULT_PATH, delimiter='\t', header=None, encoding='ISO-8859-1')
    df.columns = ['Filename', 'Categories']
    df = df.drop_duplicates(subset='Filename', keep='first')
    df = df.set_index('Filename').reindex(df2['Filename']).reset_index()
    # print(df.head())
    # print(df.shape)
    # save only Categories to Result
    categories_df = df[['Categories']]
    categories_df.to_csv(RESULT_PATH, sep='\t', index=False, header=False)

def main():
    result = []
    ids = []
    for idx in tqdm.tqdm(range(len(data))):
        row = data.iloc[idx]
        codes_list = row['similar']
        num_entries = len(codes_list)

        weights = np.linspace(start_weight, end_weight, num_entries)

        weighted_counter = Counter()

        for idx, codes in enumerate(codes_list):
            weight = weights[idx]
            # for code in codes.split(','):
            for code in codes:
                weighted_counter[code] += weight

        most_common_weighted_codes = weighted_counter.most_common()
        result.append(guess_labels(most_common_weighted_codes))
        ids.append(row['id'])
        print(guess_labels(most_common_weighted_codes))
        print(row['target'])
        print(row['similar'])
        return

    # save_labels(ids, result, RESULT_PATH)
    # clean_up()

if __name__ == '__main__':
    main()