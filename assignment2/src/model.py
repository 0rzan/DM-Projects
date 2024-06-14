import numpy as np
import pandas as pd
import ast
from collections import Counter

LABELS_TEST_PATH = 'data/DM2023_train_labels.tsv'
data = pd.read_csv(LABELS_TEST_PATH, sep='\t', header=None, names=['id', 'target', 'similar'], nrows=100)

# LABELS_TEST_PATH = 'data/DM2023_test_labels.tsv'
# data = pd.read_csv(LABELS_TEST_PATH, sep='\t', header=None, names=['id', 'similar'], nrows=100)

data['target'] = data['target'].apply(ast.literal_eval)
data['similar'] = data['similar'].apply(ast.literal_eval)

start_weight = 0.9
end_weight = 0.83
for idx, row in data.iterrows():
    codes_list = row['similar']
    num_entries = len(codes_list)

    # Calculate linear weights
    weights = np.linspace(start_weight, end_weight, num_entries)

    # Initialize a counter to store weighted frequencies
    weighted_counter = Counter()

    # Apply weights and calculate weighted frequency
    for idx, codes in enumerate(codes_list):
        weight = weights[idx]
        # for code in codes.split(','):
        for code in codes:
            weighted_counter[code] += weight

    # Get the most common codes by weighted frequency
    most_common_weighted_codes = weighted_counter.most_common()

    # Display the most probable codes
    print(most_common_weighted_codes)
    print(row['target'])