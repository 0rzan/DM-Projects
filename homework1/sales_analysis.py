import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class SalesDataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.cleaned_data = None
        self.frequent_itemsets = None
        self.rules = None
        self.frequent_personel = None
        self.rules_personel = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def clean_data(self):
        self.data['dateTime'] = pd.to_datetime(self.data['dateTime'], unit='s')

    def find_association_itemset_rules(self):
        product_data = self.data.dropna(subset=['productID'])
        te = TransactionEncoder()
        te_ary = te.fit(product_data.groupby('transactionID')['productID'].apply(list)).transform(product_data.groupby('transactionID')['productID'].apply(list))
        df = pd.DataFrame(te_ary, columns=te.columns_)
        print(f"Transaction asscoiation rules shape: {df.shape}")
        self.frequent_itemsets = apriori(df, min_support=0.002, use_colnames=True)

        self.rules = association_rules(self.frequent_itemsets, metric="lift", min_threshold=1)
        print(f"Association rules shape: {self.rules.shape}")

    def find_association_personel_rules(self):
        personel_data = self.data.dropna(subset=['salesPersonID'])
        te = TransactionEncoder()
        te_ary = te.fit(personel_data.groupby('productID')['salesPersonID'].apply(list)).transform(personel_data.groupby('productID')['salesPersonID'].apply(list))
        df = pd.DataFrame(te_ary, columns=te.columns_)
        print(f"Personel asscoiation rules shape: {df.shape}")
        self.frequent_personel = apriori(df, min_support=0.003, use_colnames=True)

        self.rules_personel = association_rules(self.frequent_personel, metric="lift", min_threshold=1)
        print(f"Personel association rules shape: {self.rules_personel.shape}")

    def visualize_association_rules(self, filename, rules):
        if rules is None:
            print("No association rules found. Cannot create heatmap.")
            return

        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(map(str, list(x))))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(map(str, list(x))))

        rules_matrix = rules.pivot(index='antecedents', columns='consequents', values='lift')

        if rules_matrix.empty:
            print("No association rules matrix to display. Cannot create heatmap.")
            return

        plt.figure(figsize=(16, 12))
        sns.heatmap(rules_matrix, annot=True, cmap='coolwarm', fmt=".1f", annot_kws={"size": 7})
        plt.title('Heatmap of Association Rules (Lift values)')
        plt.xlabel('Consequents')
        plt.ylabel('Antecedents')
        plt.xticks(rotation=90)
        plt.tight_layout() 
        plt.savefig(filename)
        print(f"Association rules heatmap saved to {filename}")

    def visualize_sales_over_time(self, filename):
        self.data['date'] = self.data['dateTime'].dt.date

        daily_sales = self.data.groupby('date')['productValue'].sum().reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=daily_sales, x='date', y='productValue')
        plt.title('Daily Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.savefig(filename)