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

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def clean_data(self):
        # Clean the data by removing rows with missing values
        self.data = self.data.dropna(subset=['productID'])

        # Convert the dateTime column to a datetime format
        self.data['dateTime'] = pd.to_datetime(self.data['dateTime'], unit='s')

        # Add a new column for the sales person ID if it's available
        self.data['salesPersonID'] = self.data['salesPersonID'].fillna('Unknown')

        # Save the cleaned data to a new CSV file
        self.cleaned_data = self.data.copy()
        self.cleaned_data.to_csv('cleaned_data.csv', index=False)

    def find_association_rules(self):
        # Convert the data into a one-hot encoded format
        te = TransactionEncoder()
        te_ary = te.fit(self.data.groupby('transactionID')['productID'].apply(list)).transform(self.data.groupby('transactionID')['productID'].apply(list))
        df = pd.DataFrame(te_ary, columns=te.columns_)
        print(df)

        # Find frequent itemsets using the Apriori algorithm
        # self.frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)

        # # Generate association rules
        # self.rules = association_rules(self.frequent_itemsets, metric="lift", min_threshold=1)

    def visualize_association_rules(self, filename):
        # Create a heatmap of the association rules
        plt.figure(figsize=(18, 18))
        rules_matrix = self.rules.pivot(index='antecedents', columns='consequents', values='lift')
        sns.heatmap(rules_matrix, annot=True, cmap='coolwarm')
        plt.title('Heatmap of Association Rules')
        plt.xlabel('Consequents')
        plt.ylabel('Antecedents')
        plt.savefig(filename)

    def visualize_sales_over_time(self, filename):
        # Convert the dateTime column to a date format
        self.data['date'] = self.data['dateTime'].dt.date

        # Calculate the total sales for each date
        daily_sales = self.data.groupby('date')['productValue'].sum().reset_index()

        # Create a line plot of daily sales
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=daily_sales, x='date', y='productValue')
        plt.title('Daily Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.savefig(filename)