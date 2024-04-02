from sales_analysis import SalesDataAnalyzer

analyzer = SalesDataAnalyzer('sample_pos_data.csv')
analyzer.load_data()
analyzer.clean_data()
analyzer.find_association_itemset_rules()
analyzer.find_association_personel_rules()
analyzer.visualize_association_rules('association_itemset_rules_plot.png', analyzer.rules)
analyzer.visualize_association_rules('association_personel_rules_plot.png', analyzer.rules_personel)
analyzer.visualize_sales_over_time('sales_over_time_plot.png')
