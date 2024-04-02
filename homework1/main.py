from sales_analysis import SalesDataAnalyzer

analyzer = SalesDataAnalyzer('sample_pos_data.csv')
analyzer.load_data()
analyzer.clean_data()
analyzer.find_association_rules()
# analyzer.visualize_association_rules('association_rules_plot.png')
# analyzer.visualize_sales_over_time('sales_over_time_plot.png')
