#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# Load dataset
data = pd.read_csv('synthetic_ecommerce_data.csv')

# Statistics
print("Dataset Info:")
print(data.info())

print("\nDescriptive Statistics:")
print(data.describe())

#%%
# Visual Analysis
num_features = ['Units_Sold', 'Discount_Applied', 'Revenue', 'Clicks', 'Impressions', 'Conversion_Rate', 'Ad_CTR', 'Ad_CPC', 'Ad_Spend']

data[num_features].hist(bins=15, figsize=(15, 10), layout=(3, 3), color='blue')
plt.suptitle("Feature Distributions")
plt.show()

#%%
# Correlation matrix & HeatMap
num_data = data.select_dtypes(include=['float64', 'int64'])

correlation_matrix = num_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='gist_heat', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#%%
