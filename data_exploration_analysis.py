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
# Seasonality and Regional Trends
data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'])

data['Year'] = data['Transaction_Date'].dt.year
data['Month'] = data['Transaction_Date'].dt.month

# Revenue trends by month
monthly_revenue = data.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Month', y='Revenue', hue='Year', data=monthly_revenue, marker='o')
plt.title("Revenue Trends Over the Months")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.show()

# Total revenue by region
region_revenue = data.groupby('Region')['Revenue'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Revenue', data=region_revenue, palette='wistia')
plt.title("Total Revenue Across Regions")
plt.xticks(rotation=45)
plt.ylabel("Total Revenue")
plt.show()

