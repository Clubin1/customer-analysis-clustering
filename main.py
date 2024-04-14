import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# function to find wcss
def find_wcss():
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)

# get datafram
df = pd.read_csv('marketing_campaign.csv', sep='\t')

# preprocessing
data = df.dropna()
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
categorical_columns = ['Education', 'Marital_Status']
encoder = LabelEncoder()
# more preprocessing, encode categorial columns
for i in range(len(categorical_columns)):
    column = categorical_columns[i]
    data[column] = encoder.fit_transform(data[column])
# more preprocessing, scale numeric types
numerics = data.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numerics])

# find wcss
wcss = find_wcss()

# plot for elbow method
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# use the most optimal k to form k clusters and print summary
k = 4  
km = KMeans(n_clusters=k, init='k-means++', random_state=42)
data['Cluster'] = km.fit_predict(data_scaled)
summary = data.groupby('Cluster').mean()
print(summary)