import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans

model = pickle.load(open('cc_clustering.sav', 'rb'))

df=pd.read_excel("CCCustomer_Cluster.xlsx")
features = ['Age', 'Credit', 'Duration']
X = df[features]

st.title('German Credit Card Customer Clustering')

numClusters = st.slider("Select Number of Clusters", min_value=1, max_value=10, value=3)

model.n_clusters = numClusters
clusters = model.fit_predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X['Age'], X['Duration'], X['Credit'], c=clusters, cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('Duration')
ax.set_zlabel('Credit')

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
st.pyplot(fig)

