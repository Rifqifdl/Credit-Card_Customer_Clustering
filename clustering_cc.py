import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans

model = pickle.load(open('cc_clustering.sav', 'rb'))

df=pd.read_excel("CCCustomer_Cluster.xlsx")
X = df['Age','Credit','Duration']

st.title('German Credit Card Customer Clustering')

numClusters = st.slider("Select Number of Clusters", min_value=1, max_value=5, value=3)

model.n_clusters = numClusters
clusters = model.fit_predict(X)

st.write("Cluster Assignments:")
st.write(clusters)
