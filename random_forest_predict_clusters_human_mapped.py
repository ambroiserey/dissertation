import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import df_ge_human
from utils import mapping_clusters_human
from sklearn.decomposition import PCA

# Prepare a dataframe with the labels
ge_human_target = pd.read_csv("labels.csv", header=None).astype(np.int)

# Map the human clusters obtained by the Louvain algorithm to the ones obtained by single-cell analysis
ge_human_target = np.vectorize(mapping_clusters_human.get)(ge_human_target)

# Create a copy of df_ge_human to make modifications without impacting the original dataframe
df_ge_human_copy = df_ge_human

# PCA analysis and transform
pca = PCA(n_components=4000)
df_ge_human_pca_components = pca.fit_transform(df_ge_human_copy)

# Split the data and the labels into training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(df_ge_human_pca_components, ge_human_target, test_size=0.2,
                                                    random_state=42)

# Random forest with PCA, mapped, without binarization with max_depth of 100 and n_estimators of 500
rf_cluster_1 = RandomForestClassifier(max_depth=100, n_estimators=500)
rf_cluster_1.fit(X_train, y_train)

# Store the metrics
accuracy_1 = round(accuracy_score(y_test, rf_cluster_1.predict(X_test)), 2)

print("The accuracy of the random forest without conversion of gene expressions to 1 is:", accuracy_1)

# Random forest with PCA, mapped, without binarization with max_depth of 200 and n_estimators of 1,000
rf_cluster_2 = RandomForestClassifier(max_depth=200, n_estimators=1000)
rf_cluster_2.fit(X_train, y_train)

# Store the metrics
accuracy_2 = round(accuracy_score(y_test, rf_cluster_2.predict(X_test)), 2)

print("The accuracy of the random forest without conversion of gene expressions to 1 is:", accuracy_2)

# Create a copy of df_ge_human to make modifications without impacting the original dataframe
df_ge_human_copy = df_ge_human

# Binarize the data
df_ge_human_copy[df_ge_human_copy != 0] = 1

# PCA analysis and transform
pca = PCA(n_components=4000)
df_ge_human_pca_components = pca.fit_transform(df_ge_human_copy)

# Split the data and the labels into training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(df_ge_human_pca_components, ge_human_target, test_size=0.2,
                                                    random_state=42)

# Random forest with PCA, mapped, with binarization with max_depth of 100 and n_estimators of 500
rf_cluster_conv_1 = RandomForestClassifier(max_depth=100, n_estimators=500)
rf_cluster_conv_1.fit(X_train, y_train)

# Store the metrics
accuracy_conv_1 = round(accuracy_score(y_test, rf_cluster_conv_1.predict(X_test)), 2)

print("The accuracy of the random forest with conversion of gene expressions to 1 is:", accuracy_conv_1)

# Random forest with PCA, mapped, with binarization with max_depth of 200 and n_estimators of 1,000
rf_cluster_conv_2 = RandomForestClassifier(max_depth=200, n_estimators=1000)
rf_cluster_conv_2.fit(X_train, y_train)

# Store the metrics
accuracy_conv_2 = round(accuracy_score(y_test, rf_cluster_conv_2.predict(X_test)), 2)

print("The accuracy of the random forest with conversion of gene expressions to 1 is:", accuracy_conv_2)

# List of columns to build a dataframe with all the models results
columns_df = ["Without conv", "With conv"]

# Index of the dataframe
index_df = ["Acc for max d of 100 and n estimators of 500", "Acc for max d of 200 and n estimators of 1,000"]

# Create a dataframe with all the models results
df_scores = pd.DataFrame(
    list(zip([round(accuracy_1, 2), round(accuracy_2, 2)], [round(accuracy_conv_1, 2), round(accuracy_conv_2, 2)])),
    columns=columns_df)
df_scores.index = index_df

print(df_scores)
