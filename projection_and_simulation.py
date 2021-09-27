import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
from data_preprocessing import df_ge_human, df_ge_mouse
from utils import umap, get_louvain_clusters_as_arrays, kernel_two_sample_test, marker_genes, mapping_clusters_projected_mouse
from sklearn.metrics import pairwise_distances, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from train_models_on_full_data import rf_cluster_human, rf_cluster_mouse
from ann_predict_clusters_human_mapped import enc
import anndata
import scanpy as sc

# Activate the eager mode
np_config.enable_numpy_behavior()

# Number of samples used to generate the data
number_of_samples = 3000

# Remove the empty rows
df_ge_human = df_ge_human.iloc[1:, :]
df_ge_human = df_ge_human.iloc[:-1, :]

df_ge_human = df_ge_human.sample(number_of_samples)
df_ge_human.reset_index

# Remove the empty rows
df_ge_mouse = df_ge_mouse.iloc[1:, :]
df_ge_mouse = df_ge_mouse.iloc[:-1, :]

df_ge_mouse = df_ge_mouse.sample(number_of_samples)
df_ge_mouse.reset_index

# PCA analysis and transform of the human dataset
pca_human = PCA(n_components=50)
pca_components_real = pca_human.fit_transform(df_ge_human)
df_pca_components_real_human = pd.DataFrame(pca_components_real)

# PCA analysis and transform of the mouse dataset
pca_mouse = PCA(n_components=50)
pca_components_real = pca_mouse.fit_transform(df_ge_mouse)
df_pca_components_real_mouse = pd.DataFrame(pca_components_real)

# Create a numpy array of 3,000 1 to be used as labels
ones = np.ones(number_of_samples)

# Create a numpy array of 3,000 0 to be used as labels
zeroes = np.zeros(number_of_samples)

# Concatenate the labels
labels = np.concatenate([ones, zeroes])

###

# Code to create the df_projected_mouse dataset

###

# Retrieve the clusters with the Louvain algorithm
#ge_mouse_target = get_louvain_clusters_as_arrays(df_projected_mouse).astype(np.int)

# Generate a sample of the mouse data
#mouse_sample = df_ge_mouse.sample(3000)

# Convert the sample to a tensor
#mouse_sample_tensor = tf.convert_to_tensor(mouse_sample.values, dtype = "float32")

# Load the generator generator_human_20000_shuffle_128_batch
#generator_human_20000_shuffle_128_batch = keras.models.load_model("generator_human_20000_shuffle_128_batch")

# Generate the genes with the generator_human_20000_shuffle_128_batch
#projected_genes_mouse = generator_human_20000_shuffle_128_batch.predict(mouse_sample_tensor)

# Transform the projected_genes_mouse into a dataframe
#df_projected_mouse = pd.DataFrame(projected_genes_mouse, columns = df_ge_mouse.columns)

# Normalize the data in order to clear the response
#df_projected_mouse[df_projected_mouse < 0] = 0

# Create the column "labels" in df_projected_mouses
#df_projected_mouse["labels"] = ge_mouse_target

# Drop the rows that have the label 2
#df_projected_mouse = df_projected_mouse[df_projected_mouse["labels"] != 2]

# Dataframe of the data
#df_projected_mouse = df_projected_mouse.drop(columns=["labels"])

# Save the dataframe to always work with the same projected data
#df_projected_mouse.to_csv("df_projected_mouse.csv")

###

# Read the projected data
df_projected_mouse = pd.read_csv("df_projected_mouse.csv")

# Drop the column "Unnamed: 0"
df_projected_mouse = df_projected_mouse.drop(columns=["Unnamed: 0"])

# Retrieve the clusters with the Louvain algorithm
ge_mouse_target = get_louvain_clusters_as_arrays(df_projected_mouse).astype(np.int)

# Map the projected data clusters obtained with the Louvain algorithm to the ones obtained with single-cell analysis
ge_mouse_target = np.vectorize(mapping_clusters_projected_mouse.get)(ge_mouse_target)

# Visualize the projected genes
umap(df_projected_mouse)

###

# The MMDs and two-kernel tests come from:
# https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py

###

# PCA analysis and transform of the projected data
pca_2 = PCA(n_components=50)
pca_components_projected_50 = pca_2.fit_transform(df_projected_mouse)
df_pca_components_projected = pd.DataFrame(pca_components_projected_50)

# Test between the human samples and the projected data
sigma2 = np.median(pairwise_distances(df_ge_human.values, df_projected_mouse.values, metric="euclidean")) ** 2
mmd2u_test, mmd2u_null_test, p_value_test = kernel_two_sample_test(df_ge_human.values, df_projected_mouse.values,
                                                                   kernel_function="rbf", gamma=1.0 / sigma2,
                                                                   verbose=True)
print(mmd2u_test, mmd2u_null_test, p_value_test)

# Concatenate df_pca_components_real_human and df_pca_components_projected
concat_human = pd.concat([df_pca_components_real_human, df_pca_components_projected])
concat_human.reset_index

# Split the concat and the labels into training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(concat_human, labels, test_size=0.2, random_state=42)

# Random forest to distinguish with human data
rf_check_1 = RandomForestClassifier(max_depth=200, n_estimators=500)
rf_check_1.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_test, rf_check_1.predict(X_test))

print("The auc of the random forest to distinguish between the human data and the projected data is:", round(auc(fpr, tpr), 2))

# Test between the mouse samples and the projected data
sigma2 = np.median(pairwise_distances(df_ge_mouse.values, df_projected_mouse.values, metric="euclidean")) ** 2
mmd2u_test, mmd2u_null_test, p_value_test = kernel_two_sample_test(df_ge_mouse.values, df_projected_mouse.values,
                                                                   kernel_function="rbf", gamma=1.0 / sigma2,
                                                                   verbose=True)
print(mmd2u_test, mmd2u_null_test, p_value_test)

# Concatenate df_pca_components_real_mouse and df_pca_components_projected
concat_mouse = pd.concat([df_pca_components_real_mouse, df_pca_components_projected])
concat_human.reset_index

# Split the concat and the labels into training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(concat_mouse, labels, test_size=0.2, random_state=42)

# Random forest to distinguish with mouse data
rf_check_1 = RandomForestClassifier(max_depth=200, n_estimators=500)
rf_check_1.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_test, rf_check_1.predict(X_test))

print("The auc of the random forest to distinguish between the mouse data and the projected data is:", round(auc(fpr, tpr), 2))

# Transform df_projected_mouse into anndata
adata_projected_mouse = anndata.AnnData(df_projected_mouse)

# Retrieve the 20 genes with the highest expression
sc.pl.highest_expr_genes(adata_projected_mouse, n_top=20)

# Generate a dotplot of marker genes per cluster
sc.pp.neighbors(adata_projected_mouse, n_neighbors=40, n_pcs=20)
sc.tl.louvain(adata_projected_mouse)
sc.tl.rank_genes_groups(adata_projected_mouse, "louvain", method="t-test")
sc.pl.rank_genes_groups(adata_projected_mouse, n_genes=20, sharey=False)
sc.pl.dotplot(adata_projected_mouse, marker_genes, groupby="louvain")

# PCA analysis and transform of the projected data for 128 principal components
pca_3 = PCA(n_components=128)
pca_components_projected_128 = pca_3.fit_transform(df_projected_mouse)
df_pca_components_projected_128 = pd.DataFrame(pca_components_projected_128)

# Load the ann_predict_clusters model
ann_predict_clusters = keras.models.load_model("ann_predict_clusters")

# Predict the cell-types of the projected data
ann_predictions = ann_predict_clusters.predict(df_pca_components_projected_128)

# Keep the highest probability between the two and convert it to 1
# I followed:
# https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0
a = np.zeros_like(ann_predictions)
a[np.arange(len(ann_predictions)), ann_predictions.argmax(1)] = 1

# Inverse transform the one-hot encoded labels
ann_labels = enc.inverse_transform(a)

# Concatenate the numpy array to flatten it
ann_labels = np.concatenate(ann_labels)

# Element-wise comparison of the two numpy arrays
ann_accuracy = np.equal(ge_mouse_target, ann_labels)

print("The accuracy of the ANN on the projected data is:", round(np.count_nonzero(ann_accuracy)/3000, 2)*100, "%")

# Predict the cell-types of the projected data with the random forest trained on the human data
random_forest_human_labels = rf_cluster_human.predict(df_pca_components_projected_128)

# Element-wise comparison of the two numpy arrays
rf_accuracy = np.equal(ge_mouse_target, random_forest_human_labels)

print("The accuracy of the random forest trained on the human data on the projected data is:", round(np.count_nonzero(rf_accuracy)/3000, 2)*100, "%")

# Predict the cell-types of the projected data with the random trained on the mouse data
random_forest_mouse_labels = rf_cluster_mouse.predict(df_pca_components_projected_128)

# Element-wise comparison of the two numpy arrays
rf_accuracy = np.equal(ge_mouse_target, random_forest_mouse_labels)

print("The accuracy of the random forest trained on the mouse data on the projected data is:", round(np.count_nonzero(rf_accuracy)/3000, 2)*100, "%")