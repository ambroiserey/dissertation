import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import csv
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from data_preprocessing import df_ge_human, df_ge_human_original, df_ge_mouse
from utils import umap_leiden, umap, kernel_two_sample_test, marker_genes

# Compute the mean of the whole df_ge_human dataset
mean_human = df_ge_human.values.mean()
print("The mean of the human gene expressions is:", round(mean_human, 2))

# Compute the std of the whole df_ge_human dataset
std_human = df_ge_human.values.std()
print("The standard deviation of the human gene expressions is:", round(std_human, 2))

# Compute the mean of the whole df_ge_mouse dataset
mean_mouse = df_ge_mouse.values.mean()
print("The mean of the mouse gene expressions is:", round(mean_mouse, 2))

# Compute the std of the whole df_ge_mouse dataset
std_mouse = df_ge_mouse.values.std()
print("The standard deviation of the mouse gene expressions is:", round(std_mouse, 2))

# Build a dataframe from df_ge_human with less single-cells
df_ge_human_reduced = df_ge_human.sample(n=4454)

# Build a dataframe with the same number of single-cells from the human and from the mouse
df_ge_concat_reduced = pd.concat([df_ge_human_reduced, df_ge_mouse])

# Reset the index of de_ge_concat_reduced
df_ge_concat_reduced = df_ge_concat_reduced.reset_index()

# Build a dataframe with the original number of single-cells from the human and from the mouse
df_ge_concat = pd.concat([df_ge_human, df_ge_mouse])

# Reset the index of df_ge_concat
df_ge_concat = df_ge_concat.reset_index()

# Initiate a list of all the dataframes I want to visualize
df_list_leiden = [df_ge_human, df_ge_mouse]

# Visualize a umap with the leiden algorithm for each dataframe in df_list_leiden
for df in df_list_leiden:
    umap_leiden(df)

# Create a df_ge_human with 5000 genes instead of 9391
df_ge_human_features_reduced = df_ge_human[df_ge_human.columns.to_series().sample(5000)]

# Initiate a list of all the dataframes I want to visualize
df_list = [df_ge_human_original, df_ge_human, df_ge_human_features_reduced, df_ge_mouse, df_ge_concat_reduced,
           df_ge_concat]

# Visualize a umap for each dataframe in df_list
for df in df_list:
    umap(df)

# Transform df_ge_human into anndata
adata_human = anndata.AnnData(df_ge_human)

# Retrieve the 20 genes with the highest expression for the human
sc.pl.highest_expr_genes(adata_human, n_top=20)

# Generate a dotplot of marker genes per cluster for the human
sc.pp.neighbors(adata_human, n_neighbors=40, n_pcs=20)
sc.tl.louvain(adata_human)
sc.tl.rank_genes_groups(adata_human, "louvain", method="t-test")
sc.pl.rank_genes_groups(adata_human, n_genes=20, sharey=False)
sc.pl.dotplot(adata_human, marker_genes, groupby="louvain")

# Transform df_ge_human into anndata
adata_mouse = anndata.AnnData(df_ge_mouse)

# Retrieve the 20 genes with the highest expression for the mouse
sc.pl.highest_expr_genes(adata_mouse, n_top=20)

# Generate a dotplot of marker genes per cluster for the mouse
sc.pp.neighbors(adata_mouse, n_neighbors=40, n_pcs=20)
sc.tl.louvain(adata_mouse)
sc.tl.rank_genes_groups(adata_mouse, "louvain", method="t-test")
sc.pl.rank_genes_groups(adata_mouse, n_genes=20, sharey=False)
sc.pl.dotplot(adata_mouse, marker_genes, groupby="louvain")

# Prepare the human samples with a rather low number of rows to lower run time
human_samples = df_ge_human.sample(1000)
human_samples.reset_index
human_samples = human_samples.values

# Prepare the mouse samples with a rather low number of rows to lower run time
mouse_samples = df_ge_mouse.sample(1000)
mouse_samples.reset_index
mouse_samples = mouse_samples.values

###

# The MMD and two-kernel tests come from:
# https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py

###

# Test that it works correctly by using the same sample
sigma2 = np.median(pairwise_distances(human_samples, human_samples, metric='euclidean')) ** 2
mmd2u_test, mmd2u_null_test, p_value_test = kernel_two_sample_test(human_samples, human_samples, kernel_function="rbf",
                                                                   gamma=1.0 / sigma2, verbose=True)
print(mmd2u_test, mmd2u_null_test, p_value_test)

# Test that it between the human and the mouse
sigma2 = np.median(pairwise_distances(human_samples, human_samples, metric='euclidean')) ** 2
mmd2u, mmd2u_null, p_value = kernel_two_sample_test(human_samples, mouse_samples, kernel_function="rbf",
                                                    gamma=1.0 / sigma2, verbose=True)
print(mmd2u, mmd2u_null, p_value)

# Load the human gene expression matrix
human_raw_h5 = h5py.File("CRC_GSE108989_expression.h5")

# Get a HDF5 dataset of all the barcodes
barcodes_human = human_raw_h5["matrix"]["barcodes"]

# Transform the HDF5 dataset into a numpy array
barcodes_human_array = np.array(barcodes_human)

# Change the type of the data to clean it
barcodes_human_array = barcodes_human_array.astype("U13")

# Build df_barcodes_human
df_barcodes_human = pd.DataFrame(barcodes_human_array, columns=["Cell"])

# Load the file with meta info of the human
tsv_file_human = open("CRC_GSE108989_CellMetainfo_table.tsv")
cell_types_human = csv.reader(tsv_file_human, delimiter="\t")

# Build df_cell_types
df_cell_types = pd.DataFrame(cell_types_human)

# Transform the first row into the columns
df_cell_types.columns = df_cell_types.iloc[0]

# Delete the columns
df_cell_types = df_cell_types[1:]

print("The two arrays are identical:", np.array_equal(df_cell_types["Cell"].to_numpy(), barcodes_human_array))

# Check whether it is interesting joining df_barcodes_human and df_cell_types on the Cell column
result = pd.merge(df_barcodes_human, df_cell_types, on="Cell")

print("The number of similar barcodes is:", len(result.to_numpy()))

# PCA analysis
pca = PCA(n_components=5000)
pca.fit(df_ge_human)

# Plot the cumulative explained variance of the first 5,000 principal components of the human dataset
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.title("Cumulative explained variance for the human dataset")
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance")
plt.show()

# PCA analysis
pca = PCA(n_components=3000)
pca.fit(df_ge_mouse)

# Plot the cumulative explained variance of the first 3,000 principal components of the mouse dataset
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.title("Cumulative explained variance for the mouse dataset")
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance")
plt.show()