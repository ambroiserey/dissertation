import pandas as pd
import numpy as np
import h5py

# Load the human gene expression matrix
human_raw_h5 = h5py.File("CRC_GSE108989_expression.h5")

# Get a HDF5 dataset of all the genes
genes_human = human_raw_h5["matrix"]["features"]["name"]

# Get a HDF5 dataset of the whole data
data_human = human_raw_h5["matrix"]["data"]

# Get a numpy array of the splits to apply to the whole data
splits_human = np.array(human_raw_h5["matrix"]["indptr"])

# Create a numpy array of the dense matrix
data_split_human = np.split(data_human, splits_human)

# Get a numpy array of the indices of each gene within a sparse matrix
indices_human = np.array(human_raw_h5["matrix"]["indices"])

# Create a numpy array of the indices of each gene within a sparse matrix per single-cell
indices_split_human = np.split(indices_human, splits_human)

# Initiate an empty list
data_list_human = []

# Transform the dense matrix in a sparse matrix
# Question asked by myself on Stack Overflow and answered partially by Mohammad (2021)
for (array, indices) in zip(data_split_human, indices_split_human):
    new_array = np.zeros(len(genes_human))
    np.put(new_array, indices, array)
    data_list_human.append(new_array)

# Transform data_list_human into a numpy array
data_array_human = np.array(data_list_human)

# Transform genes_human into a numpy array
genes_human_array = np.array(genes_human)

# Change the type of the data to clean it
genes_human_array = genes_human_array.astype("U13")

# Build df_ge_human with data_array_human as data and genes_human_array as columns
df_ge_human = pd.DataFrame(data_array_human, columns=[genes_human_array])

# Delete rows that contain less than 10 gene expressions
df_ge_human_values = df_ge_human.values
a = (df_ge_human_values[:, :-1] != 0).sum(1) > 10
df_ge_human = pd.DataFrame(df_ge_human_values[a], df_ge_human.index[a], df_ge_human.columns)

# Keep a df_ge_human with the original number of genes
df_ge_human_original = df_ge_human

# Initiate an empty list
to_change = []

# Clean the name of the columns in df_ge_human
for element in np.array(df_ge_human.columns):
    to_append = str(element).replace("('", "")
    to_append = to_append.replace("',)", "")
    to_change.append(to_append)
df_ge_human.columns = to_change

# Load the mouse gene expression matrix
mouse_raw_h5 = h5py.File("CRC_GSE112865_mouse_aPD1_expression.h5")

# Get a HDF5 dataset of all the genes
genes_mouse = mouse_raw_h5["matrix"]["features"]["name"]

# Get a HDF5 dataset of the whole data
data_mouse = mouse_raw_h5["matrix"]["data"]

# Get a numpy array of the splits to apply to the whole data
splits_mouse = np.array(mouse_raw_h5["matrix"]["indptr"])

# Create a numpy array of the dense matrix
data_split_mouse = np.split(data_mouse, splits_mouse)

# Get a numpy array of the indices of each gene within a sparse matrix
indices_mouse = np.array(mouse_raw_h5["matrix"]["indices"])

# Create a numpy array of the indices of each gene within a sparse matrix per single-cell
indices_split_mouse = np.split(indices_mouse, splits_mouse)

# Initiate an empty list
data_list_mouse = []

# Transform the dense matrix in a sparse matrix
for (array, indices) in zip(data_split_mouse, indices_split_mouse):
    new_array = np.zeros(len(genes_mouse))
    np.put(new_array, indices, array)
    data_list_mouse.append(new_array)

# Transform data_list_mouse into a numpy array
data_array_mouse = np.array(data_list_mouse)

# Transform genes_mouse into a numpy array
genes_mouse_array = np.array(genes_mouse)

# Change the type of the data to clean it
genes_mouse_array = genes_mouse_array.astype("U13")

# Read the mapping between human and mouse genes from mouse_human_homologues.csv
df_mapping = pd.read_csv(r"mouse_human_homologues.csv", sep=",")

# Rename the columns 10090 and 9606
df_mapping = df_mapping.rename(columns={"10090": "mouse", "9606": "human"})

# Create a numpy array with all the names of the genes of the human
human_mapping = df_mapping["human"].to_numpy()

# Create a numpy array with all the names of the genes of the mouse
mouse_mapping = df_mapping["mouse"].to_numpy()

# Initiate two empty lists
converted_genes_mouse = []
raw_genes_mouse = []

# Convert all the names of the genes of the mouse to human
for element in genes_mouse_array:
    if element in mouse_mapping:
        index = np.where(mouse_mapping == element)

        # Append all the raw names of the genes of the mouse
        raw_genes_mouse.append(element)

        # Append all the converted names of the genes of the mouse
        to_append = str(human_mapping[index])

        # Clean all the converted names of the genes of the mouse
        to_append = to_append.replace("['", "")
        to_append = to_append.replace("']", "")
        converted_genes_mouse.append(to_append)

# Build df_ge_mouse with data_array_mouse as data and genes_mouse_array as columns
df_ge_mouse = pd.DataFrame(data_array_mouse, columns=[genes_mouse_array])

# Delete rows that contain less than 10 gene expressions
df_ge_mouse_values = df_ge_mouse.values
b = (df_ge_mouse_values[:, :-1] != 0).sum(1) > 10
df_ge_mouse = pd.DataFrame(df_ge_mouse_values[b], df_ge_mouse.index[b], df_ge_mouse.columns)

# Initiate an empty list
to_keep = []

# Clean the name of the columns in df_ge_mouse
for element in np.array(df_ge_mouse.columns):
    to_append = str(element).replace("('", "")
    to_append = to_append.replace("',)", "")
    to_keep.append(to_append)

# Only keep the names of the genes of the mouse that can be mapped
cols = [col for col in to_keep if col in raw_genes_mouse]
df_ge_mouse = df_ge_mouse[cols]

# Replace the names of the genes of the mouse by the names of the genes of the human while keeping the same order
df_ge_mouse.columns = converted_genes_mouse

# Find the common genes between converted_genes_mouse and genes_human_array
common_genes = np.intersect1d(converted_genes_mouse, genes_human_array)

# Only keep the names of the genes that are in common to both numpy arrays
# Change the order of the names of the genes in both numpy arrays so they are the same
df_ge_human = df_ge_human[common_genes]
df_ge_mouse = df_ge_mouse[common_genes]
