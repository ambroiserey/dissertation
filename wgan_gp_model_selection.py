import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
from data_preprocessing import df_ge_human
from utils import umap, kernel_two_sample_test
from sklearn.metrics import pairwise_distances, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import anndata
import scanpy as sc

# Activate the eager mode
np_config.enable_numpy_behavior()

# Number of samples used to generate the data
number_of_samples = 3000

# Create a dataframe with 3,000 samples from the human dataset
df_ge_human = df_ge_human.sample(number_of_samples)
df_ge_human.reset_index

# PCA analysis and transform
pca_1 = PCA(n_components=50)
pca_components_real = pca_1.fit_transform(df_ge_human)
df_pca_components_real = pd.DataFrame(pca_components_real)

# Create a numpy array of 3,000 1 to be used as labels
ones = np.ones(number_of_samples)

# Create a numpy array of 3,000 0 to be used as labels
zeroes = np.zeros(number_of_samples)

# Concatenate the labels
labels = np.concatenate([ones, zeroes])

# Generate a gaussian noise tensor
# In tensorflow, the base parameters are mean=0.0, stddev=1.0
gaussian_noise = tf.random.normal((number_of_samples, 9391))

### Tests for 10,000 epochs


# Load the generator generator_human_10000_shuffle_128_batch
generator_human_10000_shuffle_128_batch = keras.models.load_model("generator_human_10000_shuffle_128_batch")

# Generate the genes with the generator_human_10000_shuffle_128_batch
generated_genes_10000_shuffle_128_batch = generator_human_10000_shuffle_128_batch.predict(gaussian_noise)

# Transform the generated data into a dataframe
df_ge_10000_shuffle_128_batch = pd.DataFrame(generated_genes_10000_shuffle_128_batch, columns=df_ge_human.columns)

# Transform all negative values into 0
df_ge_10000_shuffle_128_batch[df_ge_10000_shuffle_128_batch < 0] = 0

# Visualize the generated genes
umap(df_ge_10000_shuffle_128_batch)

###

# The MMD and two-kernel tests come from:
# https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py

###

# Test between the human samples and the data predict the generator trained on 10,000 epochs
sigma2 = np.median(pairwise_distances(df_ge_human.values, df_ge_10000_shuffle_128_batch.values, metric="euclidean")) ** 2
mmd2u_test, mmd2u_null_test, p_value_test = kernel_two_sample_test(df_ge_human.values,
                                                                   df_ge_10000_shuffle_128_batch.values,
                                                                   kernel_function="rbf", gamma=1.0 / sigma2,
                                                                   verbose=True)
print(mmd2u_test, mmd2u_null_test, p_value_test)

# PCA analysis and transform
pca_2 = PCA(n_components=50)
pca_components_generated = pca_2.fit_transform(df_ge_10000_shuffle_128_batch)
df_pca_components_generated = pd.DataFrame(pca_components_generated)

# Concatenate df_ge_human and df_pca_components_generated with the generator trained on 10,000 epochs
concat = pd.concat([df_pca_components_real, df_pca_components_generated])
concat.reset_index

# Split the data and the labels into training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(concat, labels, test_size=0.2, random_state=42)

# Train and test the random forest
rf_check_1 = RandomForestClassifier(max_depth=200, n_estimators=500)
rf_check_1.fit(X_train, y_train)
fpr, tpr, thresholds = roc_curve(y_test, rf_check_1.predict(X_test))
print("The auc of the Random Forest for the generator_human_10000_shuffle_128_batch is: ", round(auc(fpr, tpr), 2))

# Transform df_ge_10000_shuffle_128_batch into anndata
adata_10000_shuffle_128_batch = anndata.AnnData(df_ge_10000_shuffle_128_batch)

# Retrieve the 20 genes with the highest expression
sc.pl.highest_expr_genes(adata_10000_shuffle_128_batch, n_top=20)

### Test for 20,000 epochs


# Load the generator generator_human_20000_shuffle_128_batch
generator_human_20000_shuffle_128_batch = keras.models.load_model("generator_human_20000_shuffle_128_batch")

# Generate the genes with the generator_human_20000_shuffle_128_batch
generated_genes_20000_shuffle_128_batch = generator_human_20000_shuffle_128_batch.predict(gaussian_noise)

# Transform the generated data into a dataframe
df_ge_20000_shuffle_128_batch = pd.DataFrame(generated_genes_20000_shuffle_128_batch, columns=df_ge_human.columns)

# Transform all negative values into 0
df_ge_20000_shuffle_128_batch[df_ge_20000_shuffle_128_batch < 0] = 0

# Visualize the generated genes
umap(df_ge_20000_shuffle_128_batch)

###

# The MMD and two-kernel tests come from:
# https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py

###

# Test between the human samples and the data predict the generator trained on 20,000 epochs
sigma2 = np.median(
    pairwise_distances(df_ge_human.values, df_ge_20000_shuffle_128_batch.values, metric="euclidean")) ** 2
mmd2u_test, mmd2u_null_test, p_value_test = kernel_two_sample_test(df_ge_human.values,
                                                                   df_ge_20000_shuffle_128_batch.values,
                                                                   kernel_function="rbf", gamma=1.0 / sigma2,
                                                                   verbose=True)
print(mmd2u_test, mmd2u_null_test, p_value_test)

# PCA analysis and transform
pca_3 = PCA(n_components=50)
pca_components_generated = pca_3.fit_transform(df_ge_20000_shuffle_128_batch)
df_pca_components_generated = pd.DataFrame(pca_components_generated)

# Concatenate df_ge_human and df_pca_components_generated with the generator trained on 20,000 epochs
concat = pd.concat([df_pca_components_real, df_pca_components_generated])
concat.reset_index

# Split the data and the labels into training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(concat, labels, test_size=0.2, random_state=42)

# Train and test the random forest
rf_check_1 = RandomForestClassifier(max_depth=200, n_estimators=500)
rf_check_1.fit(X_train, y_train)
fpr, tpr, thresholds = roc_curve(y_test, rf_check_1.predict(X_test))
print("The auc of the Random Forest for the generator_human_20000_shuffle_128_batch is: ", round(auc(fpr, tpr), 2))

# Transform df_ge_20000_shuffle_128_batch into anndata
adata_20000_shuffle_128_batch = anndata.AnnData(df_ge_20000_shuffle_128_batch)

# Retrieve the 20 genes with the highest expression
sc.pl.highest_expr_genes(adata_20000_shuffle_128_batch, n_top=20)

# Test for 30.000 epochs


# Load the generator generator_human_30000_shuffle_128_batch
generator_human_30000_shuffle_128_batch = keras.models.load_model("generator_human_30000_shuffle_128_batch")

# Generate the genes with the generator_human_30000_shuffle_128_batch
generated_genes_30000_shuffle_128_batch = generator_human_30000_shuffle_128_batch.predict(gaussian_noise)

# Transform the generated data into a dataframe
df_ge_30000_shuffle_128_batch = pd.DataFrame(generated_genes_30000_shuffle_128_batch, columns=df_ge_human.columns)

# Transform all negative values into 0
df_ge_30000_shuffle_128_batch[df_ge_30000_shuffle_128_batch < 0] = 0

# Visualize the generated genes
umap(df_ge_30000_shuffle_128_batch)

###

# The MMD and two-kernel tests come from:
# https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py

###

# Test between the human samples and the data predict the generator trained on 30,000 epochs
sigma2 = np.median(
    pairwise_distances(df_ge_human.values, df_ge_30000_shuffle_128_batch.values, metric="euclidean")) ** 2
mmd2u_test, mmd2u_null_test, p_value_test = kernel_two_sample_test(df_ge_human.values,
                                                                   df_ge_30000_shuffle_128_batch.values,
                                                                   kernel_function="rbf", gamma=1.0 / sigma2,
                                                                   verbose=True)
print(mmd2u_test, mmd2u_null_test, p_value_test)

# PCA analysis and transform
pca_4 = PCA(n_components=50)
pca_components_generated = pca_4.fit_transform(df_ge_30000_shuffle_128_batch)
df_pca_components_generated = pd.DataFrame(pca_components_generated)

# Concatenate df_ge_human and df_pca_components_generated with the generator trained on 30,000 epochs
concat = pd.concat([df_pca_components_real, df_pca_components_generated])
concat.reset_index

# Split the data and the labels into training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(concat, labels, test_size=0.2, random_state=42)

# Train and test the random forest
rf_check_1 = RandomForestClassifier(max_depth=200, n_estimators=500)
rf_check_1.fit(X_train, y_train)
fpr, tpr, thresholds = roc_curve(y_test, rf_check_1.predict(X_test))
print("The auc of the Random Forest for the generator_human_30000_shuffle_128_batch is: ", round(auc(fpr, tpr), 2))

# Transform df_ge_30000_shuffle_128_batch into anndata
adata_30000_shuffle_128_batch = anndata.AnnData(df_ge_30000_shuffle_128_batch)

# Retrieve the 20 genes with the highest expression
sc.pl.highest_expr_genes(adata_30000_shuffle_128_batch, n_top=20)
