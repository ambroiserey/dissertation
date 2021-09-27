from __future__ import division
import anndata
import scanpy as sc
import numpy as np
from sys import stdout
from sklearn.metrics import pairwise_kernels
from data_preprocessing import df_ge_human, df_ge_mouse

sc.settings.set_figure_params(dpi=80, facecolor="white")


# Transform adata into a umap visualization with the Leiden algorithm
def umap_leiden(df):
    adata = anndata.AnnData(df)
    sc.pp.neighbors(adata, n_neighbors=40, n_pcs=20)
    sc.tl.leiden(adata)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos="paga")
    sc.pl.umap(adata, color="leiden")
    sc.pl.umap


# Transform adata into a umap visualization with the Louvain algorithm
def umap(df):
    adata = anndata.AnnData(df)
    sc.pp.neighbors(adata, n_neighbors=40, n_pcs=20)
    sc.tl.louvain(adata)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos="paga")
    sc.pl.umap(adata, color="louvain")
    sc.pl.umap


# Retrieve as arrays the clusters from a gene expression matrix
def get_louvain_clusters_as_arrays(df):
    adata = anndata.AnnData(df)
    sc.pp.neighbors(adata, n_neighbors=40, n_pcs=20)
    sc.tl.louvain(adata)
    adata_array = np.array(adata.obs)
    adata_array = np.concatenate(adata_array)
    adata_array = adata_array.astype("U13")
    return adata_array


marker_genes = [
    "PTPRC",  # CD45
    "CD2", "CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD79A", "CD79B", "BLNK",  # B cell
    "CD38", "SDC1",  # plasma cell
    "CD14", "CD68", "CSF1R",  # mono/macro/DC
    "CCL5", "NKG7", "PRF1"  # NK cells
]

mapping_clusters_human = {0: 1, 1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 0, 11: 1, 12: 1}

# 0 = T cells CD8 and/or NK cells
# 1 = T cell CD4

mapping_clusters_mouse = {0: 0, 1: 0, 2: 1, 3: 2, 4: 0, 5: 2, 6: 1, 7: 2, 8: 2, 9: 2}

# 0 = T cells CD8 and/or NK cells
# 1 = T cell CD4
# 2 = myeloid cells

mapping_clusters_projected_mouse = {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1, 10:0, 11:1}

# 0 = T cells CD8 and/or NK cells
# 1 = T cell CD4

###

# All the code below directly comes from:
# https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py

###

def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
           1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
           2.0 / (m * n) * Kxy.sum()


def compute_null_distribution(K, m, n, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m + n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null


def compute_null_distribution_given_permutations(K, m, n, permutation,
                                                 iterations=None):
    """Compute the bootstrap null-distribution of MMD2u given
    predefined permutations.
    Note:: verbosity is removed to improve speed.
    """
    if iterations is None:
        iterations = len(permutation)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = permutation[i]
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    return mmd2u_null


def kernel_two_sample_test(X, Y, kernel_function='rbf', iterations=10000,
                           verbose=False, random_state=None, **kwargs):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.
    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    mmd2u_null = compute_null_distribution(K, m, n, iterations,
                                           verbose=verbose,
                                           random_state=random_state)
    p_value = max(1.0 / iterations, (mmd2u_null > mmd2u).sum() /
                  float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

    return mmd2u, mmd2u_null, p_value


if __name__ == "__main__":
    # Save the main dfs to csv to be imported to my google drive to be used in google colab to speed up the running time
    # of the code by running the deep learning models on Google Colab
    df_ge_human.to_csv("df_ge_human.csv")
    df_ge_mouse.to_csv("df_ge_mouse.csv")

    # Save the labels to speed up the running time of the code
    get_louvain_clusters_as_arrays(df_ge_human).to_csv("labels.csv")
