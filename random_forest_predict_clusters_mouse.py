import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import df_ge_mouse
from utils import get_louvain_clusters_as_arrays, mapping_clusters_mouse
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Retrieve the clusters with the Louvain algorithm
ge_mouse_target = get_louvain_clusters_as_arrays(df_ge_mouse).astype(np.int)

# Map the mouse clusters obtained by the Louvain algorithm to the ones obtained by single-cell analysis
ge_mouse_target = np.vectorize(mapping_clusters_mouse.get)(ge_mouse_target)

# Resample both the data and the classes
ge_mouse_target = pd.DataFrame(ge_mouse_target)
oversample = SMOTE(sampling_strategy={0:2315, 1:1620, 2:1365})
df_ge_mouse, ge_mouse_target = oversample.fit_resample(df_ge_mouse, ge_mouse_target)

# Get the number of single-cells per class
print("The number of single-cells per class is:", ge_mouse_target.value_counts())

# PCA analysis and transform
pca = PCA(n_components=128)
df_ge_mouse_pca_components = pd.DataFrame(pca.fit_transform(df_ge_mouse))

# Create the column "labels" in df_ge_mouse_pca_components
df_ge_mouse_pca_components["labels"] = ge_mouse_target

# Drop the rows that have the label 2
df_ge_mouse_pca_components = df_ge_mouse_pca_components[df_ge_mouse_pca_components["labels"] != 2]

# Dataframe of the data
X = df_ge_mouse_pca_components.drop(columns=["labels"])

# Dataframe of the labels
Y = df_ge_mouse_pca_components.pop("labels")

# Split the data and the labels into training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random forest with PCA, mapped, without binarization with max_depth of 100 and n_estimators of 500
rf_cluster_1 = RandomForestClassifier(max_depth=100, n_estimators=500)
rf_cluster_1.fit(X_train, y_train)

# Store the metrics
accuracy_1 = round(accuracy_score(y_test, rf_cluster_1.predict(X_test)), 2)

print("The accuracy of the random forest without conversion of gene expressions to 1 is:", accuracy_1)
