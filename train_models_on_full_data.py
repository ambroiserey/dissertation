import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import df_ge_human, df_ge_mouse
from utils import get_louvain_clusters_as_arrays, mapping_clusters_human, mapping_clusters_mouse
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Retrieve the clusters with the Louvain algorithm
ge_mouse_target = get_louvain_clusters_as_arrays(df_ge_mouse).astype(np.int)

# Map the mouse clusters obtained with the Louvain algorithm to the ones obtained with single-cell analysis
ge_mouse_target = np.vectorize(mapping_clusters_mouse.get)(ge_mouse_target)

# Resample both the data and the classes
ge_mouse_target = pd.DataFrame(ge_mouse_target)
oversample = SMOTE(sampling_strategy={0:2315, 1:1620, 2:1365})
df_ge_mouse, ge_mouse_target = oversample.fit_resample(df_ge_mouse, ge_mouse_target)

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

# Mouse random forest with PCA, mapped, without binarization with max_depth of 100 and n_estimators of 500
rf_cluster_mouse = RandomForestClassifier(max_depth=100, n_estimators=500)
rf_cluster_mouse.fit(X, Y)

# Prepare a dataframe with the class
g_human_target = pd.read_csv("labels.csv", header=None).astype(np.int)

# Map the human clusters obtained with the Louvain algorithm to the ones obtained with single-cell analysis
g_human_target = np.vectorize(mapping_clusters_human.get)(g_human_target)

# PCA analysis and transform
pca = PCA(n_components=128)
df_ge_human_pca_components = pca.fit_transform(df_ge_human)

# Human random forest with PCA, mapped, without binarization with max_depth of 100 and n_estimators of 500
rf_cluster_human = RandomForestClassifier(max_depth=100, n_estimators=500)
rf_cluster_human.fit(df_ge_human_pca_components, g_human_target)

# This method was used to avoid training the model everytime the random forests are used
# The model is saved
if __name__ == "__main__":

    # Reshape the data
    labels_array = np.reshape(g_human_target, (-1, 1))

    # One-hot encode the labels
    enc = OneHotEncoder(handle_unknown="ignore")
    labels_array = enc.fit_transform(labels_array).toarray()

    # Transform the labels into a dataframe
    labels_df = pd.DataFrame(labels_array)

    # Keep the first 10,000 rows of labels_df
    labels_df = labels_df.head(11000)

    # Transform the data into a dataframe
    df_ge_human_pca_components = pd.DataFrame(df_ge_human_pca_components)

    # Keep the first 10,000 rows of df_ge_human_pca_components
    df_ge_human_pca_components = df_ge_human_pca_components.head(11000)

    oversample = SMOTE(sampling_strategy=0.7)

    # Resample both the datasets derived from df_learn and df_test
    X, Y = oversample.fit_resample(X, Y)

    if __name__ == "__main__":
        def ann_predict_clusters(input_shape):
            input = keras.layers.Input(shape=(input_shape,))

            x = keras.layers.Dense(128)(input)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.Dropout(0.5)(x)

            x = keras.layers.Dense(256)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.Dropout(0.5)(x)

            x = keras.layers.Dense(128)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.Dropout(0.5)(x)

            x = keras.layers.Dense(2)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Softmax()(x)

            model = keras.models.Model(input, x)

            return model


        # Instantiate the model
        ann_predict_clusters = ann_predict_clusters(128)

        # Define the loss
        optimizer = keras.optimizers.Adam()

        # Compile the model
        ann_predict_clusters.compile(loss="categorical_crossentropy", optimizer=optimizer,
                       metrics=[tf.keras.metrics.CategoricalAccuracy()])

        # Fit the model, with mapped, without binarization and for 128 principal components
        ann_predict_clusters.fit(x=df_ge_human_pca_components, y=labels_df.values, epochs=100, batch_size=128, shuffle=True)

        # Save the model to be used without having to train it again
        ann_predict_clusters.save("ann_predict_clusters")