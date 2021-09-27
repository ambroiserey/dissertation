import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from data_preprocessing import df_ge_human
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Get the labels for each single-cell
labels = pd.read_csv("labels.csv", header=None).astype(np.int)

# Reshape the data
labels_array = np.reshape(labels, (-1, 1))

# One-hot encode the labels
enc = OneHotEncoder(handle_unknown="ignore")
labels_array = enc.fit_transform(labels_array).toarray()

# Transform the labels into a dataframe
labels_df = pd.DataFrame(labels_array)

# Keep the first 10,000 rows of labels_df
labels_df = labels_df.head(11000)

# Keep the first 10,000 rows of df_ge_human
df_ge_human = df_ge_human.head(11000)

# Create a copy of df_ge_human to make modifications without impacting the original dataframe
df_ge_human_copy = df_ge_human

# PCA analysis and transform
pca = PCA(n_components=4000)
df_ge_human_pca_components = pca.fit_transform(df_ge_human_copy)

# Get the number of classes
number_of_classes = np.amax(labels) + 1

if __name__ == "__main__":


    def ann_predict_clusters(input_shape):
        input = keras.layers.Input(shape=(input_shape,))

        x = keras.layers.Dense(256)(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Dense(512)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Dense(256)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Dense(number_of_classes)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Softmax()(x)

        model = keras.models.Model(input, x)

        return model


    # Split the data and the labels into training and testing dataframes
    X_train, X_test, y_train, y_test = train_test_split(df_ge_human_pca_components, labels_df.values, test_size=0.2,
                                                        random_state=42)

    # Instantiate the model
    model1 = ann_predict_clusters(4000)

    # Define the loss
    optimizer = keras.optimizers.Adam()

    # Compile the model
    model1.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # Fit the model, with non-mapped, without binarization
    model1.fit(x=X_train, y=y_train, epochs=100, batch_size=128, shuffle=True)

    # Generate metrics
    scores = model1.evaluate(X_test, y_test, verbose=0)

    print("The categorical accuracy of the ann without conversion is:", round(scores[1], 2))

    # Create a copy of df_ge_human to make modifications without impacting the original dataframe
    df_ge_human_copy = df_ge_human

    # Binarize the data
    df_ge_human_copy[df_ge_human_copy != 0] = 1

    # PCA analysis and transform
    pca = PCA(n_components=4000)
    df_ge_human_pca_components = pca.fit_transform(df_ge_human_copy)

    # Split the data and the labels into training and testing dataframes
    X_train, X_test, y_train, y_test = train_test_split(df_ge_human_pca_components, labels_df.values, test_size=0.2,
                                                        random_state=42)

    # Instantiate the model
    model2 = ann_predict_clusters(4000)

    # Define the loss
    optimizer = keras.optimizers.Adam()

    # Compile the model
    model2.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # Fit the model, with non-mapped, with binarization
    model2.fit(x=X_train, y=y_train, epochs=100, batch_size=128, shuffle=True)

    # Generate metrics
    scores = model2.evaluate(X_test, y_test, verbose=0)

    print("The categorical accuracy of the ann with conversion is:", round(scores[1], 2))
