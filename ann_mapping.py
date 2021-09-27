from data_preprocessing import df_ge_human, df_ge_mouse
from utils import get_louvain_clusters_as_arrays, mapping_clusters_mouse, mapping_clusters_human
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from data_preprocessing import df_ge_human
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# Retrieve the clusters with the Louvain algorithm
ge_mouse_target = get_louvain_clusters_as_arrays(df_ge_mouse).astype(np.int)

# Map the mouse clusters obtained with the Louvain algorithm to the ones obtained with single-cell analysis
ge_mouse_target = np.vectorize(mapping_clusters_mouse.get)(ge_mouse_target)

# Create the column "labels" in df_ge_mouse
df_ge_mouse["labels"] = ge_mouse_target

# Get the number of single-cells per class
print("The number of single-cells per class is:", df_ge_mouse["labels"].value_counts())

# Drop the rows that have the label 2
df_ge_mouse = df_ge_mouse[df_ge_mouse["labels"] != 2]

# Transform the "labels" column of df_ge_mouse into a numpy array
ge_mouse_target = np.array(df_ge_mouse["labels"])

# Drop the "labels" column of df_ge_mouse
df_ge_mouse = df_ge_mouse.drop(columns=["labels"])

# Resample both the data and the classes
oversample = SMOTE(sampling_strategy=0.7)
df_ge_mouse, ge_mouse_target = oversample.fit_resample(df_ge_mouse, ge_mouse_target)

# Binarize the data
#df_ge_mouse[df_ge_mouse != 0] = 1

enc = OneHotEncoder(handle_unknown="ignore")

# Reshape the labels_array
ge_mouse_target = np.reshape(ge_mouse_target, (-1, 1))

# Transform the labels_array into a one-hot-encoder
ge_mouse_target = enc.fit_transform(ge_mouse_target).toarray()


# Define the model
def ann_predict_genes():
    input = keras.layers.Input(shape=(9391,))

    x = keras.layers.Dense(32)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(16)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Softmax()(x)

    model = keras.models.Model(input, x)

    return model

# Read the labels
human_labels = pd.read_csv("labels.csv", header=None).astype(np.int)

# Map the human clusters obtained with the Louvain algorithm to the ones obtained with single-cell analysis
human_labels = np.vectorize(mapping_clusters_human.get)(human_labels)

# Binarize the data
#df_ge_human[df_ge_human != 0] = 1

# Instantiate the model
ann_mapping = ann_predict_genes()

# Define the loss
optimizer = keras.optimizers.Adam()

# Compile the model
ann_mapping.compile(loss="categorical_crossentropy", optimizer=optimizer,
                    metrics=[tf.keras.metrics.BinaryAccuracy()])

# Fit the model
ann_mapping.fit(x=df_ge_mouse, y=ge_mouse_target, batch_size=64, epochs=25, shuffle=True)

# Save the model to be used without having to train it again
#ann_mapping.save("ann_mapping")

# Predict the cell-types of the human dataset
human_label_predictions = ann_mapping.predict(df_ge_human)

# Keep the highest probability and convert it to 1
# I followed:
# https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0
a = np.zeros_like(human_label_predictions)
a[np.arange(len(human_label_predictions)), human_label_predictions.argmax(1)] = 1

# Inverse transform the one-hot encoded labels
human_label_predictions = enc.inverse_transform(a)

# Concatenate the numpy array to flatten it
human_label_predictions = np.concatenate(human_label_predictions)

# Concatenate the numpy array to flatten it
labels = np.concatenate(human_labels)

# Element-wise comparison of the two numpy arrays
ann_mapping_accuracy = np.equal(human_label_predictions, labels)

print("The accuracy of the ANN mapping is:",
      round(np.count_nonzero(ann_mapping_accuracy) / df_ge_human.shape[0], 2) * 100)

# I followed:
# https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class

# Get the confusion matrix
confusion_matrix = confusion_matrix(labels, human_label_predictions)

# Now the normalize the diagonal entries
confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]

print("The accuracy of the ANN for respectively T cells CD8 and/or NK cells and T cell CD4 is:", confusion_matrix.diagonal())
