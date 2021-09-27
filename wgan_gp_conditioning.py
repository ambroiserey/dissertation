import numpy as np
from data_preprocessing import df_ge_human
import tensorflow as tf
from tensorflow import keras
from utils import get_louvain_clusters_as_arrays
import random

#The WGAN gp is inspired from Nain (2020). His model is one of the WGAN gp examples from keras.

# Number of genes
latent_dimension = 9391

# Get the labels for each single-cell
human_labels = get_louvain_clusters_as_arrays(df_ge_human).astype(np.int)

# Transform the labels into a tensor
human_labels_tensors = tf.convert_to_tensor(human_labels[:11000])

# Get the number of classes
number_of_classes = np.amax(human_labels) + 1

# Define the generator
def generator():
    input_label = keras.layers.Input(shape=(1,))

    x = keras.layers.Dense(128)(input_label)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)

    input_genes = keras.layers.Input(shape=(latent_dimension,))

    concatenate = keras.layers.Concatenate()([input_genes, x])

    x = keras.layers.Dense(128)(concatenate)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(1024)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(latent_dimension)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    generator_model = keras.models.Model([input_genes, input_label], x, name="generator")
    generator_model.summary()
    return generator_model


# Instantiate the generator
generator = generator()


# Define the discriminator
def discriminator():
    input_label = keras.layers.Input(shape=(1,))

    x = keras.layers.Dense(128)(input_label)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)

    input_genes = keras.layers.Input(shape=(latent_dimension,))

    concatenate = keras.layers.Concatenate()([input_genes, x])

    x = keras.layers.Dense(1024)(concatenate)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(512)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)

    discriminator_model = keras.models.Model([input_genes, input_label], x, name="discriminator")
    discriminator_model.summary()
    return discriminator_model


# Instantiate the discriminator
discriminator = discriminator()


class SCGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dimension, number_of_classes, discriminator_extra_steps,
                 gradient_penalty_weight):
        super(SCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dimension = latent_dimension
        self.number_of_classes = number_of_classes
        self.discriminator_steps = discriminator_extra_steps
        self.gradient_penalty_weight = gradient_penalty_weight

    def compile(self, discriminator_optimizer, generator_optimizer, discriminator_loss_fn, generator_loss_fn):
        super(SCGAN, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_loss_fn = discriminator_loss_fn
        self.generator_loss_fn = generator_loss_fn

    def gradient_penalty(self, batch_size, real_genes, fake_genes, real_genes_label):
        # Get the interpolated genes
        # In tensorflow, the base parameters of random normal are mean=0.0, stddev=1.0
        alpha = tf.math.abs(tf.random.normal([batch_size, 1], 0.0, 1.0))
        difference = fake_genes - real_genes
        interpolated_genes = real_genes + alpha * difference
        with tf.GradientTape() as gp_tape:
            gp_tape.watch([interpolated_genes, real_genes_label])
            predictions = self.discriminator([interpolated_genes, real_genes_label], training=True)

        # Calculate the gradients of the interpolated genes
        gradients = gp_tape.gradient(predictions, [interpolated_genes])[0]
        # Calculate the L2 norm of the gradients
        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        gradient_penalty = tf.reduce_mean((l2_norm - 1.0) ** 2)
        return gradient_penalty

    def train_step(self, input):
        input = list(sum(input, ()))
        real_genes = input[0]
        real_genes_label = input[1]

        batch_size = real_genes.shape[0]
        number_of_classes = self.number_of_classes

        # Train the discriminator
        for i in range(self.discriminator_steps):
            # In tensorflow, the base parameters of random normal are mean=0.0, stddev=1.0
            gaussian_noise = tf.random.normal(shape=(batch_size, self.latent_dimension))
            labels = [random.randint(0, number_of_classes) for p in range(batch_size)]
            labels = tf.convert_to_tensor(labels)
            labels = tf.math.divide(labels, 10, name=None)
            with tf.GradientTape() as tape:
                # Generate fake genes from the noise
                fake_genes = self.generator([gaussian_noise, labels], training=True)
                # Get the predictions for the fake genes
                fake_predictions = self.discriminator([fake_genes, labels], training=True)
                # Get the predictions for the real genes
                real_predictions = self.discriminator([real_genes, real_genes_label], training=True)

                # Calculate the discriminator loss using the fake and real gene predictions
                discriminator_cost = self.discriminator_loss_fn(real_predictions, fake_predictions)
                # Calculate the gradient penalty
                gradient_penalty = self.gradient_penalty(batch_size, real_genes, fake_genes, real_genes_label)
                # Add the gradient penalty to the original discriminator loss
                discriminator_loss = discriminator_cost + gradient_penalty * self.gradient_penalty_weight

            # Get the discriminator gradient
            discriminator_gradient = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

            # Update the weights of the discriminator by applying the gradients of the discriminator
            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # In tensorflow, the base parameters are mean=0.0, stddev=1.0
        gaussian_noise = tf.random.normal(shape=(batch_size, self.latent_dimension))
        labels = [random.randint(0, number_of_classes) for p in range(batch_size)]
        labels = tf.convert_to_tensor(labels)
        labels = tf.math.divide(labels, 10, name=None)
        with tf.GradientTape() as tape:
            # Generate fake genes using the generator
            fake_genes = self.generator([gaussian_noise, labels], training=True)
            # Get the predictions for the fake genes
            fake_predictions = self.discriminator([fake_genes, labels], training=True)
            # Calculate the generator loss
            generator_loss = self.generator_loss_fn(fake_predictions)

        # Get the generator gradient
        generator_gradient = tape.gradient(generator_loss, self.generator.trainable_variables)

        # Update the weights of the generator by applying the gradients of the generator
        self.generator_optimizer.apply_gradients(
            zip(generator_gradient, self.generator.trainable_variables)
        )

        return {"discriminator_loss": discriminator_loss, "generator_loss": generator_loss}


# The learning rate is low and the amsgrad is activated
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9, amsgrad=True)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9, amsgrad=True)


# Define the loss for the discriminator
def discriminator_loss(real_genes, fake_genes):
    real_loss = tf.reduce_mean(real_genes)
    fake_loss = tf.reduce_mean(fake_genes)
    return fake_loss - real_loss


# Define the loss for the generator
def generator_loss(fake_genes):
    return -tf.reduce_mean(fake_genes)


# Instantiate the scgan
scgan = SCGAN(discriminator=discriminator,
              generator=generator,
              latent_dimension=latent_dimension,
              number_of_classes=number_of_classes,
              discriminator_extra_steps=5,
              gradient_penalty_weight=10,
              )

# Compile the scgan
scgan.compile(
    discriminator_optimizer=discriminator_optimizer,
    generator_optimizer=generator_optimizer,
    discriminator_loss_fn=discriminator_loss,
    generator_loss_fn=generator_loss,
)

# Sample a number of rows divisible by the batch size
df_ge_human_reduced = df_ge_human.head(11000)

# Convert df_ge_human_reduced to a tensor
train_genes = tf.convert_to_tensor(np.array(df_ge_human_reduced), dtype="float32")

# Set the parameters
batch_size = 200
epochs = 200

# Create the input
input = [train_genes, human_labels_tensors]

### Please comment out the code if you want to test the gan with conditioning

# Train the scgan
#scgan.fit(input, batch_size = batch_size, epochs = epochs, shuffle = True)

# Save the generator
#generator.save("generator_conditioning_test")
