import numpy as np
from data_preprocessing import df_ge_human
import tensorflow as tf
from tensorflow import keras

#The WGAN gp is inspired from Nain (2020). His model is one of the WGAN gp examples from keras.

# Number of genes
latent_dimension = 9391

# Remove the empty rows
df_ge_human = df_ge_human.iloc[1:, :]
df_ge_human = df_ge_human.iloc[:-1, :]


# Define the generator
def generator():
    input = keras.layers.Input(shape=(latent_dimension,))

    x = keras.layers.Dense(128)(input)
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
    x = keras.layers.BatchNormalization()(x)

    generator_model = keras.models.Model(input, x, name="generator")
    return generator_model


# Instantiate the generator
generator = generator()


# Define the discriminator
def discriminator():
    input = keras.layers.Input(shape=(latent_dimension,))

    x = keras.layers.Dense(1024)(input)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(512)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)

    discriminator_model = keras.models.Model(input, x, name="discriminator")
    return discriminator_model


# Instantiate the discriminator
discriminator = discriminator()


class SCGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dimension, discriminator_extra_steps, gradient_penalty_weight):
        super(SCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dimension = latent_dimension
        self.discriminator_steps = discriminator_extra_steps
        self.gradient_penalty_weight = gradient_penalty_weight

    def compile(self, discriminator_optimizer, generator_optimizer, discriminator_loss_fn, generator_loss_fn):
        super(SCGAN, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_loss_fn = discriminator_loss_fn
        self.generator_loss_fn = generator_loss_fn

    def gradient_penalty(self, batch_size, real_genes, fake_genes):
        # Get the interpolated genes
        # In tensorflow, the base parameters of random normal are mean=0.0, stddev=1.0
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        difference = fake_genes - real_genes
        interpolated_genes = real_genes + alpha * difference

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_genes)
            predictions = self.discriminator(interpolated_genes, training=True)

        # Calculate the gradients of the interpolated genes
        gradients = gp_tape.gradient(predictions, [interpolated_genes])[0]
        # Calculate the L2 norm of the gradients
        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        gradient_penalty = tf.reduce_mean((l2_norm - 1.0) ** 2)
        return gradient_penalty

    def train_step(self, real_genes):
        batch_size = real_genes.shape[0]

        # Train the discriminator
        for i in range(self.discriminator_steps):
            # In tensorflow, the base parameters are mean=0.0, stddev=1.0
            gaussian_noise = tf.random.normal(shape=(batch_size, self.latent_dimension))
            with tf.GradientTape() as tape:
                # Generate fake genes from the noise
                fake_genes = self.generator(gaussian_noise, training=True)
                # Get the predictions for the fake genes
                fake_predictions = self.discriminator(fake_genes, training=True)
                # Get the predictions for the real genes
                real_predictions = self.discriminator(real_genes, training=True)

                # Calculate the discriminator loss using the fake and real gene predictions
                discriminator_cost = self.discriminator_loss_fn(real_genes=real_predictions,
                                                                fake_genes=fake_predictions)
                # Calculate the gradient penalty
                gradient_penalty = self.gradient_penalty(batch_size, real_genes, fake_genes)
                # Add the gradient penalty to the original discriminator loss
                discriminator_loss = discriminator_cost + gradient_penalty * self.gradient_penalty_weight

            # Get the discriminator gradient
            discriminator_gradient = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

            # Update the weights of the discriminator by applying the gradients of the discriminator
            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # In tensorflow, the base parameters of random normal are mean=0.0, stddev=1.0
        gaussian_noise = tf.random.normal(shape=(batch_size, self.latent_dimension))
        with tf.GradientTape() as tape:
            # Generate fake genes using the generator
            fake_genes = self.generator(gaussian_noise, training=True)
            # Get the predictions for the fake genes
            fake_predictions = self.discriminator(fake_genes, training=True)
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
df_ge_human_reduced = df_ge_human.head(11008)

# Convert df_ge_human_reduced to a tensor
train_genes = tf.convert_to_tensor(np.array(df_ge_human_reduced), dtype="float32")

# Set the parameters
batch_size = 128
epochs = 10000

# Train the scgan
scgan.fit(train_genes, batch_size=batch_size, epochs=epochs, shuffle=True)

# Save the generator
generator.save("generator_human_10000_shuffle_128_batch")
