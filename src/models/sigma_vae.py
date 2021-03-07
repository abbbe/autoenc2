# https://github.com/orybkin/sigma-vae-tensorflow

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from src.models.vae import build_encoder, build_decoder

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        def gaussian_nll(mu, log_sigma, x):
            return 0.5 * ((x - mu) / tf.math.exp(log_sigma)) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            log_sigma = tf.math.log(tf.math.sqrt(tf.reduce_mean(
                (data - reconstruction) ** 2, [0, 1, 2, 3], keepdims=True)))
            re_loss = tf.reduce_sum(gaussian_nll(
                reconstruction, log_sigma, data))
            kl_loss = - \
                tf.reduce_sum(0.5 * (1 + z_log_var - z_mean **
                                     2 - tf.math.exp(z_log_var)))

            total_loss = re_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(re_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "re_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def build_autoencoder(inputShape, latentShape, build_params):
    inputs = layers.Input(shape=inputShape)
    encoder_model, volumeSize = build_encoder(
        inputs, latentShape, build_params)

    latent = layers.Input(shape=latentShape)
    decoder_model = build_decoder(latent, volumeSize, build_params)

    autoencoder_model = VAE(encoder_model, decoder_model, name="autoencoder")
    autoencoder_model.compile(optimizer=keras.optimizers.Adam())

    return {'ae': autoencoder_model, 'enc': encoder_model, 'dec': decoder_model}
