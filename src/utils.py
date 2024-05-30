import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_probability.python.distributions import (
    MultivariateNormalLinearOperator as mvn,
    Categorical as cat,
    OneHotCategorical as cat1h,
    Mixture as mix,
    Normal as nm,
)
import io
import os


def generate_sin(thetas=(0.1, 1), length=100, samples=50, noise=0.01):
    thetas = np.asarray(thetas)
    noise = np.random.normal(size=(len(thetas), samples)) * noise
    x = np.linspace(np.zeros(len(thetas)), np.ones(len(thetas)) * length, length)[
        :, :, None
    ] * (
        thetas[None, :, None] + noise[None, :, :]
    )  #  length x clusters
    return np.sin(x)


def generate_trajectories(
    clusters_logits=(0.5, 0.5),
    clusters_drift=((1.0, 1), (-1, -1)),
    clusters_diffusion=(((0.5, 0.1), (-2.0, -0.9)), ((-0.5, 0.5), (0.5, 0.1))),
    samples=30,
    length=100,
    clusters_beta=((0.5, 0.9), (0.9, 0.6)),
    initial_pos=np.asarray([[2.0, 2.0], [-2.0, -2.0]]),
    dt=1e-1,
):
    if isinstance(clusters_diffusion, list) or isinstance(clusters_diffusion, tuple):
        clusters_diffusion = np.asarray(clusters_diffusion)

    if isinstance(clusters_drift, list) or isinstance(clusters_drift, tuple):
        clusters_drift = np.asarray(clusters_drift)

    if isinstance(clusters_beta, list) or isinstance(clusters_beta, tuple):
        clusters_beta = np.asarray(clusters_beta)

    clusters = len(clusters_logits)
    dim = clusters_drift.shape[-1]
    assert clusters == len(clusters_drift) == len(clusters_diffusion)
    ta = tf.TensorArray(
        element_shape=(samples, clusters, 2), dtype=tf.float64, size=length
    )
    distr = mvn(
        loc=clusters_drift,
        scale=tf.linalg.LinearOperatorLowerTriangular(clusters_diffusion),
    )
    initial_samples = tf.squeeze(distr.sample(samples))
    x = tf.zeros((samples, clusters, dim), dtype=tf.double)
    for t in range(length):
        x_prime = distr.sample(samples)
        x = x + tf.squeeze(x_prime) * clusters_beta[None, ...] * dt
        ta = ta.write(t, x)
    trajectories = ta.stack() + initial_samples
    return tf.cast(trajectories + initial_pos[None, None, :, :], tf.float32)