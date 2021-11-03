import tensorflow as tf
import tensorflow_probability as tfp

if __name__ == '__main__':
    tfd = tfp.distributions
    initial_distribution = tfp.distributions.Categorical(probs=[0.8, 0.2])
    transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                     [0.2, 0.8]])
    observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

    model = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=7 # how many days do we want
    )

    mean = model.mean()

    with tf.compat.v1.Session() as sess:
        print(mean.numpy())