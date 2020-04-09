### Wasserstein GAN

Aimed to improve training stability of GANs. Model constitutes a critic that explicitly tells the scores of "realness" of a given image.

### How is it different from traditional GAN

The conceptual shift is motivated mathematically using the <strong>earth mover distance</strong>, or <strong>Wasserstein distance</strong>, to train the GAN that measures the distance between the data distribution observed in the training dataset and the distribution observed in the generated examples.


It is an important extension to the GAN model and requires a conceptual shift away from a discriminator that predicts the probability of a generated image being “real” and toward the idea of a critic model that scores the “realness” of a given image.