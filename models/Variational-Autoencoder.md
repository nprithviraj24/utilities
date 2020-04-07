### Variational Autoencoder

VAE's are generative models, in contrast to regressors or classifiers.


#### Encoder

The encoder is a network that takes in an input (generally an image) and produces a much smaller representation (the encoding), that contains enough information for the next part of the network to process it into the desired output format.

Typically, the encoder is trained together with the other parts of the network, optimized via back-propagation, to produce encodings specifically useful for the task at hand.

#### Decoder

As you can make out from previous section, decoder is a network that decodes an encoding into our desired form.


<strong>Combination of Encoder and Decoder is Autoencoders.</strong>

###### Drawback of Autoencoders


The fundamental problem with autoencoders, for generation, is that the latent space they convert their inputs to and where their encoded vectors lie, may not be continuous, or allow easy interpolation.



<strong>Variational Autoencoders (VAEs)</strong> have one fundamentally unique property that separates them from vanilla autoencoders, and it is this property that makes them so useful for generative modeling: their latent spaces are, by design, continuous, allowing easy random sampling and interpolation


##### How are they different from Autoencoders?

We are never really sure if the encoder will map to a continuos latent space from which we can interpolate. 


##### What does VAE brings to the table?

VAE are novel AE because it ensures that latent space is continuos, i.e. we can sample any random value from latent space and it will generate a distinct palatable image for distinct value in this latent space.


```Encoder in VAE have two outputs, which are combined to form a single input to decoder.```

For more: Read Tutorial on VAE's
