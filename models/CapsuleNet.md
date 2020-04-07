## Capsule Networks

Thanks to  this [!article](https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8)


#### Critical analysis of CNN:

- Inconsistent in recognising and understanding spatial information.

- Pooling operation is generally avoided in recent architectures due to the fact that lots of information is lost because only the most active neurons are chosen to be moved to the next layer.


##### Hinton's approach

``Routing-by-agreement``

Low level features in an image will only get sent to a higher level layer that matches its contents.


### Capsules

- "A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or an object part." 

- These group of neurons encode spatial information as well as the probability of an object being present.

- The length of capsule vector is the probability of feature existing in the image and the direction of the vector would represent its pose information.


```The idea is that computer must be able to render an image inversely; in such a way that its representation signifies the parts of its constituents and its orientation.```


##### But how?

Capsule network learns how to predict an image's constituents along with its orientation by trying to reproduce the object it thinks it detected and comparing it to the labelled example from the training data.


For more: [!Dynamic routing Between Capsules](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf)

and [!Medium Article](https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8)