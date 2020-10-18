# Deep Learning for Capacitated Arc Routing

## Requirements:

* Python 3.6
* pytorch>=1.0
* matplotlib

To install dependencies, navigate to this folder and run ```pip install -r requirements.txt```

# To train a new model

A new model can be trained by running the ```trainer.py``` file. 
Various arguments can be passed for training as needed:

* seed            : Random seed
* checkpoint      : Link to the actor/critic checkpoints
* test            : Flag to set the mode to Testing instead of Training. 
* nodes           : Number of nodes to train for (Can be 10, 20, 50 or 100)
* actor_lr        : Learning rate for the actor
* critic_lr       : Learning rate for the critic
* max_grad_norm   : Value for clipping the gradient
* batch_size      : Training/Testing batch size
* hidden          : Number of hidden layers to be generated
* dropout         : Value for dropout between NN layers
* layers          : Number of layers in the NN
* train_size      : Number of examples to be trained on
* valid_size      : Number of examples to be validated on
* epochs          : Number of epochs to train for

To run the problem with the default set of arguments, run ```python trainer.py```

However, based on the system used, various parameters may have to be changed. 
Try reducing the number of training examples (```train-size```), batch size (```batch_size```), validation size (```valid-size```), and epochs (```epochs```) if the system crashes or if it takes too long to run.

# To test a pre-trained model

In order to get a near-optimal solution, the model will have to be trained for a very long time with a wide range of examples. During training, the model stores checkpoints for each epoch, as well as a checkpoint for the model that gives the best results.

Pre-trained checkpoints have to be stored with the names ```actor.pt``` and ```critic.pt```.

A set of pre-trained checkpoints are provided for testing. This can be found in the ```example_cp``` directory.
To load this model for testing, use the command: ```python tester.py --checkpoint=example_cp```

## Test cases

The model can be trained and tested on various combinations of input arguments. However, each would require re-training the model.

* python trainer.py --train_size=10000 --batch_size=64 --nodes=20
* python trainer.py --nodes=50 --actor_lr=1e-4
* python trainer.py --train_size=10000 --batch_size=32 --nodes=100 --critic_lr=2e-4
* python trainer.py --train_size=10000 --batch_size=32 --nodes=50 --critic_lr=2e-4 --actor_lr=1e-4
* python trainer.py --train_size=10000 --batch_size=32 --nodes=20 --critic_lr=2e-4

# Architecture Diagrams

The domain and class diagrams can be found in the diagrams directory.
