import tensorflow as _tf
from keras.callbacks import LambdaCallback as _LambdaCallback
import numpy as _np
import matplotlib.pyplot as _plt

"""
Default model setup
"""
def model_structure():
  model = _tf.keras.models.Sequential([
    _tf.keras.layers.Flatten(input_shape=(28, 28)),
    _tf.keras.layers.Dense(128, activation='relu'),
    _tf.keras.layers.Dropout(0.2),
    _tf.keras.layers.Dense(10)
  ])
  return model

"""
Create model weights and save in model.h5
"""
def create_initial_weights():
  model = model_structure()
  model.save_weights('model.h5')

"""
Load model weights saved in model.h5
"""
def load_initial_weights():
  model = model_structure()
  model.load_weights('model.h5')
  return model

"""
Single NN experiment
"""
class NN_experiment:
  def __init__(self, epochs, steps_per_epoch):
    self.epochs = epochs
    self.steps_per_epoch = steps_per_epoch
    self.model = self.make_model()
    self.weights_dict = {}
    self.weights_callback = self.make_callback()
  def make_model(self):
    model = model_structure()
    model.load_weights('model.h5')
    loss_fn = _tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model
  def make_callback(self):
    return _LambdaCallback(on_epoch_end=\
        lambda epoch, logs: \
        self.weights_dict.update({epoch:self.model.get_weights()}))
  def train_model(self,x_train,y_train):
    self.model.fit(
        x_train, 
        y_train, 
        epochs=self.epochs, 
        steps_per_epoch=self.steps_per_epoch,
        callbacks = [self.weights_callback])
  def test_weights_callback(self):
    for key in self.weights_dict:
      print([key, _np.sum(self.weights_dict[key][0].flatten())])
  def get_epoch_weights(self, n):
    return self.weights_dict[n]
  def scatter_weights(self, n_epoch=0):
    weights = self.get_epoch_weights(n_epoch)
    n_layers = len(weights)
    i_start = 0
    i_end = 0
    for i in range(n_layers):
      layer_weights = weights[i]
      layer_size = _np.size(layer_weights)
      i_end += layer_size
      _plt.plot(range(i_start,i_end),layer_weights.flatten(),'.')
      i_start += layer_size
  def hist_NN(self, n_epoch=0):
    weights = self.get_epoch_weights(n_epoch)
    _plt.subplot(121)
    _plt.title('Weights')
    for i in [0,2]:
      layer_weights = weights[i]
      layer_size = _np.size(layer_weights)
      _plt.hist(layer_weights.flatten(), 50, density=True, alpha=0.5)
    _plt.legend(['Layer 1','Layer 2'])
    
    _plt.subplot(122)
    _plt.title('Biases')
    for i in [1,3]:
      layer_weights = weights[i]
      layer_size = _np.size(layer_weights)
      _plt.hist(layer_weights.flatten(), 10, density=True, alpha=0.5)

"""
Series of NN experiments
"""
class NN_series:
  def __init__(self, x_train,y_train, n_experiments, epochs, steps_per_epoch):
    self.n_experiments = n_experiments
    self.epochs = epochs
    self.steps_per_epoch = steps_per_epoch
    self.model_template = lambda: NN_experiment(epochs, steps_per_epoch)
    self.series = self.create_series(x_train,y_train)
  def create_series(self,x_train,y_train):
    series_dict = {}
    for i in range(self.n_experiments):
      print(f'Series {i+1}/{self.n_experiments}')
      nn = self.model_template()
      nn.train_model(x_train,y_train)
      series_dict.update({i: nn})
    return series_dict
  
"""
Transform to matrix of (n_experiments, n_epochs, n_values) shape

Hardcoded to take only weights of first dense layer
"""
def make_mat(x):
  # n_experiments -> nns1.n_experiments
  # n_epochs -> nns1.epochs
  # n_values -> nns1.series[0].model.layers[1].weights[0].shape.num_elements()
  id_layer = 1
  size = (x.n_experiments, 
          x.epochs, 
          x.series[0].model.layers[id_layer].weights[0].shape.num_elements())
  
  x_mat = _np.zeros(size)
  for i, key in enumerate(x.series):
    for j in range(x.epochs):
      x_mat[i,j,:] = x.series[key].weights_dict[j][0].flatten()
  return x_mat

"""
Get matrix of initial weights

Hardcoded to take only weights of first dense layer
"""
def initial_first_weights():
  model = load_initial_weights()
  return model.get_weights()[0].flatten()

"""
Euclidian distances between an array and a matrix

Compares dimentions of (i,j,:) and (j,:)
"""
def euclidian_distance(x, x_mean, n_experiments, n_epochs):
  distances = _np.zeros((n_experiments, n_epochs))
  for i in range(n_experiments):
    for j in range(n_epochs):
      distances[i,j] = _np.linalg.norm(x[i,j,:] - x_mean[j,:])
  return distances
