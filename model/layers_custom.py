# TensorFlow and Keras imports
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import glorot_uniform, Initializer
from tensorflow.keras import activations, initializers, constraints
from tensorflow.keras.regularizers import Regularizer


class Attention(Layer):
    def __init__(self, **kwargs):
        # self.init = initializations.get('glorot_uniform')
        self.init = keras.initializers.get('glorot_uniform')
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # each sample should be a scalar
        assert len(input_shape) == 2
        # self.weights = self.init(input_shape[1:], name='weights')
        weights = self.init(input_shape[1:])
        glorot_uniform()
        # let Keras know that we want to train the multiplicand
        self.trainable_weights = [weights]

    def compute_output_shape(self, input_shape):
        # we're doing a scalar multiply, so we don't change the input shape
        assert input_shape and len(input_shape) == 2
        return input_shape

    def call(self, x, mask=None):
        # this is called during MultiplicationLayer()(input)
        return x * self.weights


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = keras.initializers.get('normal')
        # self.init = initializations.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        weighted_input = weighted_input.sum(axis=1)
        return weighted_input

    def compute_output_shape(self, input_shape):
        print('AttLayer input_shape', input_shape)
        return (input_shape[0], input_shape[-1])
        # return (input_shape[0])


class SwitchLayer(Layer):

    def __init__(self, kernel_regularizer=None, **kwargs):
        # self.output_dim = output_dim
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(SwitchLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],),
                                      initializer='uniform',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(SwitchLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        # return K.dot(x, self.kernel)
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape


# assume the inputs are connected to the layer nodes according to a pattern. The first node is connected to the first n inputs
# the second to the second n inputs and so on.
class Diagonal(Layer):
    def __init__(self, units, activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 W_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        # self.output_dim = output_dim
        # self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.units = units
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.W_regularizer = W_regularizer
        self.bias_regularizer = bias_regularizer
        # Store both names for backwards compatibility
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = bias_constraint
        super(Diagonal, self).__init__(**kwargs)

    # the number of weights, equal the number of inputs to the layer
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dimension = input_shape[1]
        self.kernel_shape = (input_dimension, self.units)
        print('input dimension {} self.units {}'.format(input_dimension, self.units))
        self.n_inputs_per_node = input_dimension // self.units  # Integer division for Python 3
        print('n_inputs_per_node {}'.format(self.n_inputs_per_node))

        rows = np.arange(input_dimension)
        cols = np.arange(self.units)
        cols = np.repeat(cols, self.n_inputs_per_node)
        self.nonzero_ind = np.column_stack((rows, cols))

        # print('self.nonzero_ind', self.nonzero_ind)
        print('self.kernel_initializer', self.W_regularizer, self.kernel_initializer, self.kernel_regularizer)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dimension,),
                                      # initializer='uniform',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True, constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Diagonal, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        n_features = tf.shape(x)[1]
        print('input dimensions {}'.format(tf.shape(x)))

        kernel = K.reshape(self.kernel, (1, n_features))
        mult = x * kernel
        mult = K.reshape(mult, (-1, self.n_inputs_per_node))
        mult = K.sum(mult, axis=1)
        output = K.reshape(mult, (-1, self.units))

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        # config = {
        #         'units': self.units, 'activation':self.activation,
        # 'kernel_shape': self.kernel_shape, 'nonzero_ind':self.nonzero_ind, 'n_inputs_per_node': self.n_inputs_per_node }

        config = {

            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            # 'W_regularizer' : self.W_regularizer,
            # 'bias_regularizer' : self.bias_regularizer,

        }
        # 'kernel_initializer' : self.kernel_initializer,
        # 'bias_initializer' : self.bias_initializer,
        # 'W_regularizer' : ,
        # 'bias_regularizer' : None
        # 'kernel_shape': self.kernel_shape
        # dsve
        base_config = super(Diagonal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# from keras.engine.topology import Layer
import tensorflow as tf


class SparseTF(Layer):
    def __init__(self, units, map=None, nonzero_ind=None, kernel_initializer='glorot_uniform', W_regularizer=None,
                 activation='tanh', use_bias=True,
                 bias_initializer='zeros', bias_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.map = map
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        # Store both for backwards compatibility
        self.W_regularizer = W_regularizer
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation_fn = activations.get(activation)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(SparseTF, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        # random sparse constarints on the weights
        # if self.map is None:
        #     mapp = np.random.rand(input_dim, self.units)
        #     mapp = mapp > 0.9
        #     mapp = mapp.astype(np.float32)
        #     self.map = mapp
        # else:
        if self.map is not None:
            self.map = self.map.astype(np.float32)

        # can be initialized directly from (map) or using a loaded nonzero_ind (useful for cloning models or create from config)
        if self.nonzero_ind is None and self.map is not None:
            nonzero_ind = np.array(np.nonzero(self.map)).T
            self.nonzero_ind = nonzero_ind
        elif self.nonzero_ind is None:
            # If neither map nor nonzero_ind is provided, create a fully connected layer
            rows = np.repeat(np.arange(input_dim), self.units)
            cols = np.tile(np.arange(self.units), input_dim)
            self.nonzero_ind = np.column_stack((rows, cols))

        self.kernel_shape = (input_dim, self.units)
        
        # Get the count of non-zero elements
        nonzero_count = self.nonzero_ind.shape[0]
        print('nonzero_count', nonzero_count)
        
        # Create the kernel weights (only for non-zero elements)
        self.kernel_vector = self.add_weight(
            name='kernel_vector',
            shape=(nonzero_count,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )
        # self.kernel = tf.scatter_nd(self.nonzero_ind, self.kernel_vector, self.kernel_shape, name='kernel')
        # --------
        # init = np.random.rand(input_shape[1], self.units).astype( np.float32)
        # sA = sparse.csr_matrix(init)
        # self.kernel = K.variable(sA, dtype=K.floatx(), name= 'kernel',)
        # self.kernel_vector = K.variable(init, dtype=K.floatx(), name= 'kernel',)

        # print(self.kernel.values)
        # ind = np.array(np.nonzero(init))
        # stf = tf.SparseTensor(ind.T, sA.data, sA.shape)
        # print(stf.dtype)
        # print(init.shape)
        # # self.kernel = stf
        # self.kernel = tf.keras.backend.variable(stf, dtype='SparseTensor', name='kernel')
        # print(self.kernel.values)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(SparseTF, self).build(input_shape)  # Be sure to call this at the end
        # self.trainable_weights = [self.kernel_vector]

    def call(self, inputs):
        # Create a dense weight matrix from the sparse representation
        kernel_dense = tf.scatter_nd(
            indices=self.nonzero_ind,
            updates=self.kernel_vector,
            shape=self.kernel_shape
        )
        
        # Apply the weights to the inputs
        output = K.dot(inputs, kernel_dense)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            # 'kernel_shape': self.kernel_shape,
            'use_bias': self.use_bias,
            'nonzero_ind': np.array(self.nonzero_ind),
            # 'kernel_initializer': initializers.serialize(self.kernel_initializer),
            # 'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),

            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'W_regularizer': regularizers.serialize(self.kernel_regularizer),

        }
        base_config = super(SparseTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def call(self, inputs):
    #     print self.kernel.shape, inputs.shape
    #     tt= tf.sparse.transpose(self.kernel)
    #     output = tf.sparse.matmul(tt, tf.transpose(inputs ))
    #     return tf.transpose(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    # def get_weights(self):
    #
    #     return [self.kernel_vector, self.bias]


class SpraseLayerTF(Layer):
    def __init__(self, mapp, activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 W_regularizer=None,
                 bias_regularizer=None,

                 **kwargs):
        # self.output_dim = output_dim
        # self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.map = mapp
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        # Store both for backwards compatibility
        self.W_regularizer = W_regularizer
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        super(SpraseLayerTF, self).__init__(**kwargs)

    # the number of weights, equal the number of inputs to the layer
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dimension = input_shape[1]
        print('input dimension {}'.format(input_dimension))
        self.n_inputs_per_node = input_dimension // self.units  # Integer division for Python 3
        print('n_inputs_per_node {}'.format(self.n_inputs_per_node))

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dimension,),
                                      # initializer='uniform',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
            # constraint=self.bias_constraint)
        else:
            self.bias = None

        super(SpraseLayerTF, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):

        n_features = x._keras_shape[1]

        print('input dimensions {}'.format(x._keras_shape))
        kernel = K.reshape(self.kernel, (1, n_features))

        mult = x * kernel

        mult = K.reshape(mult, (-1, self.n_inputs_per_node))
        mult = K.sum(mult, axis=1)
        output = K.reshape(mult, (-1, self.units))

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = {
            'units': self.units, 'activation': self.activation}
        base_config = super(SpraseLayerTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpraseLayerWithConnection(Layer):

    def __init__(self, mapp, activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 W_regularizer=None,
                 bias_regularizer=None,

                 **kwargs):
        # self.output_dim = output_dim
        # self.kernel_regularizer = regularizers.get(kernel_regularizer)
        n_inputs, n_outputs = mapp.shape
        self.mapp = mapp
        self.units = n_outputs
        super(SpraseLayerWithConnection, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        # Store both for backwards compatibility
        self.W_regularizer = W_regularizer
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

    # the number of weights, equal the number of inputs to the layer
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dimension = input_shape[1]
        print('input dimension {}'.format(input_dimension))

        # self.n_inputs_per_node = input_dimension // self.units
        # print('n_inputs_per_node {}'.format(self.n_inputs_per_node))

        self.edges = []
        # W = []
        self.kernel = []
        for col in self.mapp.T:
            connections = np.nonzero(col)
            # print('connections', type(connections), connections)
            self.edges.append(list(connections[0]))
            n_conn = connections[0].shape[0]
            # print('n_conn', n_conn)

            w = self.add_weight(name='kernel',
                                shape=(n_conn,),
                                # shape=(input_dimension,),
                                # initializer='uniform',
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                trainable=True)
            self.kernel.append(w)
            #     print conn
            # print sum(col)

        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_dimension,),
        #                               # initializer='uniform',
        #                               initializer=self.kernel_initializer,
        #                               regularizer=self.kernel_regularizer,
        #                               trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
            # constraint=self.bias_constraint)
        else:
            self.bias = None

        super(SpraseLayerWithConnection, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        n_inputs, n_outputs = self.mapp.shape
        print(K.int_shape(x))
        output_list = []
        for i in range(n_outputs):
            # print(self.edges[i])
            # print(K.int_shape(x), K.int_shape(self.kernel[i]))
            # y0 = x[:, self.edges[i]].dot(self.kernel[i].T)
            print('iter {}, weights shape {}, # connections {}'.format(i, K.int_shape(self.kernel[i]),
                                                                       len(self.edges[i])))
            print('connections', self.edges[i])
            w = self.kernel[i]
            inn = tf.gather(x, self.edges[i], axis=1)
            y0 = K.dot(inn, w)
            # print(K.int_shape(y0))
            if self.use_bias:
                y0 = K.bias_add(y0, self.bias[i:i+1])
            if self.activation is not None:
                y0 = self.activation(y0)

            # print(K.int_shape(y0))
            output_list.append(y0)
        # y = [x[:, self.edges[i]].dot(W[i].T) for i in range(n_outputs)]

        # n_features = tf.shape(x)[1]
        #
        # print('input dimensions {}'.format(tf.shape(x)))
        # kernel = K.reshape(self.kernel, (1, n_features))
        #
        # mult = x * kernel
        #
        # mult = K.reshape(mult, (-1, self.n_inputs_per_node))
        # mult = K.sum(mult, axis=1)
        # output = K.reshape(mult, (-1, self.units))

        # if self.use_bias:
        #     output = K.bias_add(output, self.bias)
        # if self.activation is not None:
        #     output = self.activation(output)
        print('concatenating')
        output = K.concatenate(output_list, axis=-1)
        output = K.reshape(output, (-1, self.units))
        # output = concatenate(output)
        # print(K.int_shape(output))
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


from scipy.sparse import csr_matrix
from tensorflow.keras import backend as K
import warnings


class RandomWithMap(Initializer):
    """Initializer that generates tensors initialized to random array.
    """

    def __init__(self, mapp):
        self.map = mapp

    def __call__(self, shape, dtype=None):
        map_sparse = csr_matrix(self.map)
        # init = np.random.rand(*map_sparse.data.shape)
        init = np.random.normal(10.0, 1., *map_sparse.data.shape)
        print('connection map data shape {}'.format(map_sparse.data.shape))
        # init = np.random.randn(*map_sparse.data.shape).astype(np.float32) * np.sqrt(2.0 / (map_sparse.data.shape[0]))
        initializers.glorot_uniform().__call__(shape)
        map_sparse.data = init
        return K.variable(map_sparse.toarray())


class L1L2_with_map(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, mapp, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.connection_map = mapp

    def __call__(self, x):
        x_masked = x * self.connection_map.astype(K.floatx())
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x_masked))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x_masked))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


# taken from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Import TensorFlow 2.x compatible versions of layers
try:
    from layers_custom_tf2 import Diagonal as DiagonalTF2
    from layers_custom_tf2 import SparseTF as SparseTFTF2
    from layers_custom_tf2 import SparseTFConstraint
    
    # For backward compatibility, provide aliases to TF2 versions
    # This allows existing code to use these classes without changes
    # while new code can import the TF2 versions directly
    DiagonalTF2Layer = DiagonalTF2
    SparseTFTF2Layer = SparseTFTF2
    
    warnings.warn(
        "TensorFlow 2.x compatible versions of Diagonal and SparseTF are available. "
        "The original implementations are deprecated and will be removed in a future version. "
        "Use DiagonalTF2 and SparseTFTF2 instead.",
        DeprecationWarning
    )
except ImportError:
    # If TF2 versions are not available, keep using the old versions
    warnings.warn(
        "TensorFlow 2.x compatible versions of Diagonal and SparseTF could not be imported. "
        "Using original implementations which may not be fully compatible with TensorFlow 2.x.",
        RuntimeWarning
    )
