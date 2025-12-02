'''
This module implements the original PointNet algorithm presented in

"PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017)"

and follows the author's original build at  https://github.com/charlesq34/pointnet/blob/master/models/pointnet_seg.py

----------------------------------------------------------------------------------------------------------
                                        THE ORIGINAL POINTNET
----------------------------------------------------------------------------------------------------------

LAYER TYPES

1. Convolutional Layers
    a. Hyperparameters
        i.      filters     - (int) dimension of the output space (number of resulting feature maps) - varies by layer (64, 128, 1024)
        ii.     kernel_size - ((int, int)) kernel size - (1, 1) everywhere
                NOTE:   The kernel_size[1] dimension must be 1 since we do not have a 2D image.
                        During convolution, the author convolves over an image of size n x 1 and treats the (x, y, z) components
                        as channels, similar to a 3-channel image (RGB).
        iii.    padding     - (str) 'same' or 'valid' - 'valid' used everywhere
        iv.     activation  - (str) see Keras documentation for options - ReLU used everywhere
        v.      apply_bn    - (bool) whether to apply batch normalization to the layer - True for all layers
        vi.     bn_momentum - (float) momentum applied to batch normalization - True for all layers; default decay is 0.99

2. Dense Layers
    a. Hyperparameters
        i.      units       - (int) dimensionality of output (number of nodes)
        ii.     activation  - (str) see Keras doumentation for options - ReLU used everywhere except output layer (None)
        iii.    apply_bn    - (bool) whether to apply batch normalization to the layer - True for all layers
        iv.     bn_momentum - (float) momentum applied to batch normalization - can't tell; default is 0.99

3. T-Net (Transformation Mini-Network)
    a. PURPOSE: The T-Net predicts the affine transformation of the point cloud, then uses that transformation matrix to perform
                an affine transformation on the input. This means that the T-Net is a parallel structure to the primary network.
    b. Structure
        i.      Copy the input      -> the copy will be matrix-multiplied by the transformation matrix at the end
        ii.     Expand dimensions   -> the dimensions will expand from b x n x 3 -> b x n x 1 x 3 for convolution
        iii.    Expanding CNN       -> 3 layers (64 -> 128 -> 1024), no batch normalization, no activation
        iv.     Squeeze dimensions  -> reduce the dimensions back to b x n x 1024 (removed axis 2)
        v.      Global max pool     -> pools over the n-dimension to produce a b x 1024 matrix
        vi.     Reducing dense      -> 2 layers (512 -> 256), with batch normalization and ReLU normalization
        vii.    Expand dimensions   -> expand one more time from b x 256 -> b x 1 x 256
        viii.   Create k x k matrix -> matrix multiply with the weight matrix to produce b x 1 x k^2
        ix.     Squeeze dimensions  -> reduce to b x k^2
        x.      Reshape to k x k    -> reduce again to k x k
        xi.     Add in bias         -> initialized to identity, eventually trained
        xii.    Regularize          -> only performed on the second T-Net, not the first
        xiii.   Merge with net      -> matrix multiply with the input and output the result to the following layer
        
COMPLETE NETWORK

1. Input layer      -> n x 3 (n = number of point in cloud)
2. T-Net            -> with bn_momentum
                    NOTE:  apply_bn is set to false on all convolutional layers in the T-Net, so this appears to only apply to the dense layers
3. Expand           -> similar to the T-Net, expand to b x n x 1 x 3 for convolution kernel
4. Convolution      -> (64, 64) with ReLU and batch normalization
5. Squeeze          -> reduce back to b x n x 3
6. T-Net            -> with bn_momentum and regularization
7. Expand           -> similar to the T-Net, expand to b x n x 1 x 64 for convolution kernel
8. Expanding Conv   -> (64, 128, 1024) with ReLU and batch normalization
9. Squeeze          -> reduce back ot b x n x 1024
10. Global max pool -> reduce dimensionality to b x 1024
11. Dense w/ D/O    -> two hidden dense layers (512, 256) with ReLU nd batch normalization, appended with a dropout layer (rate of 0.3)
12. Output layer    -> output layer (40 wide for Model40 example), no activation
13. During inference, the output layer is activated by a sigmoid function

--------------------

PointNet.py

By:     Mike Pieschl
Date:   30 July 2025
'''

import tensorflow as tf
import numpy as np
import plotly.graph_objects as go

from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Dropout
from tensorflow.keras import Model, saving, initializers

@saving.register_keras_serializable(package="Project")
class PointNet(Model):
    def __init__(self, classification_output_width: int, segmentation_output_width: int, dropout_rate: float, random_seed: int, debugging: bool = False, **kwargs):
        '''
        Implements https://github.com/luis-gonzales/pointnet_own/blob/master/src/model.py get_model

        This class implements the full PointNet model.

        @params output_width    (int) number of output classes
        '''

        super(PointNet, self).__init__(**kwargs)
        self._classification_output_width = classification_output_width
        self._segmentation_output_width = segmentation_output_width
        self._random_seed = random_seed
        self._debugging = debugging
        self._layers = []
        self.input_names = ['pointnet_seg_input']
        self.output_names = ['classification_output', 'segmentation_output', 'rotation_matrix']

        self.input_transform = TNet(name = 'input_transform', add_regularization = True, random_seed = self._random_seed)

        self.mlp_1_1 = ConvLayer(filters = 64, name = 's1_l1_64', activation = tf.nn.relu, apply_bn = True, random_seed = self._random_seed)
        self.mlp_1_2 = ConvLayer(filters = 64, name = 's1_l2_64', activation = tf.nn.relu, apply_bn = True, random_seed = self._random_seed)

        self.feature_transform = TNet(name = 'feature_transform', add_regularization = False, random_seed = self._random_seed)

        self.mlp_2_1 = ConvLayer(filters = 64, name = 's2_l1_64', activation = tf.nn.relu, apply_bn = True, random_seed = self._random_seed)
        self.mlp_2_2 = ConvLayer(filters = 128, name = 's2_l2_128', activation = tf.nn.relu, apply_bn = True, random_seed = self._random_seed)
        self.mlp_2_3 = ConvLayer(filters = 1024, name = 's2_l3_1024', activation = tf.nn.relu, apply_bn = True, random_seed = self._random_seed)

        # Classification Head
        self.mlp_cls_1 = DenseLayer(units = 512, name = 's3_l1_512', activation = tf.nn.relu, apply_bn = True)
        self.dropout_1 = Dropout(rate = dropout_rate)
        self.mlp_cls_2 = DenseLayer(units = 256, name = 's3_l2_256', activation = tf.nn.relu, apply_bn = True)
        self.dropout_2 = Dropout(rate = dropout_rate)
        self.mlp_cls_3 = DenseLayer(units = self._classification_output_width, name = 'output', activation = tf.nn.softmax)

        # Segmentation Head
        self.mlp_seg_1 = ConvLayer(filters = 512, name = 'seg_l1_512', activation = tf.nn.relu, apply_bn = True, random_seed = self._random_seed)
        self.mlp_seg_2 = ConvLayer(filters = 256, name = 'seg_l2_256', activation = tf.nn.relu, apply_bn = True, random_seed = self._random_seed)
        self.mlp_seg_3 = ConvLayer(filters = 128, name = 'seg_l3_128', activation = tf.nn.relu, apply_bn = True, random_seed = self._random_seed)
        self.mlp_seg_4 = ConvLayer(filters = 128, name = 'seg_l4_128', activation = tf.nn.relu, apply_bn = True, random_seed = self._random_seed)
        self.mlp_seg_5 = ConvLayer(filters = self._segmentation_output_width, name = 'seg_l5_output', activation = None, apply_bn = False, random_seed = self._random_seed)

        # Save layers in iterable format
        self.layers.append(self.input_transform)
        self.layers.append(self.mlp_1_1)
        self.layers.append(self.mlp_1_2)
        self.layers.append(self.feature_transform)
        self.layers.append(self.mlp_2_1)
        self.layers.append(self.mlp_2_2)
        self.layers.append(self.mlp_2_3)
        self.layers.append(self.mlp_cls_1)
        self.layers.append(self.mlp_cls_2)
        self.layers.append(self.mlp_cls_3)
        self.layers.append(self.mlp_seg_1)
        self.layers.append(self.mlp_seg_2)
        self.layers.append(self.mlp_seg_3)
        self.layers.append(self.mlp_seg_4)
        self.layers.append(self.mlp_seg_5)
        
    def build(self, input_shape):
        '''
        Build the model by calling build on all sub-layers
        
        @param input_shape: Expected input shape (b, n, 3)
        '''

        super(PointNet, self).build(input_shape)
        
        self.input_transform.build(input_shape)

        self.mlp_1_1.build((input_shape[0], input_shape[1], 1, input_shape[2]))
        self.mlp_1_2.build((input_shape[0], input_shape[1], 1, 64))
        
        # Feature transform expects (batch_size, n_points, 64)
        self.feature_transform.build((input_shape[0], input_shape[1], 64))
        
        # Continue building remaining layers
        self.mlp_2_1.build((input_shape[0], input_shape[1], 1, 64))
        self.mlp_2_2.build((input_shape[0], input_shape[1], 1, 64))
        self.mlp_2_3.build((input_shape[0], input_shape[1], 1, 128))

        # Classification network
        self.mlp_cls_1.build((input_shape[0], 1024))
        self.dropout_1.build((input_shape[0], 1024))
        self.mlp_cls_2.build((input_shape[0], 512))
        self.dropout_2.build((input_shape[0], 512))
        self.mlp_cls_3.build((input_shape[0], 256))
        
        # Segmentation network
        self.mlp_seg_1.build((input_shape[0], input_shape[1], 1, 1088))
        self.mlp_seg_2.build((input_shape[0], input_shape[1], 1, 512))
        self.mlp_seg_3.build((input_shape[0], input_shape[1], 1, 256))
        self.mlp_seg_4.build((input_shape[0], input_shape[1], 1, 128))
        self.mlp_seg_5.build((input_shape[0], input_shape[1], 1, 128))

    def call(self, pc, training = False):

        pc = pc if not self._debugging else tf.debugging.check_numerics( pc, 'Input point cloud contains nan values' )

        # Input Transform
        R = self.input_transform(pc, training = training)        # (b x 3 x 3)
        X = tf.matmul(pc, R)                                     # (b x n x 3)
        X = X if not self._debugging else tf.debugging.check_numerics( X, 'Input transform produced nan values.' )
        
        # MLP (64, 64)
        X = tf.expand_dims(X, axis = 2)                             # (b x n x 1 x 3)

        X = self.mlp_1_1(X, training = training)                    # (b x n x 1 x 64)
        X = X if not self._debugging else tf.debugging.check_numerics( X, 'mlp_1_1 produced nan values.' )

        X = self.mlp_1_2(X, training = training)                    # (b x n x 1 x 64)
        X = X if not self._debugging else tf.debugging.check_numerics( X, 'mlp_1_2 produced nan values.' )

        X = tf.squeeze(X, axis = 2)                                 # (b x n x 64)

        # Feature Transform
        R = self.feature_transform(X, training = training)          # (b x 64 x 64)
        X_64 = tf.matmul(X, R)                                      # (b x n x 64)

        # MLP (64, 128, 1024)
        X = tf.expand_dims(X_64, axis = 2)                          # (b x n x 1 x 64)

        X = self.mlp_2_1(X, training = training)                    # (b x n x 1 x 64)
        X = X if not self._debugging else tf.debugging.check_numerics( X, 'mlp_2_1 produced nan values.' )

        X = self.mlp_2_2(X, training = training)                    # (b x n x 1 x 128)
        X = X if not self._debugging else tf.debugging.check_numerics( X, 'mlp_2_2 produced nan values.' )

        X = self.mlp_2_3(X, training = training)                    # (b x n x 1 x 1024)
        X = X if not self._debugging else tf.debugging.check_numerics( X, 'mlp_2_3 produced nan values.' )

        X = tf.squeeze(X, axis = 2)                                 # (b x n x 1024)

        # Max pooling
        global_features = tf.reduce_max(X, axis = 1)                # (b x 1024)

        ### CLASSIFICATION HEAD ###

        X_cls = self.mlp_cls_1(global_features, training = training)
        X_cls = X_cls if not self._debugging else tf.debugging.check_numerics( X_cls, 'mlp_2_1 produced nan values.' )

        X_cls = self.dropout_1(X_cls, training = training)

        X_cls = self.mlp_cls_2(X_cls, training = training)
        X_cls = X_cls if not self._debugging else tf.debugging.check_numerics( X_cls, 'mlp_2_1 produced nan values.' )

        X_cls = self.dropout_2(X_cls, training = training)

        X_cls = self.mlp_cls_3(X_cls, training = training)
        X_cls = X_cls if not self._debugging else tf.debugging.check_numerics( X_cls, 'mlp_2_1 produced nan values.' )

        ### SEGMENTATION HEAD ###

        # Concatenate local feature set and global feature set
        X_seg = tf.expand_dims(global_features, axis = 1)
        X_seg = tf.tile(X_seg, [1, pc.shape[1], 1])
        X_seg = tf.concat([X_64, X_seg], axis = -1)

        # Segmentation MLP (512, 256, 128, 128, output_width)
        X_seg = tf.expand_dims(X_seg, axis = 2)
        
        X_seg = self.mlp_seg_1(X_seg, training = training)
        X_seg = X_seg if not self._debugging else tf.debugging.check_numerics( X_seg, 'mlp_3_1 produced nan values.' )

        X_seg = self.mlp_seg_2(X_seg, training = training)
        X_seg = X_seg if not self._debugging else tf.debugging.check_numerics( X_seg, 'mlp_3_2 produced nan values.' )

        X_seg = self.mlp_seg_3(X_seg, training = training)
        X_seg = X_seg if not self._debugging else tf.debugging.check_numerics( X_seg, 'mlp_3_3 produced nan values.' )

        X_seg = self.mlp_seg_4(X_seg, training = training)
        X_seg = X_seg if not self._debugging else tf.debugging.check_numerics( X_seg, 'mlp_3_4 produced nan values.' )

        X_seg = self.mlp_seg_5(X_seg, training = training)
        X_seg = X_seg if not self._debugging else tf.debugging.check_numerics( X_seg, 'mlp_3_5 produced nan values.' )

        X_seg = tf.squeeze(X_seg, axis = 2)

        return [ X_cls, X_seg, R ]
    
    def freeze_input_transform(self) -> None:
        self.input_transform.freeze()

    def thaw_input_transform(self) -> None:
        self.input_transform.thaw()

    def freeze_shared_network(self) -> None:
        self.input_transform.freeze()
        self.mlp_1_1.freeze()
        self.mlp_1_2.freeze()
        self.feature_transform.freeze()
        self.mlp_2_1.freeze()
        self.mlp_2_2.freeze()
        self.mlp_2_3.freeze()

    def thaw_shared_network(self) -> None:
        self.input_transform.thaw()
        self.mlp_1_1.thaw()
        self.mlp_1_2.thaw()
        self.feature_transform.thaw()
        self.mlp_2_1.thaw()
        self.mlp_2_2.thaw()
        self.mlp_2_3.thaw()

    def freeze_segmentation_head(self) -> None:
        self.mlp_seg_1.freeze()
        self.mlp_seg_2.freeze()
        self.mlp_seg_3.freeze()
        self.mlp_seg_4.freeze()
        self.mlp_seg_5.freeze()

    def thaw_segmentation_head(self) -> None:
        self.mlp_seg_1.thaw()
        self.mlp_seg_2.thaw()
        self.mlp_seg_3.thaw()
        self.mlp_seg_4.thaw()
        self.mlp_seg_5.thaw()

    def freeze_classification_head(self) -> None:
        self.mlp_seg_1.freeze()
        self.mlp_seg_2.freeze()
        self.mlp_seg_3.freeze()
        self.mlp_seg_4.freeze()
        self.mlp_seg_5.freeze()

    def thaw_classification_head(self) -> None:
        self.mlp_seg_1.thaw()
        self.mlp_seg_2.thaw()
        self.mlp_seg_3.thaw()
        self.mlp_seg_4.thaw()
        self.mlp_seg_5.thaw()

    def get_layer_trainability(self) -> dict:
        trainability_dict = {}
        for layer_ in self._layers:
            trainability_dict[layer_.name] = layer_.is_trainable()

        return trainability_dict
    
    def get_config(self):
        """Returns the non-layer configuration of the model."""
        config = super(PointNet, self).get_config()
        config.update({
            'classification_output_width': self._classification_output_width,
            'segmentation_output_width': self._segmentation_output_width,
            'random_seed': self._random_seed,
            'debugging': self._debugging
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a model from its config.
        
        This is the inverse of get_config(). It's required for 
        tf.keras.models.load_model() to work correctly.
        """
        return cls(**config)
    
    def get_last_predicted_dcm(self):
        return self.input_transform.get_last_predicted_transformation()

@saving.register_keras_serializable(package="Project")
class TNet(Model):
    def __init__(self, 
                 name: str, 
                 add_regularization: bool = False,
                 bn_momentum: float = 0.99,
                 layer_widths: list = [64, 128, 1024, 512, 256], 
                 random_seed = None,
                 **kwargs):
        '''
        Implements https://github.com/luis-gonzales/pointnet_own/blob/master/src/model.py TNet class
        
        When applied to a (b x n x 3) input tensor, this network estimates an SO3 matrix to make the PointNet
        transformation invariant.

        @param name                 (str) network name used to label layers; ex. 'input_transform' results in 'input_transform_conv_layer_1', etc.
        @param add_regularization   (bool) apply an orthogonalizing regularization, which punishes the network for a non-orthogonality
        @param bn_momentum          (float) add momentum to the batch normalization - bn is only applied to the dense layers
        ''' 
        super(TNet, self).__init__(**kwargs)

        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.seed = random_seed
        self.name = name

        # Layer Construction
        self.conv_layer_1 = ConvLayer(filters = layer_widths[0], activation = tf.nn.relu, kernel_size = (1, 1), strides = (1, 1), bn_momentum = bn_momentum, name = f"{name}_convolution_layer_1", random_seed = self.seed)
        self.conv_layer_2 = ConvLayer(filters = layer_widths[1], activation = tf.nn.relu, kernel_size = (1, 1), strides = (1, 1), bn_momentum = bn_momentum, name = f"{name}_convolution_layer_2", random_seed = self.seed)
        self.conv_layer_3 = ConvLayer(filters = layer_widths[2], activation = tf.nn.relu, kernel_size = (1, 1), strides = (1, 1), bn_momentum = bn_momentum, name = f"{name}_convolution_layer_3", random_seed = self.seed)
        self.dense_layer_1 = DenseLayer(units = layer_widths[3], activation = tf.nn.relu, apply_bn = True, bn_momentum = bn_momentum, name = f"{name}_dense_layer_1", random_seed = self.seed)
        self.dense_layer_2 = DenseLayer(units = layer_widths[4], activation = tf.nn.relu, apply_bn = True, bn_momentum = bn_momentum, name = f"{name}_dense_layer_2", random_seed = self.seed)

    def build(self, input_shape: tuple):
        super(TNet, self).build(input_shape)
        self.K = input_shape[-1]
        self.w = self.add_weight(shape = (256, self.K ** 2), initializer = tf.zeros_initializer, trainable = True, name = 'w')
        self.b = self.add_weight(shape = (self.K, self.K), initializer = 'identity', trainable = True, name = 'b')

    def call(self, X, training = False):
        input_X = X                                     # (b x n x 3)

        # Embed higher dimensions
        X = tf.expand_dims(input_X, axis = 2)           # (b x n x 1 x 3)
        X = self.conv_layer_1(X, training = training)                        # (b x n x 1 x 64)
        X = self.conv_layer_2(X, training = training)                        # (b x n x 1 x 128)
        X = self.conv_layer_3(X, training = training)                        # (b x n x 1 x 1024)
        X = tf.squeeze(X, axis = 2)                     # (b x n x 1024)

        # Global feature reduction
        X = tf.reduce_max(X, axis = 1)                  # (b x 1024)

        # Dense layers
        X = self.dense_layer_1(X, training = training)                       # (b x 512)
        X = self.dense_layer_2(X, training = training)                       # (b x 256)

        # Convert to K x K rotation matrix
        X = tf.expand_dims(X, axis = 1)                 # (b x 1 x 256)
        X = tf.matmul(X, self.w)                        # (b x 1 x K^2)
        X = tf.squeeze(X, axis = 1)                     # (b x K^2)
        X = tf.reshape(X, (-1, self.K, self.K))         # (b x K x K)

        # Add bias term, random_seed = self.seed
        X += self.b                                     # (b x K x K)

        # Store for recall
        self._last_predicted = X

        if(self.add_regularization):
            I = tf.constant(np.eye(self.K), dtype = tf.float32)
            X_XT = tf.matmul(X, tf.transpose(X, perm = [0, 2, 1]))  # X @ X.T = 1 for orthogonal matrix (perm for batch dimension)
            reg_loss = tf.nn.l2_loss(I - X_XT)                      # ...and punish otherwise
            self.add_loss(1e-3 * reg_loss)

        # return tf.matmul(input_X, X)                    # Apply rotation matrix to input
        return X
    
    def get_config(self):
        config = super(TNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization,
            'bn_momentum': self.bn_momentum,
            'seed': self.seed
        })

    def freeze(self):
        self.conv_layer_1.freeze()
        self.conv_layer_2.freeze()
        self.conv_layer_3.freeze()
        self.dense_layer_1.freeze()
        self.dense_layer_2.freeze()

    def thaw(self):
        self.conv_layer_1.thaw()
        self.conv_layer_2.thaw()
        self.conv_layer_3.thaw()
        self.dense_layer_1.thaw()
        self.dense_layer_2.thaw()

    def is_trainable(self):
        return  self.conv_layer_1.is_trainable() and \
                self.conv_layer_2.is_trainable() and \
                self.conv_layer_3.is_trainable() and \
                self.dense_layer_1.is_trainable() and \
                self.dense_layer_2.is_trainable()

@saving.register_keras_serializable(package="Project")
class ConvLayer(Layer):
    def __init__(self,
                 filters: int,
                 name: str,
                 kernel_size: tuple = (1, 1),
                 strides: tuple = (1, 1),
                 padding: str = 'same',
                 activation = None,
                 apply_bn: bool = True,
                 bn_momentum: float = 0.99,
                 random_seed = None,
                 **kwargs):
        '''
        Implements https://github.com/luis-gonzales/pointnet_own/blob/master/src/model.py CustomConv class

        The class defines a convolutional layer, combined with user selectable batch normalization and activation

        @param filters      (int) width of the layer / number of output feature maps
        @param kernel_size  (tuple) size of convolutional kernel - must be (x, 1) due to input size
        @param strides      (tuple) kernel stride
        @param name         (str) name appended to layer name - i.e. 'mlp_64' results in 'mlp_64_convolution_layer'
        @param padding      (str) 'valid' - decreases size of feature map - or 'same' - zero pads to maintain size of feature map
        @param activation   (tf.nn.) activation function
        @param apply_bn     (bool) apply batch normalization to layer
        @param bn_momentum  (float) if batch normalization is applied, apply this momentum
        '''
        
        super(ConvLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.bn = BatchNormalization(momentum = bn_momentum) if apply_bn else None
        self.name = f"{name}_convolution_layer"
        self.seed = random_seed

        self.initializer = initializers.GlorotUniform( seed = self.seed ) if type( self.seed ) == int else initializers.GlorotUniform()

        # The activation function is added after a batch normalization layer
        self.conv = Conv2D(filters = filters, 
                           kernel_size = kernel_size, 
                           strides = strides, 
                           padding = padding,
                           activation = None, 
                           use_bias = not apply_bn,
                           name = self.name,
                           kernel_initializer = self.initializer)

    def build(self, input_shape):

        super(ConvLayer, self).build(input_shape)
        
        self.conv.build(input_shape)
        
        if self.apply_bn:
            conv_output_shape = self.conv.compute_output_shape(input_shape)
            self.bn.build(conv_output_shape)

    def call(self, X, training = False):

        X = self.conv(X)

        if(self.apply_bn):
            X = self.bn(X, training = training)

        if(self.activation):
            X = self.activation(X)
        else:
            print(f'Layer {self.name} has no activation function assigned.')

        return X

    def get_config(self):

        config = super(ConvLayer, self).get_config()

        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum,
            'seed': self.seed,
            'initializer': self.initializer})
        
        return config
    
    def freeze(self):
        self.trainable = False
        if( self.apply_bn ): self.bn.trainable = False

    def thaw(self):
        self.trainable = True
        if( self.apply_bn ): self.bn.trainable = True

    def is_trainable(self):
        return self.trainable

@saving.register_keras_serializable(package="Project")
class DenseLayer(Layer):
    def __init__(self,
                 units: int,
                 name: str,
                 activation = None,
                 apply_bn: bool = False,
                 bn_momentum: float = 0.99,
                 random_seed = None,
                 **kwargs):
        '''
        Implements https://github.com/luis-gonzales/pointnet_own/blob/master/src/model.py CustomDense class

        The class defines a dense (fully-connected) layer, combined with user selectable batch normalization and activation

        @param units        (int) width of the layer
        @param name         (str) name appended to layer name - i.e. 'mlp_512' results in 'mlp_512_dense_layer'
        @param activation   (tf.nn.) activation function
        @param apply_bn     (bool) apply batch normalization to layer
        @param bn_momentum  (float) if batch normalization is applied, apply this momentum
        '''
        
        super(DenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.bn = BatchNormalization(momentum = bn_momentum) if apply_bn else None
        self.name = f"{name}_dense_layer"
        self.seed = random_seed

        self.initializer = initializers.GlorotUniform( seed = self.seed ) if type( self.seed ) == int else initializers.GlorotUniform()

        # The activation function is added after a batch normalization layer
        self.dense = Dense( units = units, activation = None, use_bias = not apply_bn, name = self.name, kernel_initializer = self.initializer )

    def build(self, input_shape):

        super(DenseLayer, self).build(input_shape)
        
        self.dense.build(input_shape)
        
        if self.apply_bn:
            dense_output_shape = (input_shape[0], self.units)
            self.bn.build(dense_output_shape)

    def call(self, X, training = False):

        X = self.dense(X)

        if(self.apply_bn):
            X = self.bn(X, training = training)

        if(self.activation):
            X = self.activation(X)
        else:
            print(f'Layer {self.name} has no activation function assigned.')

        return X

    def get_config(self):

        config = super(DenseLayer, self).get_config()

        config.update({
            'units': self.units,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum,
            'seed': self.seed,
            'initializer': self.initializer})
        
        return 
    
    def freeze(self):
        self.trainable = False
        if( self.apply_bn ): self.bn.trainable = False

    def thaw(self):
        self.trainable = True
        if( self.apply_bn ): self.bn.trainable = True

    def is_trainable(self):
        return self.trainable