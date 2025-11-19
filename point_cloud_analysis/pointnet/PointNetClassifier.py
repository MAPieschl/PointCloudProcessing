'''
This module implements the original PointNet algorithm presented in

"PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017)"

and follows the author's original build at https://github.com/luis-gonzales/pointnet_own/blob/master/src/model.py

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
        vi.     bn_momentum - (float) momentum applied to batch normalization - can't tell; default is 0.99

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

from plotly.subplots import make_subplots
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Dense, Dropout
from tensorflow.keras import Model, saving

@saving.register_keras_serializable(package="Project")
class PointNetClassifier(Model):
    def __init__(self, output_width: int = 2, dropout_rate: float = 0.3, **kwargs):
        '''
        Implements https://github.com/luis-gonzales/pointnet_own/blob/master/src/model.py get_model

        This class implements the full PointNet model.

        @params output_width    (int) number of output classes
        '''

        super(PointNetClassifier, self).__init__(**kwargs)

        self.input_transform = TNet(name = 'input_transform')
        self.mlp_1_1 = ConvLayer(filters = 64, name = 's1_l1_64', activation = tf.nn.relu, apply_bn = True)
        self.mlp_1_2 = ConvLayer(filters = 64, name = 's1_l2_64', activation = tf.nn.relu, apply_bn = True)
        self.feature_transform = TNet(name = 'feature_transform', add_regularization = True)
        self.mlp_2_1 = ConvLayer(filters = 64, name = 's2_l1_64', activation = tf.nn.relu, apply_bn = True)
        self.mlp_2_2 = ConvLayer(filters = 128, name = 's2_l2_128', activation = tf.nn.relu, apply_bn = True)
        self.mlp_2_3 = ConvLayer(filters = 1024, name = 's2_l3_1024', activation = tf.nn.relu, apply_bn = True)
        self.dense_1 = DenseLayer(units = 512, name = 's3_l1_512', activation = tf.nn.relu, apply_bn = True)
        self.dropout_1 = Dropout(rate = dropout_rate)
        self.dense_2 = DenseLayer(units = 256, name = 's3_l2_256', activation = tf.nn.relu, apply_bn = True)
        self.dropout_2 = Dropout(rate = dropout_rate)
        self.dense_out = DenseLayer(units = output_width, name = 'output', activation = tf.nn.softmax)
        
    def build(self, input_shape):
        '''
        Build the model by calling build on all sub-layers
        
        @param input_shape: Expected input shape (b, n, 3)
        '''

        super(PointNetClassifier, self).build(input_shape)
        
        self.input_transform.build(input_shape)

        conv_input_shape = (input_shape[0], input_shape[1], 1, 3)
        self.mlp_1_1.build(conv_input_shape)
        
        conv_input_shape = (input_shape[0], input_shape[1], 1, 64)
        self.mlp_1_2.build(conv_input_shape)
        
        # Feature transform expects (batch_size, n_points, 64)
        feature_shape = (input_shape[0], input_shape[1], 64)
        self.feature_transform.build(feature_shape)
        
        # Continue building remaining layers
        self.mlp_2_1.build(conv_input_shape)
        
        conv_input_shape = (input_shape[0], input_shape[1], 1, 64)
        self.mlp_2_2.build(conv_input_shape)
        
        conv_input_shape = (input_shape[0], input_shape[1], 1, 128)
        self.mlp_2_3.build(conv_input_shape)
        
        # Dense layers expect (batch_size, features)
        dense_input_shape = (input_shape[0], 1024)
        self.dense_1.build(dense_input_shape)
        self.dropout_1.build(dense_input_shape)
        
        dense_input_shape = (input_shape[0], 512)
        self.dense_2.build(dense_input_shape)
        self.dropout_2.build(dense_input_shape)
        
        dense_input_shape = (input_shape[0], 256)
        self.dense_out.build(dense_input_shape)

    def call(self, input, training = False):
        # Input Transform
        R = self.input_transform(input, training = training)         # (b x 3 x 3)
        X = tf.matmul(input, R)                     # (b x n x 3)
        
        # MLP (64, 64)
        X = tf.expand_dims(X, axis = 2)         # (b x n x 1 x 3)
        X = self.mlp_1_1(X, training = training)                     # (b x n x 1 x 64)
        X = self.mlp_1_2(X, training = training)                     # (b x n x 1 x 64)
        X = tf.squeeze(X, axis = 2)             # (b x n x 64)

        # Feature Transform
        R = self.feature_transform(X, training = training)           # (b x 64 x 64)
        X = tf.matmul(X, R)                     # (b x n x 64)

        # MLP (64, 128, 1024)
        X = tf.expand_dims(X, axis = 2)         # (b x n x 1 x 64)
        X = self.mlp_2_1(X, training = training)                     # (b x n x 1 x 64)
        X = self.mlp_2_2(X, training = training)                     # (b x n x 1 x 128)
        X = self.mlp_2_3(X, training = training)                     # (b x n x 1 x 1024)
        X = tf.squeeze(X, axis = 2)             # (b x n x 1024)

        # Max pooling
        X = tf.reduce_max(X, axis = 1)          # (b x 1024)

        # Fully connected reduction to output
        X = self.dense_1(X, training = training)
        X = self.dropout_1(X, training = training)
        X = self.dense_2(X, training = training)
        X = self.dropout_2(X, training = training)

        return self.dense_out(X)
    
    def get_last_predicted_dcm(self):
        return self.input_transform.get_last_predicted_transformation()

class TNetRegressor(Model):
    def __init__(self, add_regularization: bool = False, bn_momentum: float = 0.99, tnet_layer_widths: list = [64, 128, 1024, 512, 256], **kwargs):
        '''
        This model can be used to pretrain the TNet layer of the PointNet to correctly predict the rotation of the object prior to training
        the classification network.

        @param name                 (str) network name used to label layers; ex. 'input_transform' results in 'input_transform_conv_layer_1', etc.
        @param add_regularization   (bool) apply an orthogonalizing regularization, which punishes the network for a non-orthogonality
        @param bn_momentum          (float) add momentum to the batch normalization - bn is only applied to the dense layers
        '''
        super(TNetRegressor, self).__init__(**kwargs)

        self.input_transform = TNet(name = 'input_transform', add_regularization = add_regularization, bn_momentum = bn_momentum, layer_widths = tnet_layer_widths)

    def build(self, input_shape):
        super(TNetRegressor, self).build(input_shape)

        self.input_transform.build(input_shape)

    def call(self, X, training = False):

        X = self.input_transform(X, training = training)

        return X
    
    def get_last_predicted_dcm(self):
        return self.input_transform.get_last_predicted_transformation()
    
    # def plot_training_performance(self, history, train_accuracy_key:str, val_accuracy_key: str):

    #     x = np.arange(1, len(history.history[val_accuracy_key]))

    #     fig = make_subplots(specs = [[{'secondary_y': True}]])

    #     fig.add_trace(go.Scatter(
    #         x = x,
    #         y = history.history[train_accuracy_key],
    #         mode = 'lines',
    #         name = 'Training Accuracy'
    #     ),
    #     secondary_y = False)
        
    #     fig.add_trace(go.Scatter(
    #         x = x,
    #         y = history.history['loss'],
    #         mode = 'lines',
    #         name = 'Training Loss'
    #     ),
    #     secondary_y = True)
        
    #     fig.add_trace(go.Scatter(
    #         x = x,
    #         y = history.history[val_accuracy_key],
    #         mode = 'lines',
    #         name = 'Validation Accuracy'
    #     ),
    #     secondary_y = False)
        
    #     fig.add_trace(go.Scatter(
    #         x = x,
    #         y = history.history['val_loss'],
    #         mode = 'lines',
    #         name = 'Validation Loss'
    #     ),
    #     secondary_y = True)

    #     fig.update_layout(
    #         title = 'Training Performance',
    #         xaxis_title = 'Epoch'
    #     )

    #     fig.update_yaxes(title_text = 'Accuracy', secondary_y = False)
    #     fig.update_yaxes(title_text = 'Loss', secondary_y = True)

    #     fig.show()
    
class TNet(Model):
    def __init__(self, name: str, add_regularization: bool = False, bn_momentum: float = 0.99, layer_widths: list = [64, 128, 1024, 512, 256], **kwargs):
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
        self._last_predicted = None

        # Layer Construction
        self.conv_layer_1 = ConvLayer(filters = layer_widths[0], kernel_size = (1, 1), strides = (1, 1), bn_momentum = bn_momentum, name = f"{name}_convolution_layer_1")
        self.conv_layer_2 = ConvLayer(filters = layer_widths[1], kernel_size = (1, 1), strides = (1, 1), bn_momentum = bn_momentum, name = f"{name}_convolution_layer_2")
        self.conv_layer_3 = ConvLayer(filters = layer_widths[2], kernel_size = (1, 1), strides = (1, 1), bn_momentum = bn_momentum, name = f"{name}_convolution_layer_3")
        self.dense_layer_1 = DenseLayer(units = layer_widths[3], activation = tf.nn.relu, apply_bn = True, bn_momentum = bn_momentum, name = f"{name}_dense_layer_1")
        self.dense_layer_2 = DenseLayer(units = layer_widths[4], activation = tf.nn.relu, apply_bn = True, bn_momentum = bn_momentum, name = f"{name}_dense_layer_2")

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

        # Add bias term
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
            'bn_momentum': self.bn_momentum
        })

    def get_last_predicted_transformation(self):
        return None if self._last_predicted == None else self._last_predicted

class ConvLayer(Layer):
    def __init__(self,
                 filters: int,
                 name: str,
                 kernel_size: tuple = (1, 1),
                 strides: tuple = (1, 1),
                 padding: str = 'valid',
                 activation = None,
                 apply_bn: bool = False,
                 bn_momentum: float = 0.99,
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
        self.conv = Conv2D(filters = filters, 
                           kernel_size = kernel_size, 
                           strides = strides, 
                           padding = padding,
                           activation = None, 
                           use_bias = not apply_bn,
                           name = f"{name}_convolution_layer")
        
        if(apply_bn):
            self.bn = BatchNormalization(momentum = bn_momentum)

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
            'bn_momentum': self.bn_momentum})
        
        return config
    
class DenseLayer(Layer):
    def __init__(self,
                 units: int,
                 name: str,
                 activation = None,
                 apply_bn: bool = False,
                 bn_momentum: float = 0.99,
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
        self.dense = Dense(units = units, activation = None, use_bias = not apply_bn, name = f"{name}_dense_layer")

        if(apply_bn):
            self.bn = BatchNormalization(momentum = bn_momentum)

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

        return X

    def get_config(self):

        config = super(DenseLayer, self).get_config()

        config.update({
            'units': self.units,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        
        return config