print( "Importing packages..." )

import sys
import os
import json
import pickle
import shutil
import tf2onnx
import numpy as np
import tensorflow as tf
import onnxruntime as ort
import onnx

from pointnet.PointNet import PointNet
from point_cloud.PointCloudSet import PointCloudSet

from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping

print( "Package import complete." )

class TrainProfile:
    def __init__( self, config_file ):
        '''
        TrainProfile takes care of all of the initialization steps prior to model training. Instantiation returns a 
        TrainProfile model, instantiation was successful and .train() can be called to begin training.
        '''
        
        config: dict = {}

        try:
            with open( config_file, 'r' ) as cf:
                config = json.load( cf )

        except:
            print( f"Unable to open {config_file}" )

        # save config path ( will be saved in model folder )
        self._config_file = config_file

        # general info
        self._name                              : str   = config['info']['name']
        self._class_labels                      : list  = list( config['info']['class_labels'].values() )
        self._part_labels                       : list  = list( config['info']['part_labels'].values() )
        self._training_profiles                 : dict  = config['info']['training_profiles']
        self._pretrained_model                  : str   = config['info']['continue_training_model']

        # training parameters
        self._input_width                       : int   = config['params']['input_width']
        self._epochs                            : int   = config['params']['epochs']
        self._patience                          : int   = config['params']['patience']
        self._batch_size                        : int   = config['params']['batch_size']
        self._learning_rate                     : float = config['params']['learning']['rate']
        self._learning_decay_steps              : int   = config['params']['learning']['decay_steps']
        self._learning_decay_rate               : float = config['params']['learning']['decay_rate']
        self._random_seed                       : int   = config['params']['random_seed']
        self._debugging                         : bool = config['params']['debugging']

        # filesystem information (relative paths)
        self._model_path                        : str = config['file_system']['model_path']  
        self._input_path                        : str = config['file_system']['input_path']
        
        # set debug mode
        if( self._debugging ): tf.config.run_functions_eagerly( True )

        # verify all paths
        if( not os.path.isdir( self._model_path ) ): return self._advise_and_abort( f"{self._model_path} does not exists" )
        if( not os.path.isdir( self._input_path ) ): return self._advise_and_abort( f"{self._input_path} does not exists" )
        for prof in list(self._training_profiles.keys()):
            for ds in list(self._training_profiles[prof]['datasets'].values()):
                if( not os.path.isdir( f'{self._input_path}{ds}' ) ): return self._advise_and_abort( f"{self._input_path}{ds} does not exists" )
        if( self._pretrained_model != "" ):
            if( not os.path.isfile( f'{self._model_path}{self._pretrained_model}' ) ): return self._advise_and_abort( f"{self._model_path}{self._pretrained_model} does not exists" )

        # create model training data directory
        self._specific_model_path = f"{self._model_path}{self._name}/"
        if( not os.path.isdir(self._specific_model_path) ):
            os.mkdir(self._specific_model_path)

        # create a pc set and training subdirectory for each training step
        for prof in list(self._training_profiles.keys()):

            # create pc set
            self._training_profiles[prof]['pc'] = PointCloudSet( one_hot = True,
                                                                 class_labels = self._class_labels,
                                                                 part_labels = self._part_labels,
                                                                 network_input_width = self._input_width,
                                                                 jitter_stdev_m = np.ndarray( [ self._training_profiles[prof]['noise']['x_stdev_n'], \
                                                                                                self._training_profiles[prof]['noise']['y_stdev_n'], \
                                                                                                self._training_profiles[prof]['noise']['z_stdev_n'], ] ),
                                                                 batch_size = 2,
                                                                 rand_seed = 42,
                                                                 description = prof )

            # create training subdirectory
            self._training_profiles[prof]['path'] = f"{self._specific_model_path}{prof}"
            if( not os.path.isdir(self._training_profiles['path']) ):
                os.mkdir(self._training_profiles['path'])
            
        
    def train( self ):
        '''
        Train the model and save it to the given:
            model_path/name/name.keras
            model_path/name/name.onnx
        '''

        train = self._pc.get_train_seg_set()
        val = self._pc.get_val_seg_set()

        # Train model
        history = self._model.fit( x = train, epochs = self._epochs, verbose = 1, validation_data = val, callbacks = self._training_callbacks )

        # Save Keras model
        self._model.save(f'{self._specific_model_path}{self._name}.keras')

        # Output training history
        with open(f'{self._specific_model_path}{self._name}_history.json', 'w') as j:
            json.dump({
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'categorical_accuracy': history.history['categorical_accuracy'],
                'val_categorical_accuracy': history.history['val_categorical_accuracy'],
            }, j)

        # Save ONNX model
        input_signature = [
            tf.TensorSpec((None, self._input_width, 3), dtype = tf.float32)
        ]

        onnx_model, _ = tf2onnx.convert.from_keras(
            self._model,
            input_signature = input_signature,
            opset = 13
        )

        onnx.save(onnx_model, f'{self._specific_model_path}{self._name}.onnx')

        # Copy config file into specific model directory
        shutil.copy( self._config_file, self._specific_model_path )
        
    def _profile_datasets( self ) -> None:
        for ds, set_name in enumerate( self._datasets ):
            print( f"Adding data set {ds + 1} of {len( self._datasets )}" )
            self._pc.add_from_aftr_output( dir_path = f"{self._input_path}{set_name}", class_label = self._class_labels[0], shuffle_points = True )

        print( '\nDataset added successfully:\n' )
        print( self._pc.get_info() )

    def _build_pointnet( self ):

        model = PointNet( classification_output_width = len( self._class_labels ),
                         segmentation_output_width = len( self._part_labels ), 
                         dropout_rate = 0.3,
                         random_seed = self._random_seed, 
                         debugging = self._debugging )
        
        model.build(input_shape = (None, self._input_width, 3))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            self._learning_rate,
            decay_steps = self._learning_decay_steps,
            decay_rate = self._learning_decay_rate,
            staircase = False
        )

        optimizer = keras.optimizers.Adam(
            learning_rate = lr_schedule,
            global_clipnorm = 1.0
        )

        def debug_loss(y_true, y_pred):
            y_true = tf.debugging.check_numerics(y_true, "Labels (y_true) contain NaNs or Infs")
            loss = keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
            loss = tf.debugging.check_numerics(loss, "Loss calculation produced NaN")
            return loss

        model.compile(
            optimizer = optimizer,
            loss = debug_loss if self._debugging else keras.losses.CategoricalCrossentropy( from_logits = True ),
            metrics = [keras.metrics.CategoricalAccuracy()]
        )

        self._training_callbacks.append( EarlyStopping(
            monitor = 'val_loss',
            patience = self._patience,
            verbose = 1,
            restore_best_weights = True
        ) )

        return model
    
    def _advise_and_abort( self, msg: str ) -> None:
        print( f"Error in TrainProfile:  {msg}" )
        return None

def train_pointnet_seg( *args, **kwargs ) -> bool:
    configs = args[0][1:]

    if( len(configs) == 0 ):
        print_help()
        return False

    elif( args[0] == "-h" or args[0] == "--help" ):
        print_help()
        return False
    
    for cf in configs:
        try:
            if( cf.split('.')[-1] != 'json' ):
                print( f"{cf} has incorrect type - use example train_config_template.json" )
                return False
        
        except:
            print( "Please provide a config.json file - example:\n\t> train_pointnet_seg.sh config.json" )

    # Check Tensorflow
    physical_devices = tf.config.experimental.list_physical_devices( 'GPU' )
    if( len( physical_devices ) > 0 ): 
        print( 'GPUs Available: ', len( physical_devices ) )
        tf.config.experimental.set_memory_growth( physical_devices[0], True )

    else:   
        print( "No GPUs available. Would you like to continue?" )
        if( input( "(Y/n)" ) != "Y" ):
            return False

    # Build model
    for cf in configs:
        tp = TrainProfile( cf )
        if( type(tp) != None ):
            tp.train()
        else:   return False

    return True

def print_help():
    print(
        '''PointNetSegmentation Training Module\n\n
        This module is capable of training both new and pretrained PointNetSegmentation models. The configuration file 
        must follow the examples/train_config_pointnet_segmentation_template.json provided. In its absense, here is an 
        overview of the configuration file required to train a PointNetSegmentation Model:
        {
        \tinfo: {
        \t\tname: this will be the output name for model trained
        \t\tclass_labels: {
        \t\t\t0: for segmentation models, there should be only one class label (this is the class of the entire object)
        \t\t},
        \t\tpart_labels: {
        \t\t\t0: input a new row for each segmented part; these labels must match the mesh labels in the virtual training\n\t\t\t\tpipeline 
        \t\t\t1: ...
        \t\t},
        \t\tdatasets: {
        \t\t\t0: add a row for each dataset; the name should match the outer folder from AftrBurner, such as -> \n\t\t\t\tcollect_2025.Nov.19_00.33.24.3472488.UTC
        \t\t},
        \t\tcontinue_training_model: input the model directory you would like to continue training; the directory \n\t\t\tMUST reside in the model path listed below and exist in the format output by this program; empty \n\t\t\tquotes assume a fresh model
        \t},
        \tparams: {
        \t\tinput_width: input width of the network (number of points in the point cloud)
        \t\tepochs: max number of epochs for this training session
        \t\tpatience: per tensorflow
        \t\tbatch_size: per tensorflow (>= 1)
        \t\tlearning: {
        \t\t\trate: per tensorflow
        \t\t\tdecay_steps: per tensorflow
        \t\t\tdecay_rate: per tensorflow
        \t\t},
        \t\trandom_seed: used for all random processes
        \t\tdebugging: sets Tensorflow to run eagerly and checks for non-numerics at each layer
        \t\tjitter_stdev_m: the standard deviation of the jitter applied to the dataset in meters -> 0 will apply no jitter
        \t},
        \tfile_system: {
        \t\tmodel_path: directory where the model will be stored after training
        \t\tinput_path: directory where the datasets (above) are stored
        \t}'''
    )

if __name__=='__main__':
    if( train_pointnet_seg( sys.argv ) ):
        print( "Model training completed successfully." )
    else:
        print( "Model training failed." )