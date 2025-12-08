### For interactive mode, use the run commands below; this will prevent lengthy imports every time
### the program runs (such as Tensorflow) -- recommend running from the terminal once fully tested
### to avoid issues/limitations with the Jupyter environment
###
### The autoreload modes are:
### - 0: no reloads
### - 1: whitelisted reloads only (using %aimport module_name)
### - 2: reload any module that has changed since the last import

# %%
# %load_ext autoreload
# %autoreload 2

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
import joblib
import logging
import datetime

from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping

import pointnet.PointNet as PointNet
import pointcloud.PointCloudSet as PointCloudSet

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
        self._reg_input_transform               : bool = config['params']['regularize_input_transform']
        self._reg_feature_transform             : bool = config['params']['regularize_feature_transform']

        # filesystem information (relative paths)
        self._model_path                        : str = config['file_system']['model_path']  
        self._input_path                        : str = config['file_system']['input_path']
        self._data_path                         : str = config['file_system']['data_path']

        # leave empty callback list (appended in _build_pointnet)
        self._training_callbacks = []
        
        # set debug mode
        if( self._debugging ): tf.config.run_functions_eagerly( True )

        # verify all paths
        if( not os.path.isdir( self._model_path ) ): return self._advise_and_abort( f"{self._model_path} does not exists" )
        if( not os.path.isdir( self._input_path ) ): return self._advise_and_abort( f"{self._input_path} does not exists" )
        if( not os.path.isdir( self._data_path ) ):  return self._advise_and_abort( f"{self._data_path} does not exist" )
        for prof in list(self._training_profiles.keys()):
            for ds in list(self._training_profiles[prof]['datasets'].values()):
                if( not os.path.isdir( f"{self._input_path}{ds}" ) ): return self._advise_and_abort( f"{self._input_path}{ds} does not exists" )
        if( self._pretrained_model != "" ):
            if( not os.path.isfile( f"{self._model_path}{self._pretrained_model}" ) ): return self._advise_and_abort( f"{self._model_path}{self._pretrained_model} does not exists" )

        # create model training data directory
        self._specific_model_path = f"{self._name}/"
        if( not os.path.isdir(f"{self._model_path}{self._specific_model_path}") ):
            os.mkdir(f"{self._model_path}{self._specific_model_path}")

        # create logger
        dt = datetime.datetime.now()
        self._log = logging.getLogger()
        self._log.setLevel( logging.DEBUG )

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler( f"{self._model_path}{self._specific_model_path}log_{dt.strftime( '%Y%m%d_%H:%M%S' )}.log" )

        console_handler.setFormatter( logging.Formatter('%(name)s - %(levelname)s - %(message)s') )
        file_handler.setFormatter( logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') )

        self._log.addHandler( console_handler )
        self._log.addHandler( file_handler )

        # create a pc set and training subdirectory for each training step
        for prof in list(self._training_profiles.keys()):

            if( os.path.isdir( f"{self._data_path}{self._name}_{prof}" ) ):
                self._log.info( f"Training profile {self._name}_{prof} already exists. Using existing profile..." )

                with open( f"{self._data_path}{self._name}_{prof}/pc_set.joblib", "rb" ) as jl:
                    self._training_profiles[prof]['pc'] = joblib.load( jl )

            else:
                # create pc set
                self._training_profiles[prof]['pc'] = PointCloudSet.PointCloudSet( name = f"{self._name}_{prof}",
                                                                                   class_labels = self._class_labels,
                                                                                   part_labels = self._part_labels,
                                                                                   network_input_width = self._input_width,
                                                                                   jitter_stdev_m = np.array( [ self._training_profiles[prof]['noise']['x_stdev_m'], \
                                                                                                                self._training_profiles[prof]['noise']['y_stdev_m'], \
                                                                                                                self._training_profiles[prof]['noise']['z_stdev_m'] ] ),
                                                                                   batch_size = 2,
                                                                                   rand_seed = 42,
                                                                                   description = prof,
                                                                                   print_func = self._log.info,
                                                                                   data_path = self._data_path )
                
            self._profile_datasets( prof )

            # create training subdirectory
            self._training_profiles[prof]['path'] = f"{self._specific_model_path}{prof}/"
            if( not os.path.isdir( f"{self._model_path}{self._training_profiles[prof]['path']}" ) ):
                os.mkdir( f"{self._model_path}{self._training_profiles[prof]['path']}" )
            
    def train( self ):
        '''
        This function goes through each of the training stages defined in under training_profiles in the config.json provided. Models are output to
        file_system.model_path/info.name/training_profiles/. Each model directory will contain:
            {info.name}.onnx
            {info.name}.keras
            {info.name}_history.json
            {config}.json
            {continue_training_model}/
        where {continue_training_model}/ contains all of the above components for the model upon which the current model was trained. This builds a
        full history of the model through recursion.
        '''
        for prof in list(self._training_profiles.keys()):

            model = self._build_pointnet( prof )

            self._log.info( f"PointNet Build" )
            self._log.info( f"\tTrainable Layers" )

            trainability_summary = model.get_layer_trainability()
            for l in list( trainability_summary.keys() ):
                self._log.info( f"\t\t-> {l}: {trainability_summary[l]}" )

            # train model
            history = model.fit( x = self._training_profiles[prof]['pc'].get_train_set(), 
                                 validation_data = self._training_profiles[prof]['pc'].get_val_set(),
                                 epochs = self._epochs, 
                                 verbose = 1,  
                                 callbacks = self._training_callbacks,
                                 steps_per_epoch = int( self._training_profiles[prof]['pc']._data_size['train']['count'] / self._batch_size ),
                                 validation_steps = int( self._training_profiles[prof]['pc']._data_size['val']['count'] / self._batch_size )
            )

            # save Keras model
            model.save(f"{self._model_path}{self._training_profiles[prof]['path']}{self._name}_{prof}.keras")

            # output training history
            with open(f"{self._model_path}{self._training_profiles[prof]['path']}{self._name}_{prof}_history.json", 'w') as j:
                json.dump(history.history, j)

            # save ONNX model
            input_signature = [
                tf.TensorSpec((None, self._input_width, 3), dtype = tf.float32)
            ]

            onnx_model, _ = tf2onnx.convert.from_keras(
                model,
                input_signature = input_signature,
                opset = 13
            )

            onnx.save(onnx_model, f"{self._model_path}{self._training_profiles[prof]['path']}{self._name}_{prof}.onnx")

            # copy config file into specific model directory
            shutil.copy( self._config_file, f"{self._model_path}{self._training_profiles[prof]['path']}" )

            # copy pretrained model into current directory
            path_list = f"{self._model_path}{self._pretrained_model}".split("/")[:-1]
            if( os.path.isdir( f"{self._model_path}{self._pretrained_model}" ) and self._pretrained_model != "" ):  shutil.copytree( '/'.join(path_list), f"{self._model_path}{self._training_profiles[prof]['path']}" )
        
            self._pretrained_model = f"{self._training_profiles[prof]['path']}{self._name}_{prof}.keras"

    def _profile_datasets( self, profile ) -> None:

        datasets = PointCloudSet.get_dir_contents( f"{self._data_path}{self._name}_{profile}", self._log.info )

        if( len( datasets ) > 0 ):
            self._log.info( f"The following datasets were found in {self._data_path}{self._name}_{profile}:" )
            for ds in datasets:
                self._log.info( f"\t-> {ds}\t{'(not requested, but will included in training profile)' if ds in list( self._training_profiles[profile]['datasets'].values() ) else ''}" )

        for ds, set_name in enumerate( list( self._training_profiles[profile]['datasets'].values() ) ):
            if( set_name not in datasets ):
                self._log.info( f"Adding data set {ds + 1} of {len( self._training_profiles[profile]['datasets'] )}" )
                self._training_profiles[profile]['pc'].add_from_aftr_output( dir_path = f"{self._input_path}{set_name}", shuffle_points = True )

        self._log.info( '\nDatasets added successfully:\n' )
        self._log.info( self._training_profiles[profile]['pc'].get_info() )

    def _build_pointnet( self, profile: str ):

        self._training_callbacks = []

        if( self._pretrained_model != "" ):

            self._log.info( f"Continuing training on model {self._pretrained_model}" )
            
            custom_objects = {
                "PointNetSegmentation": PointNet.PointNet,
                "TNet": PointNet.TNet,
                "ConvLayer": PointNet.ConvLayer,
                "DenseLayer": PointNet.DenseLayer
            }

            model = tf.keras.models.load_model(
                f"{self._model_path}{self._pretrained_model}",
                custom_objects = custom_objects
            )

        else:

            model = PointNet.PointNet( classification_output_width = len( self._class_labels ),
                                       segmentation_output_width = len( self._part_labels ), 
                                       dropout_rate = 0.3,
                                       random_seed = self._random_seed, 
                                       debugging = self._debugging,
                                       regularize_input_transform = self._reg_input_transform,
                                       regularize_feature_transform = self._reg_feature_transform )
            
            model.build(input_shape = (None, self._input_width, 3))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            self._learning_rate,
            decay_steps = self._learning_decay_steps,
            decay_rate = self._learning_decay_rate,
            staircase = False
        )

        optimizer = keras.optimizers.Adam(
            learning_rate = lr_schedule
        )

        # Freeze / thaw specified layers
        if( self._training_profiles[profile]['trainable']['shared_network'] ):  model.thaw_shared_network()
        else:                                                                   model.freeze_shared_network()
        
        if( self._training_profiles[profile]['trainable']['input_transform'] ): model.thaw_input_transform()
        else:                                                                   model.freeze_input_transform()
        
        if( self._training_profiles[profile]['trainable']['classification_head'] ): model.thaw_classification_head()
        else:                                                                       model.freeze_classification_head()
        
        if( self._training_profiles[profile]['trainable']['segmentation_head'] ):   model.thaw_segmentation_head()
        else:                                                                       model.freeze_segmentation_head()

        model.compile(
            optimizer = optimizer,
            loss = {
                'classification_output': keras.losses.SparseCategoricalCrossentropy(),
                'segmentation_output': keras.losses.SparseCategoricalCrossentropy(),
                'se3': keras.losses.MeanSquaredError()
            },
            loss_weights = {
                'classification_output': self._training_profiles[profile]['loss_weights']['classification'],
                'segmentation_output': self._training_profiles[profile]['loss_weights']['segmentation'],
                'se3': self._training_profiles[profile]['loss_weights']['rotation']
            },
            metrics = {
                'classification_output': [keras.metrics.SparseCategoricalAccuracy()],
                'segmentation_output': [keras.metrics.SparseCategoricalAccuracy()],
                'se3': [keras.metrics.RootMeanSquaredError()]
            }
        )

        self._training_callbacks.append( EarlyStopping(
            monitor = 'val_loss',
            patience = self._patience,
            verbose = 1,
            restore_best_weights = True
        ) )

        return model
    
    def _advise_and_abort( self, msg: str ) -> None:

        try:
            self._log.warning( f"Error in TrainProfile:  {msg}" )
        except AttributeError:
            print( f"Error in TrainProfile:  {msg}" )

        return None

def train_pointnet( *args, **kwargs ) -> bool:
    configs = [ i for i in args[0] if i.split( '.' )[-1] == 'json' ]

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
        must follow the examples/template_config.json provided. In its absense, here is an 
        overview of the configuration file required to train a PointNet Model. NOTE:  The file name MUST end in {somename}_config.json:
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
    if( not any([ i.split('_')[-1] == 'config.json' for i in sys.argv ]) ):
        sys.argv = ['vizzer_config.json']
        print( f"No config file found. Defaulting to: {sys.argv}" )

    if( train_pointnet( sys.argv ) ):
        print( "Model training completed successfully." )
    else:
        print( "Model training failed." )