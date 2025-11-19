import sys

def _big_imports():
    '''
    Non-standard function to allow use of the help functionality without waiting for the
    large packages to import. MUST be called immediately after the program determines
    valid inputs exist.
    '''
import os
import json
import pickle
import tf2onnx
import numpy as np
import tensorflow as tf
import onnxruntime as ort
import onnx

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

from ..pointnet.PointNetSegmentation import PointNetSegmentation
from ..point_cloud.PointCloudSet import PointCloudSet

class TrainProfile:
    def __init__( self, config_file ):
        
        config: dict = {}

        try:
            with open( config_file, 'r' ) as cf:
                config = json.load( cf )

        except:
            print( f"Unable to open {config_file}" )

        try:
            # General info
            self._name = config['info']['name']
            self._class_labels = list( config['info']['class_labels'].values() )
            self._part_labels = list( config['info']['part_labels'].values() )
            self._datasets = list( config['info']['datasets'].values() )
            self._pretrained_model = config['info']['continue_training_model']

            # Training Parameters
            self._input_width = config['params']['input_width']
            self._epochs = config['params']['epochs']
            self._patience = config['params']['patience']
            self._batch_size = config['params']['batch_size']
            self._learning_rate = config['params']['learning']['rate']
            self._learning_decay_steps = config['params']['learning']['decay_steps']
            self._learning_decay_rate = config['params']['learning']['decay_rate']
            self._random_seed = config['params']['random_seed']

            # Filesystem Information (relative paths)
            self._model_path = config['file_system']['model_path']
            self._input_path = config['file_system']['input_path']

            # Verify All Paths
            if( not os.path.isdir( self._model_path ) ): return self._advise_and_abort( f"{self._model_path} does not exists" )
            if( not os.path.isdir( self._input_path ) ): return self._advise_and_abort( f"{self._input_path} does not exists" )
            for ds in self._datasets:
                if( not os.path.isfile( f'{self._input_path}{ds}' ) ): return self._advise_and_abort( f"{self._input_path}{ds} does not exists" )
            if( self._pretrained_model != "" ):
                if( not os.path.isfile( f'{self._model_path}{self._pretrained_model}' ) ): return self._advise_and_abort( f"{self._model_path}{self._pretrained_model} does not exists" )


            # Build Point Cloud
            self._pc = PointCloudSet(one_hot = True,
                                     class_labels = self._class_labels, 
                                     part_labels = self._part_labels, 
                                     pretrain_tnet = False, 
                                     network_input_width = self._input_width,
                                     batch_size = self._batch_size,
                                     rand_seed = self._random_seed)
            
            self._profile_datasets()
            
            # Build Segmentation Model
            self._model = self._build_pointnet()

        except:
            print( f"{config_file} is not the correct format. Please use example train_config_template.json." )
            return None
        
    def _profile_datasets( self ) -> None:
        if( len( self._class_labels ) != 1 ):  print( "For a segmentation network there should be only one class label. Using the first class label listed for the set." )
        for set, set_name in enumerate( self._datasets ):
            print( f"Adding data set {set} of {len( self._datasets )}" )
            self._pc.add_from_aftr_output( dir_path = f"{self._input_path}{set_name}", class_label = self._class_labels[0], shuffle_points = True )

        print( '\nDataset added successfully:\n' )
        print( self._pc.get_info() )

    def _build_pointnet( self ) -> PointNetSegmentation:

        model = PointNetSegmentation( output_width = len( self._part_labels ) )

    def _advise_and_abort( self, msg: str ) -> None:
        print( f"Error in TrainProfile:  {msg}" )
        return None

def main( *args, **kwargs ) -> bool:

    if( len(args) == 1 ):
        print_help()
        return False

    elif( args[1] == "-h" or args[1] == "--help" ):
        print_help()
        return False
    
    # Import all packages
    _big_imports()

    # Check Tensorflow
    physical_devices = tf.config.experimental.list_physical_devices( 'GPU' )
    if( len( physical_devices ) > 0 ): 
        print( 'GPUs Available: ', len( physical_devices ) )
        tf.config.experimental.set_memory_growth( physical_devices[0], True )
    else:   
        print( "No GPUs available. Would you like to continue?" )
        if( input( "(Y/n)" ) != "Y" ):
            return False
        
    # Check input config
    config_file: str = ""
    try:
        config_file = args[1]
        if( config_file.split('.')[-1] != 'json' ):
            print( f"{config_file} has incorrect type - use example train_config_template.json" )
            return False
    
    except:
        print( "Please provide a config.json file - example:\n\t> train_pointnet_seg.sh config.json" )

    # Build model
    train = TrainProfile( config_file )    

def print_help():
    print(
        '''PointNetSegmentation Training Module\n\n
        This module is capable of training both new and pretrained PointNetSegmentation models. To use,
        append a separate configuration file to the train_pointnet_seg.sh command for each training session.
        The configuration file should follow the train_config_pointnet_segmentation_template.json provided.
        In its absense, here is an overview of the configuration file required to train a PointNetSegmentation Model:
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
        \t\tcontinue_training_model: input the model name (ending in .keras) you would like to continue training; empty \n\t\t\tquotes assume a fresh model
        \t},
        \tparams: {
        \t\tinput_width: input width of the network (number of points in the point cloud)
        \t\tepochs: max number of epochs for this training session
        \t\tpatience: per tensorflow
        \t\tbatch_size: per tensorflow; a value of 0 will force the model to perform a search for the maximum batch size \n\t\t\tallowable for the GPU in use
        \t\tlearning: {
        \t\t\trate: per tensorflow
        \t\t\tdecay_steps: per tensorflow
        \t\t\tdecay_rate: per tensorflow
        \t\t},
        \t\trandom_seed: used for all random processes
        \t},
        \tfile_system: {
        \t\tmodel_path: directory where the model will be stored after training
        \t\tinput_path: directory where the datasets (above) are stored
        \t}'''
    )
    
if __name__=='__main__':
    if( main( sys.argv ) ):
        print( "Model training completed successfully." )
    else:
        print( "Model training failed." )