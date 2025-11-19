import sys
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

from pointnet.PointNetSegmentation import PointNetSegmentation
from point_cloud.PointCloudSet import PointCloudSet

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
            self._datasets = list( config['info']['datasets'].values() )

            # Training Parameters

        except:
            print( f"{config_file} is not the correct format. Please use example train_config_template.json." )
            return None

def main( *args, **kwargs ) -> bool:

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
        config_file = args[0]
        if( config_file.split('.')[-1] != 'json' ):
            print( f"{config_file} has incorrect type - use example train_config_template.json" )
            return False
    
    except:
        print( "Please provide a config.json file - example:\n\t> train_pointnet_seg.sh config.json" )

    # Build model
    train = TrainProfile( config_file )    
    
if __name__=='__main__':
    if( main( sys.argv ) ):
        print( "Model training completed successfully." )
    else:
        print( "Model training failed." )