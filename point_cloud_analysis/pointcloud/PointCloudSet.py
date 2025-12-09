'''
Organizes point cloud data for training.

This is a robust utility class for handling large qpiont clouds. The general flow for use:

1. Instantiate the object.
2. If using AftrBurner Palindrome data, call add_from_aftr_output( dir_path ) for each dataset
   (should be the specific folder inside of DataCollect)
3. Use get_train_set(), get_val_set(), and get_test_set() as required

--------------------

PointNet.py

By:     Mike Pieschl
Date:   31 July 2025
'''

import os
import sys
import joblib
sys.path.append('')

import numpy as np
import pandas as pd
import tensorflow as tf
import utils.global_constants as constants

from tqdm import tqdm
from copy import deepcopy
from collections.abc import Callable

class PointCloudSet:
    def __init__(self,
                 name: str,
                 class_labels: list,
                 part_labels: list,
                 network_input_width: int,
                 jitter_stdev_m: np.ndarray = np.array( [ 0, 0, 0 ] ),
                 val: float = 0.15,
                 test: float = 0.10,
                 batch_size: int = 32,
                 rand_seed = None,
                 description: str = '',
                 print_func: Callable[[str], None] = print,
                 data_path: str = '' ):
        
        self._description: str = description
        self._batch_size: int = batch_size
        self._name: str = name
        self._class_labels: dict[str, int] = {}
        self._class_str: dict[int, str] = {}
        for i, label in enumerate( class_labels ):
            self._class_labels[label] = i
            self._class_str[i] = label
        self._part_labels: dict[str, int] = {}
        self._part_str: dict[int, str] = {}
        for i, label in enumerate( part_labels ):
            self._part_labels[label] = i
            self._part_str[i] = label
        self._network_input_width: int = network_input_width
        self._jitter_stdev_m: np.ndarray = jitter_stdev_m
        self._print: Callable[[str], None] = print_func
        self._data_path = data_path
        self._sets_added = 0
        self._data_size = {
            'train': {
                'count': 0,
                'class_count': {},
                'part_count': {}
            }, 
            'val': {
                'count': 0,
                'class_count': {},
                'part_count': {}
            }, 
            'test': {
                'count': 0,
                'class_count': {},
                'part_count': {}
            }
            }

        if(type(rand_seed) == int):
            np.random.default_rng(seed = rand_seed)
            self._random_seed = rand_seed
        else:
            self._random_seed = None

        if( val < 1.0 and test < 1.0 and 1.0 - (val + test) < 1.0 ):
            self._train_amt = 1.0 - (val + test)
            self._val_amt = val
            self._test_amt = test
        else:
            self._train_amt = 0.75
            self._val_amt = 0.15
            self._test_amt = 0.10
            self._print('PointCloudSet:  train_val_test_split incorrect format - set to default 75% / 15% / 10%')

        self._feature_description = {
            'observations': tf.io.FixedLenFeature([self._network_input_width * 3], tf.float32),
            'class_label':  tf.io.FixedLenFeature([], tf.int64),
            'part_labels':  tf.io.FixedLenFeature([self._network_input_width], tf.int64),
            'se3':          tf.io.FixedLenFeature([9], tf.float32),
        }
        
        if( not os.path.isdir( f"{self._data_path}{self._name}" ) ):  os.mkdir( f"{self._data_path}{self._name}" )

        self.save()

    def save( self ):

        with open( f"{self._data_path}{self._name}/pc_set.joblib", "wb" ) as jl:
            joblib.dump( self, jl )
        
    def add_from_aftr_output( self, dir_path: str, shuffle_points: bool = True ) -> bool:
        '''
        Parses the standard SensorDatumLogger output.

        TODO:  Need to add the class label into the aftrburner output and omit it
        from the input parameters of this function

        @param dir_path     (str) the path of a directory named \'collect_YYYY.Mmm.dd_hh.mm.ss.UTC\' expected content are:
                            -> Virtual Flash Lidar
                                -> _Virtual Flash Lidar__YYYY.Mmm.dd_hh.mm.ss.UTC.log
                                -> frame_0.txt
                                -> ...
                                -> frame_n.txt
                            -> _palindrome_state__YYYY.Mmm.dd_hh.mm.ss.UTC.log

        @return True if parsing is successful / False otherwise
        '''

        observations: list[np.ndarray] = []
        class_labels: list[int] = []
        part_labels: list[np.ndarray] = []
        se3: list[np.ndarray] = []

        frames_searched: int = 0
        non_num_found: int = 0
        
        collect_contents: list[str] = get_dir_contents( dir_path, self._print )
        lidar_contents: list[str] = get_dir_contents( f'{dir_path}/Virtual Flash Lidar', self._print )

        # treat the log file (containing pose information) as a csv and import into a pandas dataframe
        pose_log: list[str] = [ i for i in collect_contents if '_palindrome_state' in i ]
        if( len( pose_log ) == 1 ):
            state_info: dict = self._parse_state_info( f'{dir_path}/{pose_log[0]}' )
        else:
            raise Exception( f"No state info found in {dir_path}" )

        # Parse frames
        self._print( f"Parsing frames in {dir_path}..." )
        for i in tqdm( range( len( lidar_contents ) ) ):
            try:
                with open( f'{dir_path}/Virtual Flash Lidar/frame_{i}.txt', 'r' ) as f:
                    obs = []
                    cl = None
                    pl = []
                    se = None
                    for j, line in enumerate( f ):
                        line = line.strip()

                        # parse position
                        pos_start_idx = line.find( '(' )
                        pos_end_idx = line.find( ')' )

                        pos_str = line[pos_start_idx + 1:pos_end_idx].split( ',' )
                        pos = []
                        for val in pos_str:
                            pos.append( float( val ) )

                        # parse labels
                        labels = line[pos_end_idx + 1:].split( " " )

                        # removes unnecessary '' characters
                        labels = [ l for l in labels if len( l ) > 1 ]

                        if( len( labels ) == 2 ):
                            if( labels[0] not in self._class_labels ):
                                raise Exception( f"Class label {labels[0]} not known" )
                            if( labels[1] not in self._part_labels ):
                                raise Exception( f"Part label {labels[1]} not known" )
                        else:
                            raise Exception( f"Dataset must contain both a class label and part label. {labels} is not correct." )

                        # check for non-numeric values
                        if( np.isfinite( np.array( pos ) ).all() ):

                            # check for valid class and part labels
                            if( labels[0] in self._class_labels and labels[1] in self._part_labels ):
                                obs.append( np.array( pos ) )
                                cl = self._class_labels.get( labels[0], -1 )
                                pl.append( self._part_labels.get( labels[1], -1 ) )
                                se = state_info[i]['tanker_in_sensor_frame'][:3, :3]

                        else:
                            non_num_found += 1

                    if( len(obs) != 0 ):
                        obs, pl = self._adjust_to_input_width( np.array( obs ), np.array( pl ) )

                        if( np.isfinite( obs ).all() ):
                            observations.append( obs )
                            class_labels.append( cl )
                            part_labels.append( pl )
                            se3.append( se )

                        else:
                            self._print( f'Per-line check failed - frame_{i} discarded after detecting non-finite value.' )

            except Exception as e:
                if( frames_searched == 0 ): frames_searched = i
                self._print( f"Failed to add file {dir_path}/Virtual Flash Lidar/frame_{i}.txt:\n\t{type( e ).__name__} : {e}" )

        self.add_data( dir_path.split("/")[-1], np.array( observations ), np.array( class_labels ), np.array( part_labels ), np.array( se3 ), shuffle_points )

        return True

    def add_data( self, set_name: str, observations: np.ndarray, class_labels: np.ndarray, part_labels: np.ndarray, se3: np.ndarray, shuffle_points: bool = True ) -> None:
        '''
        Adds data to the PointCloud set and automatically separates the data into train, validate, and test sets per the ratio
        set during object instantiation. Optional shuffle parameter shuffles only the newly input data and ensures alignment of
        parallel input arrays.

        @param set_name         (str)               unique identifier for dataset
        @param observations     (np.ndarray)        (num_pc, n, 3)
        @param class_labels     (np.ndarray[str])   (num_pc, )
        @param part_labels      (np.ndarray[str])   (num_pc, n)
        @param se3              (np.ndarray)        (num_pc, 3, 3)
        @param shuffle_points   (bool)              (default = True) shuffle points using the random seed provided during instantiation          

        @return None
        '''

        if( shuffle_points ):
            indices = np.arange( observations.shape[0] )
            np.random.shuffle( indices )
            
            observations = observations[indices]
            class_labels = class_labels[indices]
            part_labels = part_labels[indices]
            se3 = se3[indices]
        
        splits = [( 0, int( np.ceil( observations.shape[0] * self._test_amt ) ) ),
                  ( int( np.ceil( observations.shape[0] * self._test_amt ) ), int( np.ceil( observations.shape[0] * self._test_amt ) ) + int( np.ceil( observations.shape[0] * self._val_amt ) ) ),
                  ( int( np.ceil( observations.shape[0] * self._test_amt ) ) + int( np.ceil( observations.shape[0] * self._val_amt ) ), observations.shape[0] )]

        if( not os.path.isdir( f"{self._data_path}{self._name}/{set_name}" ) ): os.mkdir( f"{self._data_path}{self._name}/{set_name}" )

        with tf.io.TFRecordWriter( f"{self._data_path}{self._name}/{set_name}/test_{self._sets_added}.tfrecord" ) as writer:
            for i in tqdm( range( splits[0][0], splits[0][1] ) ):

                # Account for class and part occurrences
                try:    self._data_size['test']['class_count'][self._class_str[class_labels[i]]] += 1
                except: self._data_size['test']['class_count'][self._class_str[class_labels[i]]] =  1
                for lbl in list( self._part_labels.keys() ):
                    try:        self._data_size['test']['part_count'][lbl] += int( np.count_nonzero( part_labels[i] == self._part_labels[lbl], axis = 0 ) )
                    except:     self._data_size['test']['part_count'][lbl] =  int( np.count_nonzero( part_labels[i] == self._part_labels[lbl], axis = 0 ) )

                writer.write( self._serialize_sample( observations[i], class_labels[i], part_labels[i], se3[i] ) )
                self._data_size['test']['count'] += 1

        with tf.io.TFRecordWriter( f"{self._data_path}{self._name}/{set_name}/val_{self._sets_added}.tfrecord" ) as writer:
            for i in tqdm( range( splits[1][0], splits[1][1] ) ):

                # Account for class and part occurrences
                try:    self._data_size['val']['class_count'][self._class_str[class_labels[i]]] += 1
                except: self._data_size['val']['class_count'][self._class_str[class_labels[i]]] =  1
                for lbl in list( self._part_labels.keys() ):
                    try:        self._data_size['val']['part_count'][lbl] += np.count_nonzero( part_labels[i] == self._part_labels[lbl], axis = 0 )
                    except:     self._data_size['val']['part_count'][lbl] =  np.count_nonzero( part_labels[i] == self._part_labels[lbl], axis = 0 )

                writer.write( self._serialize_sample( observations[i], class_labels[i], part_labels[i], se3[i] ) )
                self._data_size['val']['count'] += 1
                
        with tf.io.TFRecordWriter( f"{self._data_path}{self._name}/{set_name}/train_{self._sets_added}.tfrecord" ) as writer:
            for i in tqdm( range( splits[2][0], splits[2][1] ) ):

                # Account for class and part occurrences
                try:    self._data_size['train']['class_count'][self._class_str[class_labels[i]]] += 1
                except: self._data_size['train']['class_count'][self._class_str[class_labels[i]]] =  1
                for lbl in list( self._part_labels.keys() ):
                    try:        self._data_size['train']['part_count'][lbl] += np.count_nonzero( part_labels[i] == self._part_labels[lbl], axis = 0 )
                    except:     self._data_size['train']['part_count'][lbl] =  np.count_nonzero( part_labels[i] == self._part_labels[lbl], axis = 0 )

                writer.write( self._serialize_sample( observations[i], class_labels[i], part_labels[i], se3[i] ) )
                self._data_size['train']['count'] += 1

        self._sets_added += 1
        
        self.save()

    def _float_feature( self, value ):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature( self, value ):
        """Returns an int64_list from a bool / enum / int / uint."""

        if( isinstance( value, ( list, np.ndarray ) ) ):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _serialize_sample(self, obs: np.ndarray, cls: str, seg: np.ndarray, se3: np.ndarray):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Flatten arrays for storage
        obs_flat = obs.flatten().tolist()
        seg_flat = seg.flatten().tolist()
        se3_flat = se3.flatten().tolist()

        feature = {
            'observations': self._float_feature(obs_flat),
            'class_label':  self._int64_feature(cls),       # Integer index
            'part_labels':  self._int64_feature(seg_flat),  # List of Ints
            'se3':          self._float_feature(se3_flat),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    def _parse_function( self, example_proto ):
        # 1. Decode the binary blob
        ex = tf.io.parse_single_example(example_proto, self._feature_description)

        # 2. Reshape flat arrays back to Matrix shapes
        x = tf.reshape( ex['observations'], (self._network_input_width, 3) )
        y_cls = tf.cast( ex['class_label'], tf.int32 )
        y_seg = tf.cast( ex['part_labels'], tf.int32 )
        y_se3 = tf.reshape( ex['se3'], (3, 3) )

        # 3. Apply Jitter (Data Augmentation)
        noise = tf.random.normal( shape = tf.shape(x), mean = 0.0, stddev = 1.0 )
        x = x + ( noise * self._jitter_stdev_m ) 

        # 4. Format for Keras Model
        # Returns: inputs, {targets}
        return x, {
            'classification_output': y_cls,
            'segmentation_output':   y_seg,
            'se3':       y_se3 
        }

    def get_train_set( self ):

        files = tf.data.Dataset.list_files( f"{self._data_path}{self._name}/*/train_*.tfrecord" )

        dataset = files.interleave(
            tf.data.TFRecordDataset,
            cycle_length = 2,
            num_parallel_calls = tf.data.AUTOTUNE
        )

        dataset = dataset.shuffle( buffer_size = 2048 )
        dataset = dataset.repeat()
        dataset = dataset.map( self._parse_function, num_parallel_calls = tf.data.AUTOTUNE )
        dataset = dataset.batch( self._batch_size )
        dataset = dataset.prefetch( tf.data.AUTOTUNE )

        return dataset

    def get_val_set( self ):

        files = tf.data.Dataset.list_files( f"{self._data_path}{self._name}/*/val_*.tfrecord" )

        dataset = tf.data.TFRecordDataset( files )

        dataset = dataset.shuffle( buffer_size = 2048 )
        dataset = dataset.repeat()
        dataset = dataset.map( self._parse_function, num_parallel_calls = tf.data.AUTOTUNE )
        dataset = dataset.batch( self._batch_size )
        dataset = dataset.prefetch( tf.data.AUTOTUNE )

        return dataset
    
    def get_test_set( self ):

        files = tf.data.Dataset.list_files( f"{self._data_path}{self._name}/*/test_*.tfrecord" )

        dataset = tf.data.TFRecordDataset( files )

        dataset = dataset.shuffle( buffer_size = 2048 )
        dataset = dataset.repeat()
        dataset = dataset.map( self._parse_function, num_parallel_calls = tf.data.AUTOTUNE )
        dataset = dataset.batch( self._batch_size )
        dataset = dataset.prefetch( tf.data.AUTOTUNE )

        return dataset
    
    def get_description(self):
        return self._description
    
    def get_info(self):
        out = f'{self._description}\n'
        out += f'Random seed: {self._random_seed}\n' if (type(self._random_seed) == int) else f'Is not seeded\n'
        out += f'Class labels: {self._class_labels.keys()}\n'
        out += f'Part labels: {self._part_labels.keys()}\n'

        out += f'\n--- Train Set ---\n'
        out += f'Specified proportion:  {self._train_amt}\n'
        out += f"Actual proportion: {self._data_size['train']['count'] / (self._data_size['train']['count'] + self._data_size['val']['count'] + self._data_size['test']['count'])}\n"
        out += f"Total count: {self._data_size['train']['count']}\n"
        out += f'Class count:\n'
        for label in list( self._class_labels.keys() ):
            try:    out += f"\t{label}: {self._data_size['train']['class_count'][label]}\n"
            except: pass            
        out += f'Part count:\n'
        for label in list( self._part_labels.keys() ):
            try:    out += f"\t{label}: {self._data_size['train']['part_count'][label]}\n"
            except: pass

        out += f'\n--- Validation Set ---\n'
        out += f'Specified proportion:  {self._val_amt}\n'
        out += f"Actual proportion: {self._data_size['val']['count'] / (self._data_size['train']['count'] + self._data_size['val']['count'] + self._data_size['test']['count'])}\n"
        out += f"Total count: {self._data_size['val']['count']}\n"
        out += f'Class count:\n'
        for label in list( self._class_labels.keys() ):
            try:    out += f"\t{label}: {self._data_size['val']['class_count'][label]}\n"
            except: pass
        out += f'Part count:\n'
        for label in list( self._part_labels.keys() ):
            try:    out += f"\t{label}: {self._data_size['val']['part_count'][label]}\n"
            except: pass

        out += f'\n--- Test Set ---\n'
        out += f'Specified proportion:  {self._test_amt}\n'
        out += f"Actual proportion: {self._data_size['test']['count'] / (self._data_size['train']['count'] + self._data_size['val']['count'] + self._data_size['test']['count'])}\n"
        out += f"Total count: {self._data_size['test']['count']}\n"
        out += f'Class count:\n'
        for label in list( self._class_labels.keys() ):
            try:    out += f"\t{label}: {self._data_size['test']['class_count'][label]}\n"
            except: pass
        out += f'Part count:\n'
        for label in list( self._part_labels.keys() ):
            try:    out += f"\t{label}: {self._data_size['test']['part_count'][label]}\n"
            except: pass

        return out
    
    def _adjust_to_input_width( self, observations: np.ndarray, part_labels: np.ndarray ) -> tuple:
        '''
        Adjusts the input parameters to a uniform arrays of length _network_input_width by either splicing the first 
        _network_input_width samples from the oversized array, or appending a uniform sampling of exiting points. This
        method ensures that points remain aligned with their label when duplicated

        @param observations (np.ndarray) (n,3)
        @param part_labels  (np.ndarray) (n,)

        @return (observations (np.ndarray), class_labels(np.ndarray), part_labels(np.ndarray), se3(np.ndarray))
        '''

        if( observations.shape[0] > self._network_input_width ):
            return observations[:self._network_input_width], part_labels[:self._network_input_width]
        
        else:
            repeated_indices = np.random.uniform( 0, observations.shape[0], self._network_input_width - observations.shape[0] )
            repeated_indices = repeated_indices.astype(np.int_)

            additional_obs = deepcopy( observations[repeated_indices] )
            observations = np.concatenate( ( observations, additional_obs ), axis = 0 )
            assert observations.shape[0] == self._network_input_width, f'Failed to adjust observations to the network input width - should be {self._network_input_width}, not {observations.shape[0]}'

            additional_pl = deepcopy( part_labels[repeated_indices] )
            part_labels = np.concatenate( ( part_labels, additional_pl ), axis = 0 )
            assert part_labels.shape[0] == self._network_input_width, f'Failed to adjust part_labels to the network input width - should be {self._network_input_width}, not {part_labels.shape[0]}'

        return observations, part_labels
    
    def _parse_state_info( self, filepath: str ) -> dict:
        '''
        Parses the _palindrom_state_... file in the AftrBurner data collection. The output dictionary
        contains the time information of each frame, the frame number, and all SE3 matrices defined in
        the file as an np.ndarray with shape = (4, 4). One additional SE3 matrix, tanker_in_sensor_frame,
        is included in the dictionary for direct reading of the tanker pose in the sensor frame.
        '''

        with open( filepath, 'r' ) as f:

            # get the descriptor line
            keys = f.readline().strip().split( "   " )

            # removes unnecessary '' characters
            keys = [ i for i in keys if len( i ) > 1 ]

            # readline above incremented the iterator - this will grab the data only
            data: dict = {}
            for line in f:
                data_line = line.strip().split( " " )
                data[ int( data_line[1] ) ] = {}

                data[ int( data_line[1] ) ][ keys[0] ] = data_line[0]
                data[ int( data_line[1] ) ][ keys[1] ] = data_line[1]

                for i, key in enumerate( keys[2:] ):
                    cols = []
                    for col in range( constants.SE3_COLS ):
                        cols.append( data_line[ 2 + i * constants.SE3_SIZE + col * constants.SE3_ROWS : 2 + i * constants.SE3_SIZE + ( col + 1 ) * constants.SE3_ROWS ] )
                    data[ int( data_line[1] ) ][ key ] = np.array( cols, dtype = np.float64 ).T
                
                if( 'Sensor Pose' in keys and 'Tanker Pose' in keys ):
                    so3 = data[ int( data_line[1] ) ]['Sensor Pose'][:3, :3].T @ data[ int( data_line[1] ) ]['Tanker Pose'][:3, :3]
                    t_t_s = data[ int( data_line[1] ) ]['Sensor Pose'][:3, :3].T @ ( data[ int( data_line[1] ) ]['Tanker Pose'][:3, 3:] - data[ int( data_line[1] ) ]['Sensor Pose'][:3, 3:] )
                    se3_partial = np.concatenate( [so3, t_t_s], axis = 1 )
                    data[ int( data_line[1] ) ]['tanker_in_sensor_frame'] = np.concatenate( [se3_partial, np.array( [[0, 0, 0, 1]] )], axis = 0 )
        
        return data

### FREE HELPER FUNCTIONS ###
def load_from_file( joblib_file: str ) -> PointCloudSet:
    
    # Add file extension if not present
    if( joblib_file.split(".")[-1] != '.pkl' ):
        joblib_file += '.pkl'

    with open( joblib_file, 'rb' ) as pf:
        pc_set: PointCloudSet = joblib.load( pf )

    return pc_set

def get_dir_contents(dir_path: str, _print: Callable[[str], None] = print) -> list:

    try:
        contents = os.listdir(dir_path)
        if(not contents):   return []
        else:               return contents

    except FileNotFoundError:
        _print(f"Error: The directory '{dir_path}' was not found.", file = sys.stderr)
    except NotADirectoryError:
        _print(f"Error: The path '{dir_path}' is not a directory.", file = sys.stderr)
    except PermissionError:
        _print(f"Error: Permission denied to read '{dir_path}'.", file = sys.stderr)
    except Exception as e:
        _print(f"An error occurred: {e}", file = sys.stderr)

    return []