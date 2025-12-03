'''
Organizes point cloud data for training.

This is a robust utility class for handling large qpiont clouds. The general flow for use:

1. Instantiate the object.
2. If using AftrBurner Palindrome data, call add_from_aftr_output( dir_path ) for each dataset
   (should be the specific folder inside of DataCollect)
3. Use get_train_set(), get_val_set(), and get_test_set() as required

The class automatically manages memory usage, organizes the data into train/val/test sets, and 
jitters the points as needed.

If reloading a saved model, call:

pc = PointCloudSet.load_from_file( pickle_file.pkl )

--------------------

PointNet.py

By:     Mike Pieschl
Date:   31 July 2025
'''

import os
import sys
import pickle
import time
import psutil
import uuid
import gc
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
                 one_hot: bool,
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
                 save_to_filename: str = '' ):
        
        self._description: str = description
        self._batch_size: int = batch_size
        self._one_hot: bool = one_hot
        self._class_labels: list[str] = class_labels
        self._part_labels: list[str] = part_labels
        self._network_input_width: int = network_input_width
        self._jitter_stdev_m: np.ndarray = jitter_stdev_m
        self._print: Callable[[str], None] = print_func
        self._save_to_filename: str = save_to_filename
        self.uid = uuid.uuid4()

        # Memory management tools
        self._cached_set_paths: list[str] = []

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

        self._data_size = {'train': 0, 'val': 0, 'test': 0}

        self.save()

    def __del__( self ):

        # Delete cached datasets if the PointCloudSet is saved
        if( self._save_to_filename == '' ):
            for cached_pc in self._cached_set_paths:
                if( os.path.exists( cached_pc ) ):
                    os.remove( cached_pc )
                else:
                    # Python does not garuantee that the object underlying
                    # self._print will exist beyond this class; if it does
                    # not, print to console
                    try:
                        self._print( f"Failed to remove {cached_pc}" )
                    except Exception as e:
                        print( f"Failed to remove {cached_pc}:\n{type(e).__name__}: {e}'" )

    def save( self ):

        if( self._save_to_filename != '' ):

            try:
                with open( self._save_to_filename, 'wb' ) as pf:
                    pickle.dump( self, pf )

            except Exception as e:
                self._print( f'Failed to save PointCloudSet to {self._save_to_filename}:\n{type(e).__name__}: {e}' )

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

        has_class_labels = None
        has_part_labels = None
        has_state_info = None

        frame_id: list[str] = []
        observations: list[np.ndarray] = []
        class_labels: list[str] = []
        part_labels: list[str] = []
        se3: list[np.ndarray] = []

        frames_searched: int = 0
        non_num_found: int = 0
        
        collect_contents: list[str] = get_dir_contents( dir_path, self._print )
        lidar_contents: list[str] = get_dir_contents( f'{dir_path}/Virtual Flash Lidar', self._print )

        # treat the log file (containing pose information) as a csv and import into a pandas dataframe
        pose_log: list[str] = [ i for i in collect_contents if '_palindrome_state' in i ]
        if( len( pose_log ) == 1 ):
            state_info: dict = self._parse_state_info( f'{dir_path}/{pose_log[0]}' )
            has_state_info = True
        else:
            state_info: dict = {}
            has_state_info = False
            self._print( f"No pose information file (which must contain the substring '_palindrome_state') was found. No pose information will be recorded from data collect." )

        # Memory check
        mem_available_GB = psutil.virtual_memory().available / (1024 ** 3)
        mem_required_GB = ( len( lidar_contents ) / 1000 ) * constants.GB_PER_1000_SAMPLES
        if( mem_available_GB < mem_required_GB ):
            self._print( f"Unable to add dataset. Please reduce size (< {1000 * ((mem_required_GB - mem_available_GB) / constants.GB_PER_1000_SAMPLES)}) or open up at least {mem_required_GB - mem_available_GB} GB of memory." )
            return False

        # Parse frames
        self._print( f"Parsing frames in {dir_path}..." )
        for i in tqdm( range( len( lidar_contents ) ) ):
            try:
                with open( f'{dir_path}/Virtual Flash Lidar/frame_{i}.txt', 'r' ) as f:
                    obs = []
                    cl = []
                    pl = []
                    se = []
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

                        # determine if dataset has class labels and part labels on first iteration
                        if( type( has_class_labels ) == type( None ) and type( has_part_labels ) == type( None ) ):

                            if( len( labels ) > 0 ):
                                if( labels[0] in self._class_labels ):  has_class_labels = True
                            else:
                                has_class_labels = False
                                self._print( f"No class labels found in {dir_path}. Ignoring class labels for this set." )
                            
                            if( len( labels ) > 1 ):
                                if( labels[1] in self._part_labels ): has_part_labels = True
                            else:
                                has_part_labels = False
                                self._print( f"No part labels found in {dir_path}. Ignoring part labels for this set." )

                        # check for non-numeric values
                        if( np.isfinite( np.array( pos ) ).all() ):

                            # add data here if a valid class label exists, but no part label
                            if( len( labels ) > 0 and has_class_labels and not has_part_labels ):

                                # check for valid class label
                                if( labels[0] in self._class_labels ):
                                    obs.append( np.array( pos ) )
                                    cl.append( labels[0] )

                                if( has_state_info ):
                                    se.append( state_info[i]['tanker_in_sensor_frame'] )

                            # add data here if both a valid class and part label exist
                            elif( len( labels ) > 1 and has_class_labels and has_part_labels ):

                                # check for valid class and part labels
                                if( labels[0] in self._class_labels and labels[1] in self._part_labels ):
                                    obs.append( np.array( pos ) )
                                    cl.append( labels[0] )
                                    pl.append( labels[1] )

                                if( has_state_info ):
                                    se.append( state_info[i]['tanker_in_sensor_frame'] )

                            else:
                                self._print( f'No valid class or part label found for frameID {dir_path}_frame_{i}, line {j}' )
                        else:
                            non_num_found += 1

                    if( len(obs) != 0 ):
                        obs, cl, pl, se = self._adjust_to_input_width( np.array( obs ), np.array( cl ), np.array( pl ), np.array( se ) )

                        if( np.isfinite( obs ).all() ):
                            frame_id.append( f"{dir_path}_frame_{i}" )
                            observations.append( obs )
                            class_labels.append( cl )
                            part_labels.append( pl )
                            se3.append( se )

                        else:
                            self._print( f'Per-line check failed - frame_{i} discarded after detecting non-finite value.' )

            except:
                if( frames_searched == 0 ): frames_searched = i
                self._print( f"Failed to add file {dir_path}/Virtual Flash Lidar/frame_{i}.txt" )

        self.add_data( np.array( frame_id ), np.array( observations ), np.array( class_labels ), np.array( part_labels ), np.array( se3 ), shuffle_points )
            
        self._print( f'{dir_path} parsed:  found {len( frame_id )} valid frames out of {frames_searched} total. {non_num_found} total lines discarded for non-numeric values.' )

        self.save()

        return True

    def add_data( self, frame_id: np.ndarray, observations: np.ndarray, class_labels: np.ndarray, part_labels: np.ndarray, se3: np.ndarray, shuffle_points: bool = True ) -> None:
        '''
        Adds data to the PointCloud set and automatically separates the data into train, validate, and test sets per the ratio
        set during object instantiation. Optional shuffle parameter shuffles only the newly input data and ensures alignment of
        parallel input arrays.

        @param frame_id         (np.ndarray[str])   (num_pc,)
        @param observations     (np.ndarray)        (num_pc, n, 3)
        @param class_labels     (np.ndarray[str])   (num_pc, n)
        @param part_labels      (np.ndarray[str])   (num_pc, n)
        @param se3              (np.ndarray)        (num_pc, n, 3, 3)
        @param shuffle_points   (bool)              (default = True) shuffle points using the random seed provided during instantiation          

        @return None
        '''

        # Jitter points
        observations = self._jitter_observation( observations )

        # Shuffle points in point cloud
        if( shuffle_points ):
            indices = np.arange( 0, observations.shape[1] )
            # Loop through to randomly shuffle each point cloud separately
            for i in range( observations.shape[0] ):
                np.random.shuffle( indices )
                observations[i] = observations[i][indices]
                class_labels[i] = class_labels[i][indices]
                if( len( part_labels[i] ) > 0 ): part_labels[i] = part_labels[i][indices]
                if( len( se3[i] ) > 0 ): se3[i] = se3[i][indices]

        # Shuffle frames
        indices = np.arange( 0, observations.shape[0] )
        np.random.shuffle( indices )
        frame_id = frame_id[indices]
        observations = observations[indices]
        class_labels = class_labels[indices]
        part_labels = part_labels[indices]
        se3 = se3[indices]
        
        splits = [( 0, int( np.ceil( observations.shape[0] * self._test_amt ) ) ),
                  ( int( np.ceil( observations.shape[0] * self._test_amt ) ), int( np.ceil( observations.shape[0] * self._test_amt ) ) + int( np.ceil( observations.shape[0] * self._val_amt ) ) ),
                  ( int( np.ceil( observations.shape[0] * self._test_amt ) ) + int( np.ceil( observations.shape[0] * self._val_amt ) ), observations.shape[0] )]

        self._data_size['train'] += splits[2][1] - splits[2][0]
        self._data_size['val'] += splits[1][1] - splits[1][0]
        self._data_size['test'] += splits[0][1] - splits[0][0]

        train = self._get_new_subset()
        val = self._get_new_subset()
        test = self._get_new_subset()

        for i in range( splits[0][0], splits[0][1] ):
            test['frame_id'].append( frame_id[i] )
            test['observations'].append( observations[i] )
            test['class_labels'].append( class_labels[i] )
            test['part_labels'].append( part_labels[i] )
            test['se3'].append( se3[i] )

        for i in range( splits[1][0], splits[1][1] ):
            val['frame_id'].append( frame_id[i] )
            val['observations'].append( observations[i] )
            val['class_labels'].append( class_labels[i] )
            val['part_labels'].append( part_labels[i] )
            val['se3'].append( se3[i] )

        for i in range( splits[2][0], splits[2][1] ):
            train['frame_id'].append( frame_id[i] )
            train['observations'].append( observations[i] )
            train['class_labels'].append( class_labels[i] )
            train['part_labels'].append( part_labels[i] )
            train['se3'].append( se3[i] )

        self._cache_data( train, val, test )

        self.save()

    def get_train_set( self ):

        def generator():
            for i in range( len( self._train['observations'] ) ):
                x = np.array( self._train['observations'][i] )
                cls = self._train['class_labels'][i][0] == np.array( self._class_labels ) if self._one_hot else self._train['class_labels'][i][0]
                prt = np.array( [ i == np.array( self._part_labels ) for i in self._train['part_labels'][i] ] ) if self._one_hot else self._train['part_labels'][i]
                so3 = np.array( self._train['se3'][i][0] )[:3, :3]

                y = {
                    'classification_output': cls,
                    'segmentation_output': prt,
                    'se3': so3
                }

                yield x, y

        if( self._one_hot ):
            cls_spec = tf.TensorSpec( shape = ( len( self._class_labels ) ), dtype = tf.float32 )
            seg_spec = tf.TensorSpec( shape = ( self._network_input_width, len( self._part_labels ) ), dtype = tf.float32 )
        else:
            cls_spec = tf.TensorSpec( shape = ( ), dtype = tf.int32 )      # Scalar integer
            seg_spec = tf.TensorSpec( shape = ( self._network_input_width, ), dtype = tf.int32 ) # Vector of integers (points,)

        output_signature = (
            tf.TensorSpec( shape = ( self._network_input_width, 3 ), dtype = tf.float32 ),
            {
                'classification_output':    cls_spec,
                'segmentation_output':      seg_spec,
                'se3':                      tf.TensorSpec( shape = ( 3, 3 ), dtype = tf.float32 )
            }
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature = output_signature
        )

        return dataset.batch( self._batch_size ).prefetch( tf.data.AUTOTUNE )

    def get_val_set( self ):

        def generator( ):
            for i in range( len( self._val['observations'] ) ):
                x = np.array( self._val['observations'][i] )
                cls = self._val['class_labels'][i][0] == np.array( self._class_labels ) if self._one_hot else self._val['class_labels'][i][0]
                prt = np.array( [ i == np.array( self._part_labels ) for i in self._val['part_labels'][i] ] ) if self._one_hot else self._val['part_labels'][i]
                so3 = np.array( self._val['se3'][i][0] )[:3, :3]

                y = {
                    'classification_output': cls,
                    'segmentation_output': prt,
                    'se3': so3
                }

                yield x, y

        if( self._one_hot ):
            cls_spec = tf.TensorSpec( shape = ( len( self._class_labels ) ), dtype = tf.float32 )
            seg_spec = tf.TensorSpec( shape = ( self._network_input_width, len( self._part_labels ) ), dtype = tf.float32 )
        else:
            cls_spec = tf.TensorSpec( shape = ( ), dtype = tf.int32 )      # Scalar integer
            seg_spec = tf.TensorSpec( shape = ( self._network_input_width, ), dtype = tf.int32 ) # Vector of integers (points,)

        output_signature = (
            tf.TensorSpec( shape = ( self._network_input_width, 3 ), dtype = tf.float32 ),
            {
                'classification_output':    cls_spec,
                'segmentation_output':      seg_spec,
                'se3':                      tf.TensorSpec( shape = ( 3, 3 ), dtype = tf.float32 )
            }
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature = output_signature
        )

        return dataset.batch( self._batch_size ).prefetch( tf.data.AUTOTUNE )
    
    def get_test_set( self ):

        def generator( ):
            for i in range( len( self._test['observations'] ) ):
                x = np.array( self._test['observations'][i] )
                cls = self._test['class_labels'][i][0] == np.array( self._class_labels ) if self._one_hot else self._test['class_labels'][i][0]
                prt = np.array( [ i == np.array( self._part_labels ) for i in self._test['part_labels'][i] ] ) if self._one_hot else self._test['part_labels'][i]
                so3 = np.array( self._train['se3'][i][0] )[:3, :3]

                y = {
                    'classification_output': cls,
                    'segmentation_output': prt,
                    'se3': so3
                }

                yield x, y

        if( self._one_hot ):
            cls_spec = tf.TensorSpec( shape = ( len( self._class_labels ) ), dtype = tf.float32 )
            seg_spec = tf.TensorSpec( shape = ( self._network_input_width, len( self._part_labels ) ), dtype = tf.float32 )
        else:
            cls_spec = tf.TensorSpec( shape = ( ), dtype = tf.int32 )      # Scalar integer
            seg_spec = tf.TensorSpec( shape = ( self._network_input_width, ), dtype = tf.int32 ) # Vector of integers (points,)

        output_signature = (
            tf.TensorSpec( shape = ( self._network_input_width, 3 ), dtype = tf.float32 ),
            {
                'classification_output':    cls_spec,
                'segmentation_output':      seg_spec,
                'se3':                      tf.TensorSpec( shape = ( 3, 3 ), dtype = tf.float32 )
            }
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature = output_signature
        )

        return dataset.batch( self._batch_size ).prefetch( tf.data.AUTOTUNE )
    
    def get_class_label_with_confidence( self, one_hot_vector: np.ndarray ):
        labels = []
        for y_pred in one_hot_vector:
            labels.append( ( self._class_labels[np.argmax( y_pred )], y_pred[np.argmax( y_pred )]))

        return labels
    
    def get_part_label_with_confidence( self, one_hot_vector: np.ndarray ):
        labels = []
        for y_pred in one_hot_vector:
            labels.append( ( self._part_labels[np.argmax( y_pred )], y_pred[np.argmax( y_pred )]))

        return labels

    def get_description(self):
        return self._description
    
    def get_info(self):
        out = f'{self._description}\n'
        out += f'Is one-hot encoded: {self._one_hot}\n'
        out += f'Random seed: {self._random_seed}\n' if (type(self._random_seed) == int) else f'Is not seeded\n'
        out += f'Class labels: {self._class_labels}\n'
        out += f'Part labels: {self._part_labels}\n'

        out += f'\n--- Train Set ---\n'
        out += f'Specified proportion:  {self._train_amt}\n'
        out += f"Actual proportion: {self._data_size['train'] / (self._data_size['train'] + self._data_size['val'] + self._data_size['test'])}\n"
        out += f"Total count: {self._data_size['train']}\n"
        out += f'Class count:\n'
        # for label in self._class_labels:
        #     out += f"\t{label}: {np.count_nonzero(np.array(self._train['class_labels']) == label)}\n"
        out += f'Part count:\n'
        # for label in self._part_labels:
        #     out += f"\t{label}: {np.count_nonzero(np.array(self._train['part_labels']) == label)}\n"

        out += f'\n--- Validation Set ---\n'
        out += f'Specified proportion:  {self._val_amt}\n'
        out += f"Actual proportion: {self._data_size['val'] / (self._data_size['train'] + self._data_size['val'] + self._data_size['test'])}\n"
        out += f"Total count: {self._data_size['val']}\n"
        out += f'Class count:\n'
        # for label in self._class_labels:
        #     out += f"\t{label}: {np.count_nonzero(np.array(self._val['class_labels']) == label)}\n"
        out += f'Part count:\n'
        # for label in self._part_labels:
        #     out += f"\t{label}: {np.count_nonzero(np.array(self._val['part_labels']) == label)}\n"

        out += f'\n--- Test Set ---\n'
        out += f'Specified proportion:  {self._test_amt}\n'
        out += f"Actual proportion: {self._data_size['test'] / (self._data_size['train'] + self._data_size['val'] + self._data_size['test'])}\n"
        out += f"Total count: {self._data_size['test']}\n"
        out += f'Class count:\n'
        # for label in self._class_labels:
        #     out += f"\t{label}: {np.count_nonzero(np.array(self._test['class_labels']) == label)}\n"
        out += f'Part count:\n'
        # for label in self._part_labels:
        #     out += f"\t{label}: {np.count_nonzero(np.array(self._test['part_labels']) == label)}\n"

        return out
    
    def _cache_data( self, train, val, test ) -> None:

        if( len( train['observations'] ) > 0 ):
            cache_path = f"pointcloud/cache/pc_subset_{time.strftime('%Y_%m_%D_%H:%M:%S')}"

            while( os.path.exists( cache_path ) ):
                cache_path = f'_{cache_path}'

            try:
                with open( cache_path, 'wr' ) as pf:
                    ss = PointCloudSubset( self.uid, train, val, test )
                    pickle.dump( ss, pf )

                    del ss
                    gc.collect()

            except Exception as e:
                self._print( f'Unable to cache dataset:\n{type(e).__name__}:  {e}' )
    
    def _get_new_subset( self ):
        return {'frame_id': [], 'observations': [], 'class_labels': [], 'part_labels': [], 'se3': []}
    
    def _one_hot_encode_class_labels(self, labels: list):
        '''
        @param labels   (list) shape (samples, input width)

        @return (np.ndarray) shape (samples, input_width, num_labels)
        '''
        samples = []
        for sample in labels:
            labels = []
            for pt in sample:
                labels.append(np.array( pt ) == np.array( self._class_labels ))
            samples.append( labels )

        return np.array( samples )
    
    def _one_hot_encode_part_labels(self, labels: list):
        '''
        @param labels   (list) shape (samples, input width)

        @return (np.ndarray) shape (samples, input_width, num_labels)
        '''
        samples = []
        for sample in labels:
            labels = []
            for pt in sample:
                labels.append(np.array( pt ) == np.array( self._part_labels ))
            samples.append( labels )

        return np.array( samples )
    
    def _adjust_to_input_width( self, observations: np.ndarray, class_labels: np.ndarray, part_labels: np.ndarray, se3: np.ndarray ) -> tuple:
        '''
        Adjusts the input parameters to a uniform arrays of length _network_input_width by either splicing the first 
        _network_input_width samples from the oversized array, or appending a uniform sampling of exiting points. This
        method ensures that points remain aligned with their label when duplicated

        @param observations (np.ndarray) (n,3)
        @param class_labels (np.ndarray) (n,)
        @param part_labels  (np.ndarray) (n,)
        @param rotation     (np.ndarray) (n,3,3)

        @return (observations (np.ndarray), class_labels(np.ndarray), part_labels(np.ndarray), se3(np.ndarray))
        '''

        if( observations.shape[0] > self._network_input_width ):
            return observations[:self._network_input_width], class_labels[:self._network_input_width], part_labels[:self._network_input_width], se3[:self._network_input_width]
        
        else:
            repeated_indices = np.random.uniform( 0, observations.shape[0], self._network_input_width - observations.shape[0] )
            repeated_indices = repeated_indices.astype(np.int_)

            additional_obs = deepcopy( observations[repeated_indices] )
            observations = np.concatenate( ( observations, additional_obs ), axis = 0 )
            assert observations.shape[0] == self._network_input_width, f'Failed to adjust observations to the network input width - should be {self._network_input_width}, not {observations.shape[0]}'

            if( len( class_labels ) > 0 ):
                additional_cl = deepcopy( class_labels[repeated_indices] )
                class_labels = np.concatenate( ( class_labels, additional_cl ), axis = 0 )
                assert class_labels.shape[0] == self._network_input_width, f'Failed to adjust class_labels to the network input width - should be {self._network_input_width}, not {class_labels.shape[0]}'

            if( len( part_labels ) > 0 ):
                additional_pl = deepcopy( part_labels[repeated_indices] )
                part_labels = np.concatenate( ( part_labels, additional_pl ), axis = 0 )
                assert part_labels.shape[0] == self._network_input_width, f'Failed to adjust part_labels to the network input width - should be {self._network_input_width}, not {part_labels.shape[0]}'

            if( len( se3 ) > 0 ):
                additional_se3 = deepcopy( se3[repeated_indices] )
                se3 = np.concatenate( ( se3, additional_se3 ), axis = 0 )
                assert se3.shape[0] == self._network_input_width, f'Failed to adjust rotations to the network input width - should be {self._network_input_width}, not {se3.shape[0]}'

        return observations, class_labels, part_labels, se3
    
    def _jitter_observation( self, obs: np.ndarray ) -> np.ndarray:
        '''
        Apply Gaussian noise to the points in obs. The distribution can be uniquely defined in each axis via
        the initialization parameter jitter_stdev_m, which defines the standard deviation (in meters) along
        each axis.

        @param obs      (np.ndarray) (num_pc, n, 3)

        @return obs + noise (np.ndarray) (num_pc, n, 3)
        '''

        rng = np.random.default_rng()
        x_noise = rng.normal( loc = 0, scale = self._jitter_stdev_m[0], size = (obs.shape[0], obs.shape[1], 1) )
        y_noise = rng.normal( loc = 0, scale = self._jitter_stdev_m[1], size = (obs.shape[0], obs.shape[1], 1) )
        z_noise = rng.normal( loc = 0, scale = self._jitter_stdev_m[2], size = (obs.shape[0], obs.shape[1], 1) )
        noise = np.concatenate([x_noise, y_noise, z_noise], axis = -1)
        return obs + noise
    
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
    
class PointCloudSubset:
    def __init__( self, belongs_to:  uuid.UUID, train: dict, val: dict, test: dict ):

        self._belongs_to = belongs_to

        self._ss = {'train': train, 'val': val, 'test': test}
        self._idx = {'train': 0, 'val': 0, 'test': 0}

    def load_full( self, belongs_to: uuid.UUID, print_func: Callable[[str], None] = print ):

        if( belongs_to == self._belongs_to ):   return self._ss
        else:
            print_func( f"PointCloudSubset requested does not belong to this PointCloudSet." )
            return None
        
    def load_amount( self, belongs_to: uuid.UUID, num_samples: int, subsubset: str, print_func: Callable[[str], None] = print ):
        '''
        Load specified number of samples from a specific sub-subset ('train', 'val', or 'test')
        '''

        if( subsubset not in list( self._ss.keys() ) ):
            print_func( f"subsubset must be either 'train', 'val', or 'test', not {subsubset}" )
            return None

        if( belongs_to != self._belongs_to ):
            print_func( f"PointCloudSubset requested does not belong to this PointCloudSet." )
            return None

        if( len( self._ss[subsubset] ) - self._idx[subsubset] < num_samples ):
            out = self._ss[subsubset][self._idx[subsubset] : self._idx[subsubset] + num_samples]
            self._idx[subsubset] += num_samples
            return out
        
        else:
            print_func( f"num_samples exceeds the number of samples left in PointCloudSubset, returning {len( self._ss[subsubset] ) - self._idx[subsubset]} samples." )
            
            out = self._ss[subsubset][self._idx :]
            self._idx[subsubset] = 0
            return out

### FREE HELPER FUNCTIONS ###
def load_from_file( pickle_file: str ) -> PointCloudSet:
    
    # Add file extension if not present
    if( pickle_file.split(".")[-1] != '.pkl' ):
        pickle_file += '.pkl'

    with open( pickle_file, 'rb' ) as pf:
        pc_set: PointCloudSet = pickle.load( pf )

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

if __name__ == "__main__":

    MODEL_PATH = 'models/'
    MESH_PATH = 'mesh/'
    FIGURE_PATH = 'figures/'
    DATA_PATH = 'data/'
    PALINDROME_DATA_PATH = '/mnt/d/repos/aburn/usr/hub/palindrome_playground/DataCollect/'
    INPUT_SIZE = 4096
    RANDOM_SEED = 42

    class_labels = ["a-10", "airship", "b-1b", "b-2", "c-5", "c-12", "c-17a", "c-32", "c-130j", "drogue", "e-3", "f-15e", "f-16", "f-18e", "f-22", "g-iii", "kc-46",
                    "kc-135", "lj-25", "lj-25_nosecone", "mig-29", "mq-20", "sr-71", "su-27", "vc-25a", "x-47b" ]
    part_labels = ['fuselage', 'left_engine', 'right_engine', 'left_wing', 'right_wing', 'left_hstab', 'right_hstab', 'vstab', 'left_boom_stab', 'right_boom_stab', 
                   'boom_wing', 'boom_hull', 'boom_hose']
    pc = PointCloudSet(one_hot = True, 
                       class_labels = class_labels, 
                       part_labels = part_labels, 
                       network_input_width = INPUT_SIZE, 
                       rand_seed = RANDOM_SEED, 
                       jitter_stdev_m = np.array( [ 0.1, 0.1, 0.1 ] ))
    
    pc.add_from_aftr_output(f'{DATA_PATH}collect_2025.Nov.24_15.47.57.5524625.UTC')
    pc.get_info()
    pc.add_from_aftr_output(f'{DATA_PATH}collect_2025.Nov.24_15.55.18.9256538.UTC')
    pc.get_info()
    pc.add_from_aftr_output(f'{DATA_PATH}collect_2025.Nov.25_13.12.04.1835786.UTC')
    pc.get_info()