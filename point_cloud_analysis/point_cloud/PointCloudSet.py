'''
Organizes point cloud data for training.

--------------------

PointNet.py

By:     Mike Pieschl
Date:   31 July 2025
'''

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from copy import deepcopy

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
                 description: str = '',):
        
        self._description = description
        self._batch_size = batch_size
        self._one_hot = one_hot
        self._class_labels = class_labels
        self._part_labels = part_labels
        self._network_input_width = network_input_width
        self._jitter_stdev_m: np.ndarray = jitter_stdev_m

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
            print('PointCloudSet:  train_val_test_split incorrect format - set to default 75% / 15% / 10%')

        self._train = {'frame_id': [], 'observations': [], 'class_labels': [], 'part_labels': []}
        self._val = {'frame_id': [], 'observations': [], 'class_labels': [], 'part_labels': []}
        self._test = {'frame_id': [], 'observations': [], 'class_labels': [], 'part_labels': []}

    def add_from_aftr_output(self, dir_path: str, class_label: str, shuffle_points: bool = True):
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

        frame_id = []
        observations = []
        class_labels = []
        part_labels = []

        frames_searched = 0
        non_num_found = 0
        
        collect_contents = get_dir_contents(dir_path)
        lidar_contents = get_dir_contents(f'{dir_path}/Virtual Flash Lidar')

        print(f'Parsing frames in {dir_path}...')
        for i in tqdm(range(len(lidar_contents))):
            try:
                with open(f'{dir_path}/Virtual Flash Lidar/frame_{i}.txt', 'r') as f:
                    obs = []
                    label = []
                    for j, line in enumerate(f):
                        line = line.strip().replace(" ", "")
                        pos_start_idx = line.find('(')
                        pos_end_idx = line.find(')')

                        pos_str = line[pos_start_idx + 1:pos_end_idx].split(',')
                        pos = []
                        for val in pos_str:
                            pos.append(float(val))

                        if( np.isfinite( np.array( pos ) ).all() ):
                            obs.append(np.array(pos))
                            label.append(line[pos_end_idx + 1:])
                        else:
                            non_num_found += 1

                    if(len(obs) != 0):
                        obs, label = self._adjust_to_input_width(np.array(obs), np.array(label))

                        if( np.isfinite( obs ).all() ):
                            frame_id.append(i)
                            observations.append(obs)
                            class_labels.append(class_label)
                            part_labels.append(label)
                        else:
                            print( f'Per-line check failed - frame_{i} discarded after detecting non-finite value.' )

            except:
                if(frames_searched == 0): frames_searched = i

        self.add_data(np.array(frame_id), np.array(observations), np.array(class_labels), np.array(part_labels), shuffle_points)
            
        print(f'{dir_path} parsed:  found {len(frame_id)} valid frames out of {frames_searched} total. {non_num_found} total lines discarded for non-numeric values.')

    def add_data(self, frame_id: np.ndarray, observations: np.ndarray, class_labels: np.ndarray, part_labels: np.ndarray, shuffle_points: bool = True):

        error_string = f'One or more of the inputs has incorrect shape:\n\tframe_id.shape = {frame_id.shape}\n\tobservations.shape = {observations.shape}\n\tclass_labels.shape = {class_labels.shape}\n\tpart_labels.shape = {part_labels.shape}'
        assert frame_id.shape == class_labels.shape, error_string
        assert observations.shape[1] == part_labels.shape[1], error_string
        assert frame_id.shape[0] == observations.shape[0], error_string
        assert frame_id.shape[0] == class_labels.shape[0], error_string
        assert frame_id.shape[0] == part_labels.shape[0], error_string

        # Jitter points
        observations = self._jitter_observation( observations )

        # Shuffle points in point cloud
        if(shuffle_points):
            indices = np.arange(0, observations.shape[1])
            # Loop through to randomly shuffle each point cloud separately
            for i in range(observations.shape[0]):
                np.random.shuffle(indices)
                observations[i] = observations[i][indices]
                part_labels[i] = part_labels[i][indices]

        # Shuffle frames
        indices = np.arange(0, observations.shape[0])
        np.random.shuffle(indices)
        frame_id = frame_id[indices]
        observations = observations[indices]
        class_labels = class_labels[indices]
        part_labels = part_labels[indices]
        
        if(not (frame_id.shape[0] == observations.shape[0] and frame_id.shape[0] == class_labels.shape[0] and frame_id.shape[0] == part_labels.shape[0])):
            print(f"Number of observations must be equal to the number of labels and number of view_points. Point clouds discarded.")
            return
        
        splits = [(0, int(np.ceil(observations.shape[0] * self._test_amt))),
                  (int(np.ceil(observations.shape[0] * self._test_amt)), int(np.ceil(observations.shape[0] * self._test_amt)) + int(np.ceil(observations.shape[0] * self._val_amt))),
                  (int(np.ceil(observations.shape[0] * self._test_amt)) + int(np.ceil(observations.shape[0] * self._val_amt)), observations.shape[0])]

        for i in range(splits[0][0], splits[0][1]):
            self._test['frame_id'].append(frame_id[i])
            self._test['observations'].append(observations[i])
            self._test['class_labels'].append(class_labels[i])
            self._test['part_labels'].append(part_labels[i])

        for i in range(splits[1][0], splits[1][1]):
            self._val['frame_id'].append(frame_id[i])
            self._val['observations'].append(observations[i])
            self._val['class_labels'].append(class_labels[i])
            self._val['part_labels'].append(part_labels[i])

        for i in range(splits[2][0], splits[2][1]):
            self._train['frame_id'].append(frame_id[i])
            self._train['observations'].append(observations[i])
            self._train['class_labels'].append(class_labels[i])
            self._train['part_labels'].append(part_labels[i])
    
    def get_train_class_set(self):
        labels = np.array(self._train['class_labels']) if not self._one_hot else self._one_hot_encode_class_labels(self._train['class_labels'])
        return tf.data.Dataset.from_tensor_slices((np.array(self._train['observations']), labels)).batch(batch_size = self._batch_size)
    
    def get_train_seg_set(self):
        labels = np.array(self._train['part_labels']) if not self._one_hot else self._one_hot_encode_part_labels(self._train['part_labels'])
        print(f"Training data size:  obs = {np.array(self._train['observations']).shape} | labels = {np.array(self._train['part_labels']).shape} ")
        return tf.data.Dataset.from_tensor_slices((np.array(self._train['observations']), labels)).batch(batch_size = self._batch_size)

    def get_train_tnet_set(self):
        return tf.data.Dataset.from_tensor_slices((np.array(self._train['observations']), np.array(self._train['dcm']))).batch(batch_size = self._batch_size)
    
    def get_val_class_set(self):
        labels = np.array(self._val['class_labels']) if not self._one_hot else self._one_hot_encode_class_labels(self._val['class_labels'])
        return tf.data.Dataset.from_tensor_slices((np.array(self._val['observations']), labels)).batch(batch_size = self._batch_size)
    
    def get_val_seg_set(self):
        labels = np.array(self._val['part_labels']) if not self._one_hot else self._one_hot_encode_part_labels(self._val['part_labels'])
        print(f"Validation data size:  obs = {np.array(self._val['observations']).shape} | labels = {np.array(self._val['part_labels']).shape} ")
        return tf.data.Dataset.from_tensor_slices((np.array(self._val['observations']), labels)).batch(batch_size = self._batch_size)
    
    def get_val_tnet_set(self):
        return tf.data.Dataset.from_tensor_slices((np.array(self._val['observations']), np.array(self._val['dcm']))).batch(batch_size = self._batch_size)
    
    def get_random_val_sample(self):
        sample_i = int(len(self._val['observations']) * np.random.uniform())
        sample_obs = self._val['observations'][sample_i]

        return {
            'observation': np.expand_dims(sample_obs, axis = 0), 
            'label': self._val['labels'][sample_i],
            'position': self._val['positions'][sample_i],
            'dcm': self._val['dcm'][sample_i]
        }
    
    def get_raw_val_set(self):
        return self._val
    
    def get_test_class_set(self):
        labels = np.array(self._test['class_labels']) if not self._one_hot else self._one_hot_encode_class_labels(self._test['class_labels'])
        return tf.data.Dataset.from_tensor_slices((np.array(self._test['observations']), labels)).batch(batch_size = self._batch_size)
    
    def get_test_seg_set(self):
        labels = np.array(self._test['part_labels']) if not self._one_hot else self._one_hot_encode_part_labels(self._test['part_labels'])
        return tf.data.Dataset.from_tensor_slices((np.array(self._test['observations']), labels)).batch(batch_size = self._batch_size)
    
    def get_test_tnet_set(self):
        return tf.data.Dataset.from_tensor_slices((np.array(self._test['observations']), np.array(self._test['dcm']))).batch(batch_size = self._batch_size)

    def get_random_test_sample(self):

        sample_i = int(len(self._test['observations']) * np.random.uniform())
        sample_obs = self._test['observations'][sample_i]

        return {
            'observation': np.expand_dims(sample_obs, axis = 0), 
            'label': self._test['labels'][sample_i],
            'position': self._test['positions'][sample_i],
            'dcm': self._test['dcm'][sample_i]
        }
    
    def get_raw_test_set(self):
        return self._test
    
    def get_labels_with_confidence(self, one_hot_vector: np.ndarray):
        labels = []
        for y_pred in one_hot_vector:
            labels.append((self._class_labels[np.argmax(y_pred)], y_pred[np.argmax(y_pred)]))

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
        out += f"Actual proportion: {len(self._train['observations']) / (len(self._train['observations']) + len(self._val['observations']) + len(self._test['observations']))}\n"
        out += f"Total count: {len(self._train['observations'])}\n"
        out += f'Class count:\n'
        for label in self._class_labels:
            out += f"\t{label}: {np.count_nonzero(np.array(self._train['class_labels']) == label)}\n"
        out += f'Part count:\n'
        for label in self._part_labels:
            out += f"\t{label}: {np.count_nonzero(np.array(self._train['part_labels']) == label)}\n"

        out += f'\n--- Validation Set ---\n'
        out += f'Specified proportion:  {self._val_amt}\n'
        out += f"Actual proportion: {len(self._val['observations']) / (len(self._train['observations']) + len(self._val['observations']) + len(self._test['observations']))}\n"
        out += f"Total count: {len(self._val['observations'])}\n"
        out += f'Class count:\n'
        for label in self._class_labels:
            out += f"\t{label}: {np.count_nonzero(np.array(self._val['class_labels']) == label)}\n"
        out += f'Part count:\n'
        for label in self._part_labels:
            out += f"\t{label}: {np.count_nonzero(np.array(self._val['part_labels']) == label)}\n"

        out += f'\n--- Test Set ---\n'
        out += f'Specified proportion:  {self._test_amt}\n'
        out += f"Actual proportion: {len(self._test['observations']) / (len(self._train['observations']) + len(self._val['observations']) + len(self._test['observations']))}\n"
        out += f"Total count: {len(self._test['observations'])}\n"
        out += f'Class count:\n'
        for label in self._class_labels:
            out += f"\t{label}: {np.count_nonzero(np.array(self._test['class_labels']) == label)}\n"
        out += f'Part count:\n'
        for label in self._part_labels:
            out += f"\t{label}: {np.count_nonzero(np.array(self._test['part_labels']) == label)}\n"

        return out
    
    def _one_hot_encode_class_labels(self, labels: list):
        labels_out = []
        for label in self._class_labels:
            labels_out.append(np.array(labels) == label)

        return np.array(labels_out).T
    
    def _one_hot_encode_part_labels(self, labels: list):
        '''
        @param labels   (list) shape (samples, input width)

        @return (np.ndarray) shape (samples, input_width, num_labels)
        '''
        labels_out = []
        for pc in labels:
            curr_pc = []
            for label in pc:
                curr_pc.append(np.array(self._part_labels) == label)
            labels_out.append(curr_pc)

        return np.array(labels_out)
    
    def _adjust_to_input_width(self, observations: np.ndarray, part_labels: np.ndarray) -> tuple:
        '''
        Adjusts the input parameters to a uniform arrays of length _network_input_width by either splicing the first 
        _network_input_width samples from the oversized array, or appending a uniform sampling of exiting points. This
        method ensures that points remain aligned with their label when duplicated

        @param observations (np.ndarray) (n,3)
        @param part_labels  (np.ndarray) (n,)

        @return (observations (np.ndarray), part_labels(np.ndarray))
        '''

        assert observations.shape[0] == part_labels.shape[0], f'The input arrays, observations and part_labels, must have equal length. Currently they are {observations.shape} and {part_labels.shape}, respectively'

        if(observations.shape[0] > self._network_input_width):
            return observations[:self._network_input_width], part_labels[:self._network_input_width]
        
        else:
            repeated_indices = np.random.uniform(0, observations.shape[0], self._network_input_width - observations.shape[0])
            repeated_indices = repeated_indices.astype(np.int_)

            additional_obs = deepcopy(observations[repeated_indices])
            additional_pl = deepcopy(part_labels[repeated_indices])

            observations = np.concatenate((observations, additional_obs), axis = 0)
            part_labels = np.concatenate((part_labels, additional_pl), axis = 0)

        assert observations.shape[0] == self._network_input_width, f'Failed to adjust observations to the network input width - should be {self._network_input_width}, not {observations.shape}'
        assert part_labels.shape[0] == self._network_input_width, f'Failed to adjust part_labels to the network input width - should be {self._network_input_width}, not {part_labels.shape}'

        return observations, part_labels
    
    def _jitter_observation( self, obs: np.ndarray ):
        rng = np.random.default_rng()
        x_noise = rng.normal( loc = 0, scale = self._jitter_stdev_m[0], size = (obs.shape[0], obs.shape[1], 1) )
        y_noise = rng.normal( loc = 0, scale = self._jitter_stdev_m[1], size = (obs.shape[0], obs.shape[1], 1) )
        z_noise = rng.normal( loc = 0, scale = self._jitter_stdev_m[2], size = (obs.shape[0], obs.shape[1], 1) )
        noise = np.concatenate([x_noise, y_noise, z_noise], axis = -1)
        return obs + noise
    
### FREE HELPER FUNCTIONS ###
def get_dir_contents(dir_path: str) -> list:

    try:
        contents = os.listdir(dir_path)
        if(not contents):   return []
        else:               return contents

    except FileNotFoundError:
        print(f"Error: The directory '{dir_path}' was not found.", file = sys.stderr)
    except NotADirectoryError:
        print(f"Error: The path '{dir_path}' is not a directory.", file = sys.stderr)
    except PermissionError:
        print(f"Error: Permission denied to read '{dir_path}'.", file = sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file = sys.stderr)

    return []

if __name__ == "__main__":
    import pickle 

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
    
    pc.add_from_aftr_output(f'{PALINDROME_DATA_PATH}collect_2025.Nov.24_18.52.46.7947129.UTC', 'kc46')
    pc.get_info()
    pc.add_from_aftr_output(f'{PALINDROME_DATA_PATH}collect_2025.Nov.24_19.16.10.8395925.UTC', 'kc46')
    pc.get_info()
