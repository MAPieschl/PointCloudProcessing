'''
This module outputs samplings of a given mesh.

--------------------

PointNet.py

By:     Mike Pieschl
Date:   30 July 2025
'''

import copy
import open3d as o3d
import numpy as np
import plotly.graph_objects as go

import mat_ops

class MeshSampler:
    def __init__(self, 
                 mesh_path: str, 
                 mesh_label: str, 
                 rotation_matrix: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                 center_point: np.ndarray = np.array([0, 0, 0]),
                 scale: float = 1.0,
                 random_seed: int = 42):
        '''
        Creates a MeshSampler object.

        @param  mesh_path       (str) path and file name for mesh to be loaded (ex. 'mesh/object.obj')
        @param  mesh_label      (str) classification label
        @param  rotation_matrix (ndarray) 3x3 DCM to transform the object into the desired global orientation
        @param  center_point    (ndarray) (x, y, z) of the center of the loaded mesh
        @param  random_seed     (int) seed for all random functions

        Methods:
            MeshSampler.show_mesh( title: str ) >> outputs a 3D plotly of mesh
            MeshSampler.display_point_clouds( self, clouds: list, labels: list, title: str = 'Point Cloud' ) >> outputs 3D plotly of color-coded 3D point clouds
            MeshSampler.create_viewpoint_observations( self,
                                                        n: int, 
                                                        p: int, 
                                                        dist_range: tuple = (5.0, 10.0), 
                                                        az_range: tuple = (0.0, 359.0),
                                                        elev_range: tuple = (-5.0, 20.0),
                                                        camera_rot: float = 0.0,
                                                        sample_mult: int = 300 ) -> np.array(observations), np.array(labels), np.array(view_point)
        '''

        self.ADVISORY_HEADER = 'MeshSampler:  '

        self.path = mesh_path
        self._R = rotation_matrix
        self._p = center_point
        self.mesh = self._init_mesh( scale ) if self.path != None else None
        self.label = mesh_label

        self._seed = random_seed
        
        self._rotate_and_center_3d_mesh()
    
    def show_mesh( self, title: str = 'Mesh' ):
        '''
        Displays a mesh object output from open3d.io.read_traingle_mesh()
        
        @param title     (str)  plot title   
        '''

        if( self.mesh != None ):

            vertices = np.asarray(self.mesh.vertices)
            faces = np.asarray(self.mesh.triangles)

            fig = go.Figure(data = [go.Mesh3d(
                x = vertices[:, 0],
                y = vertices[:, 1],
                z = vertices[:, 2],
                i = faces[:, 0],
                j = faces[:, 1],
                k = faces[:, 2],
                color = 'lightblue',
                opacity = 1.0
            )])
            
            fig.update_layout(scene = dict(aspectmode = 'data'))
            fig.show()

        else:
            print(f'{self.ADVISORY_HEADER}No mesh loaded.')

    def display_point_clouds( self, clouds: list, labels: list, title: str = 'Point Cloud' ):
        '''
        Displays a point cloud as a 3D scatter plot

        @param clouds    (list) of ndarrays     -> [cloud][points]
        @param labels    (list) parallel array  -> [cloud]
        @param title     (str)  plot title   
        '''

        assert len(clouds) == len(labels), "display_point_clouds:  ensure there is a label for each cloud"

        plots = []
        for i, cloud in enumerate(clouds):
            
            plots.append( go.Scatter3d(
                x = cloud[:, 0],
                y = cloud[:, 1],
                z = cloud[:, 2],
                mode = 'markers',
                marker = dict(
                    size = 2,
                    opacity = 1.0
                ),
                name = labels[i]
            ))

        fig = go.Figure(data = plots)

        fig.update_layout(scene = dict(
            xaxis_title = 'X',
            yaxis_title = 'Y',
            zaxis_title = 'Z',
            aspectmode = 'data'
        ),
        title = title,
        margin = dict( l = 0, r = 0, b = 0, t = 40 )
        )

        fig.show()

    def show_scene( self, cloud, title: str = 'Capture Scene' ):
        '''
        Displays the mesh object output from open3d.io.read_triangle_mesh(), along with the point clouds provided
        
        @param cloud     (pd.DataFrame) output from data_exploration.structure_dataframes()
        @param title     (str)  plot title    
        '''

        assert self.mesh != None, "show_scene:  no mesh loaded"

        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.triangles)

        # Add mesh
        fig = go.Figure(data = [
            go.Mesh3d(
                x = vertices[:, 0],
                y = vertices[:, 1],
                z = vertices[:, 2],
                i = faces[:, 0],
                j = faces[:, 1],
                k = faces[:, 2],
                color = 'lightblue',
                opacity = 1.0
            ),
            go.Scatter3d(
                x = cloud['xg'],
                y = cloud['yg'],
                z = cloud['zg'],
                mode = 'markers',
                marker = dict(
                    size = 2,
                    color = cloud['strength'],
                    colorscale = 'Viridis',
                    colorbar = dict(title = 'Magnitude')
                ),
                name = 'Returns'
            ),
            go.Scatter3d(
                x = cloud['cam_xg'],
                y = cloud['cam_yg'],
                z = cloud['cam_zg'],
                mode = 'markers',
                marker = dict(
                    size = 2,
                    color = 'red'
                ),
                name = 'Camera'
            )
        ])

        fig.update_layout(scene = dict(
            xaxis_title = 'X',
            yaxis_title = 'Y',
            zaxis_title = 'Z',
            aspectmode = 'data'
        ),
        title = title,
        margin = dict( l = 0, r = 0, b = 0, t = 40 )
        )
        
        fig.show()

    def create_full_sample_observations( self,
                                         n: int, 
                                         p: int, 
                                         pad: int,
                                         dist_range: tuple = (5.0, 10.0), 
                                         az_range: tuple = (0.0, 359.0),
                                         elev_range: tuple = (-5.0, 20.0),
                                         camera_rot: float = 0.0,
                                         reproject: bool = False ):
        '''
        Returns parallel tensors of point cloud samples uniformly distributed about the mesh

        @param obj_classes  (dict) with the structure [mesh_ID][mesh_object]
        @param n            (int) number of viewpoints (observations) to acquire for each class
        @param p            (int) number of points to acquire for each observation
        @param pad          (int) size of output data set (if smaller, observations will be zero-padded)
        @param dist_range   (tuple(float, float)) (mesh units) min and max range from the origin point from which to sample
        @param az_range     (tuple(float, float)) (deg) about the z-axis to observe
        @param elev_range   (tuple(float, float)) (deg) where 0 = xy-plane, 90 = +z axis, -90 = -z axis
        @param camera_rot   (float) number of degrees CW and CCW to randomly rotate camera during sampling (CURRENTLY NOT IMPLEMENTED)
        @param rand_seed    (int) used in random camera position and random point downsampling
        @param reproject    (bool) project points in camera space (CURRENTLY NOT IMPLEMENTED)

        @returns ( observations, labels, view_point )   observations    - (ndarray) shape ( num_classes * n, p, 3 )
                                                        labels          - (ndarray) shape ( num_classes * n, )
                                                        position        - (ndarray) shape ( 3 ) - object_pos if reproject else camera_pos
                                                        (parallel arrays)
        '''

        observations = []
        labels = []
        position = []
        dcm = []
        
        if( self.mesh != None ):

            # Sample the mesh densely to ensure enough visible points for all aspects
            points = self.mesh.sample_points_uniformly( number_of_points = p )
            
            # Iterate through random camera positions
            gen = np.random.default_rng( seed = self._seed )
            viewangles = np.array( [ gen.uniform( low = dist_range[0], high = dist_range[1] , size = n ), gen.uniform( low = az_range[0], high = az_range[1], size = n ), gen.uniform( low = elev_range[0], high = elev_range[1], size = n ) ] )
            viewpoints = np.array( [ viewangles[0] * np.cos(np.deg2rad(viewangles[1])) * np.sin(np.deg2rad(90 - viewangles[2])), viewangles[0] * np.sin(np.deg2rad(viewangles[1])) * np.sin(np.deg2rad(90 - viewangles[2])), viewangles[0] * np.cos(np.deg2rad(90 - viewangles[2])) ] ).T
            camera_roll = np.array( gen.uniform( low = -camera_rot, high = camera_rot, size = n ) )
            
            # Remove hidden points from camera view, shuffle the points, and append a subset of length p to output
            for i, vp in enumerate(viewpoints):
                c_R_w = mat_ops.get_DCM_positive_x_pointing_at_origin(vp, camera_roll[i])
                in_sight = np.asarray( copy.deepcopy(points.points) )
                in_sight = (c_R_w @ (in_sight.T - vp.reshape((3, 1)))).T if reproject else in_sight
                observations.append(np.array(pad_observation(pad, list(in_sight))))
                labels.append(self.label)
                position.append(np.array([0, 0, 0]) if reproject else vp)
                dcm.append(c_R_w)
                
        else:
            print(f'{self.ADVISORY_HEADER}No mesh loaded.')

        return np.array(observations), np.array(labels), np.array(position), np.array(dcm)

    def create_viewpoint_observations( self,
                                      n: int, 
                                      p: int, 
                                      pad: int,
                                      dist_range: tuple = (5.0, 10.0), 
                                      az_range: tuple = (0.0, 359.0),
                                      elev_range: tuple = (-5.0, 20.0),
                                      camera_rot: float = 0.0,
                                      reproject: bool = False ):
        '''
        Returns parallel tensors of point cloud samples randomly selected from a uniform distribution of points within sight of the camera view

        @param obj_classes  (dict) with the structure [mesh_ID][mesh_object]
        @param n            (int) number of viewpoints (observations) to acquire for each class
        @param p            (int) number of points to acquire for each observation
        @param dist_range   (tuple(float, float)) (mesh units) min and max range from the origin point from which to sample
        @param az_range     (tuple(float, float)) (deg) about the z-axis to observe
        @param elev_range   (tuple(float, float)) (deg) where 0 = xy-plane, 90 = +z axis, -90 = -z axis
        @param camera_rot   (float) number of degrees CW and CCW to randomly rotate camera during sampling (CURRENTLY NOT IMPLEMENTED)
        @param rand_seed    (int) used in random camera position and random point downsampling
        @param reproject    (bool) project points in camera space (CURRENTLY NOT IMPLEMENTED)

        @returns ( observations, labels, view_point )   observations    - (ndarray) shape ( num_classes * n, p, 3 )
                                                        labels          - (ndarray) shape ( num_classes * n, )
                                                        view_point      - (ndarray) shape ( 3 ) - object_pos if reproject else camera_pos
                                                        (parallel arrays)
        '''

        observations = []
        labels = []
        position = []
        dcm = []
        
        if( self.mesh != None ):

            # Sample the mesh densely to ensure enough visible points for all aspects
            points = self.mesh.sample_points_uniformly( number_of_points = p )
            
            # Iterate through random camera positions
            gen = np.random.default_rng( seed = self._seed )
            viewangles = np.array( [ gen.uniform( low = dist_range[0], high = dist_range[1] , size = n ), gen.uniform( low = az_range[0], high = az_range[1], size = n ), gen.uniform( low = elev_range[0], high = elev_range[1], size = n ) ] )
            viewpoints = np.array( [ viewangles[0] * np.cos(np.deg2rad(viewangles[1])) * np.sin(np.deg2rad(90 - viewangles[2])), viewangles[0] * np.sin(np.deg2rad(viewangles[1])) * np.sin(np.deg2rad(90 - viewangles[2])), viewangles[0] * np.cos(np.deg2rad(90 - viewangles[2])) ] ).T
            camera_roll = np.array( gen.uniform( low = -camera_rot, high = camera_rot, size = n ) )

            # Remove hidden points from camera view, shuffle the points, and append a subset of length p to output
            for i, vp in enumerate(viewpoints):
                c_R_w = mat_ops.get_DCM_positive_x_pointing_at_origin(vp, camera_roll[i])
                _, i = copy.deepcopy(points.hidden_point_removal(-vp, 1))
                in_sight = np.asarray(points.points)[i]
                in_sight = (c_R_w @ (in_sight.T - vp.reshape((3, 1)))).T if reproject else in_sight
                gen.shuffle(in_sight)
                observations.append(np.array(pad_observation(pad, list(in_sight))))
                labels.append(self.label)
                position.append(np.array([0, 0, 0]) if reproject else vp)
                dcm.append(c_R_w)
                
        else:
            print(f'{self.ADVISORY_HEADER}No mesh loaded.')

        return np.array(observations), np.array(labels), np.array(position), np.array(dcm)

    def _init_mesh( self, scale ):
        '''
        (private) Initialize mesh
        '''

        mesh = o3d.io.read_triangle_mesh( self.path )
        mesh.scale(scale, np.array([0, 0, 0]))
        mesh.compute_vertex_normals()

        return mesh
    
    def _rotate_and_center_3d_mesh( self ):
        '''
        Rotates the mesh via the defined R_KEY (contained in obj_classes) about the origin and removes the key/value
        pair from the dict. Must follow load_3D_mesh.

        @param obj_classes  (dict) with the structure [mesh_ID][mesh_object]
                                                    [mesh_ID][rotation_matrix]
        '''
        
        if( self.mesh != None ):
            
            self.mesh.rotate( self._R, self._p )
            self.mesh.translate( -self._p )
            
        else:
            print(f'{self.ADVISORY_HEADER}No mesh loaded.')

def pad_observation(n: int, observation: list):

    if(n <= len(observation)):  return observation[:n]

    for i in range(0, n - len(observation)):
        observation.append(observation[i])
    
    return observation