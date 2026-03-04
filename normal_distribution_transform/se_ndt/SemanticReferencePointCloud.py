import numpy as np

from ndt.Voxel import VoxelGrid, Voxel

class SemanticReferencePointCloud:
    def __init__( self, pts: np.ndarray, labels: list[str], voxel_size: float, conditioning_eps: float = 1e-3 ):

        self.__voxel_grid: dict[str, VoxelGrid] = self.__create_semantic_voxel_grids(
            self.__organize_points_by_part( pts, labels ),
            voxel_size,
            conditioning_eps
        )

    def get_voxel_grid_for( self, part: str ) -> VoxelGrid | None:
        if( part in self.__voxel_grid.keys() ): return self.__voxel_grid[part]
        else:                                   return None

    def get_labels( self ) -> list[str]:
        return list( self.__voxel_grid.keys() )

    def get_weighted_8_nearest_voxels( self, pt: np.ndarray, label: str ) -> list[tuple[Voxel | None, float]]:
        if( label in self.__voxel_grid.keys() ):    return self.__voxel_grid[label].get_weighted_8_nearest_voxels( pt )
        else:                                       return [ ( None, 1.0 ) ]
    
    def resize_voxel_grids( self, voxel_size: float ) -> None:
        for lbl, vg in self.__voxel_grid.items():
            vg.build_voxel_grid( voxel_size )

    def get_voxel_size( self ) -> dict[str, float] | float:
        voxel_sizes_labeled = {}
        list_of_voxel_sizes = []
        for lbl, vg in self.__voxel_grid.items():
            voxel_sizes_labeled[lbl] = vg.get_voxel_size()
            list_of_voxel_sizes.append( vg.get_voxel_size() )

        if( np.all( np.array(list_of_voxel_sizes) == list_of_voxel_sizes[0] ) ):
            return float( list_of_voxel_sizes[0] )
        
        else:
            return voxel_sizes_labeled

    def get_list_of_points( self ) -> list[np.ndarray]:
        pts = []
        for lbl, vg in self.__voxel_grid.items():
            pts += vg.get_list_of_points()

        return pts

    def __create_semantic_voxel_grids( self, pts_by_part: dict[str, np.ndarray], voxel_size: float, conditioning_eps: float ):

        vg_by_part: dict[str, VoxelGrid] = {}
        for lbl, pts in pts_by_part.items():
            vg_by_part[lbl] = VoxelGrid( pts, voxel_size, conditioning_eps )

        return vg_by_part

    def __organize_points_by_part( self, pts: np.ndarray, labels: list[str] ) -> dict[str, np.ndarray]:

        pts_by_part: dict[str, np.ndarray] = {}
        part_list = list( np.unique( np.array( labels ) ) )
        for part in part_list:
            pts_by_part[part] = pts[np.where( np.array( labels ) == part )]

        return pts_by_part