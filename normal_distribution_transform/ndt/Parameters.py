import numpy as np
from utils import mat_ops as mat

class Parameters:
    def __init__( self, se3: np.ndarray ):

        assert se3.ndim == 2, f"se3 must be a 4x4 np.ndarray, not ${se3.shape}$"
        assert se3.shape[0] == 4, f"se3 must be a 4x4 np.ndarray, not ${se3.shape}$"
        assert se3.shape[1] == 4, f"se3 must be a 4x4 np.ndarray, not ${se3.shape}$"

        euler_angles: dict[str, float] = mat.get_roll_pitch_yaw_deg( se3 )

        self.se3 = se3

        self.x = se3[0, 3].squeeze()
        self.y = se3[1, 3].squeeze()
        self.z = se3[2, 3].squeeze()
        self.r_x = np.deg2rad( euler_angles['roll'] ).squeeze()
        self.r_y = np.deg2rad( euler_angles['pitch'] ).squeeze()
        self.r_z = np.deg2rad( euler_angles['yaw'] ).squeeze()

        self.cx = np.cos( self.r_x ).squeeze()
        self.sx = np.sin( self.r_x ).squeeze()

        self.cy = np.cos( self.r_y ).squeeze()
        self.sy = np.sin( self.r_y ).squeeze()

        self.cz = np.cos( self.r_z ).squeeze()
        self.sz = np.sin( self.r_z ).squeeze()

        self.vec6 = np.array( [self.x, self.y, self.z, self.r_x, self.r_y, self.r_z] ).reshape(( 6, 1 ))

    def __call__( self ) -> np.ndarray:
        return self.vec6
    
    def get_dcm( self ) -> np.ndarray:
        return mat.get_dcm( float( np.rad2deg( self.r_x ) ), float( np.rad2deg( self.r_y ) ), float( np.rad2deg( self.r_z ) ) )
    
    def get_position( self ) -> np.ndarray:
        return np.array( [self.x, self.y, self.z] ).reshape( ( 3, 1 ) )
    
    def to_string( self ) -> str:
        return f"Position:  ( {float( self.x ):.3f}, {float( self.y ):.3f}, {float( self.z ):.3f} ) | Roll: {float( np.rad2deg( self.r_x ) ):.1f} | Pitch: {float( np.rad2deg( self.r_y ) ):.1f} | Yaw: {float( np.rad2deg( self.r_z ) ):.1f}"
    
    def update( self, vec6: np.ndarray ) -> None:

        new_se3 = np.zeros( ( 4, 4 ) )

        delta_se3 = mat.get_se3_from_vec6( vec6, is_in_degrees = False )

        new_se3[:3, :3] = self.se3[:3, :3] @ delta_se3[:3, :3]
        new_se3[:3, 3:] = self.se3[:3, :3] @ delta_se3[:3, 3:] + self.se3[:3, 3:]
        new_se3[3, 3] = 1

        self.vec6 = mat.get_vec6_from_se3( new_se3, get_degrees = False )
        self.se3 = new_se3

        self.x = self.vec6[0].squeeze()
        self.y = self.vec6[1].squeeze()
        self.z = self.vec6[2].squeeze()
        self.r_x = self.vec6[3].squeeze()
        self.r_y = self.vec6[4].squeeze()
        self.r_z = self.vec6[5].squeeze()

        self.cx = np.cos( self.r_x ).squeeze()
        self.sx = np.sin( self.r_x ).squeeze()

        self.cy = np.cos( self.r_y ).squeeze()
        self.sy = np.sin( self.r_y ).squeeze()

        self.cz = np.cos( self.r_z ).squeeze()
        self.sz = np.sin( self.r_z ).squeeze()

    def set_vec6( self, vec6: np.ndarray ) -> None:

        self.x = vec6[0].squeeze()
        self.y = vec6[1].squeeze()
        self.z = vec6[2].squeeze()
        self.r_x = vec6[3].squeeze()
        self.r_y = vec6[4].squeeze()
        self.r_z = vec6[5].squeeze()

        self.vec6 = np.array( [self.x, self.y, self.z, self.r_x, self.r_y, self.r_z] ).reshape(( 6, 1 ))
        self.se3 = mat.get_se3_from_vec6( self.vec6, is_in_degrees = False )

        self.cx = np.cos( self.r_x ).squeeze()
        self.sx = np.sin( self.r_x ).squeeze()

        self.cy = np.cos( self.r_y ).squeeze()
        self.sy = np.sin( self.r_y ).squeeze()

        self.cz = np.cos( self.r_z ).squeeze()
        self.sz = np.sin( self.r_z ).squeeze()

