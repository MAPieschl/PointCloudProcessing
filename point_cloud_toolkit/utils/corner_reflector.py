from dependencies import *
import utils.globals as globals

def moller_trumbore(ray_origin, ray_vector, triangle):
    """
    Möller–Trumbore intersection algorithm (adapted from Gemini)
    
    Args:
        ray_origin (np.array): Origin of the ray (x, y, z)
        ray_vector (np.array): Direction of the ray (x, y, z)
        v0, v1, v2 (np.array): Vertices of the triangle
        
    Returns:
        float or None: Distance 't' from origin to intersection if hit, else None.
    """

    epsilon = 1e-6 # Tolerance for floating point errors

    v0 = triangle['corners'][0]
    v1 = triangle['corners'][1]
    v2 = triangle['corners'][2]
    
    # 1. Find edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # 2. Begin calculating determinant - also used to calculate u parameter
    h = np.cross(ray_vector, edge2)
    
    # 3. Check if ray is parallel to the triangle
    # If the determinant is near zero, the ray lies in the plane of the triangle
    det = np.dot(edge1, h)
    
    if -epsilon < det < epsilon:
        return None # This ray is parallel to the triangle
    
    inv_det = 1.0 / det
    
    # 4. Calculate distance from V0 to ray origin
    s = ray_origin - v0
    
    # 5. Calculate u parameter and test bounds
    u = inv_det * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None # Intersection is outside the triangle
        
    # 6. Prepare to test v parameter
    q = np.cross(s, edge1)
    
    # 7. Calculate v parameter and test bounds
    v = inv_det * np.dot(ray_vector, q)
    if v < 0.0 or u + v > 1.0:
        return None # Intersection is outside the triangle
        
    # 8. Calculate t - the distance along the ray to the intersection
    t = inv_det * np.dot(edge2, q)
    
    if t > epsilon: # Ray intersection
        return t
    else: # This means there is a line intersection but not a ray intersection (it's behind the sensor)
        return None

def get_reflection(ray_origin: np.ndarray, ray_vector: np.ndarray, corners: np.ndarray, apex: np.ndarray, print_func: Callable[[str], None] = print) -> dict[str, np.ndarray]:
    """
    Adapted from Gemini implementation of the Möller–Trumbore intersection algorithm.
    
    @param  ray_origin  (np.array) Origin of the ray (x, y, z)
    @param  ray_vector  (np.array) Direction of the ray (x, y, z)
    @param  corners     (np.array) with shape (3, 3) indicating (corner, (x, y, z)) CCW when looking into the reflector
    @param  apex        (np.array) (3,) indicating the apex location of the corner reflector
        
    Returns:
        if collision occurs {'collision_point': (x, y, z), 'reflection_vector': (x, y, z)}
        else {}
    """
    
    if( ray_origin.shape != (3,) ):
        print_func( f"Parameter 'ray_origin' must be shape (3,), not {ray_origin.shape}" )
        return {}
    
    if( ray_vector.shape != (3,) ):
        print_func( f"Parameter 'ray_vector' must be shape (3,), not {ray_vector.shape}" )
        return {}

    if( corners.shape != (3, 3) ):
        print_func( f"Parameter 'corners' must be shape (3, 3), not {corners.shape}" )
        return {}
    
    if( apex.shape != (3,) ):
        print_func( f"Parameter 'apex' must be shape (3,), not {apex.shape}" )
        return {}

    corners = np.concatenate( ( corners, np.expand_dims( corners[0], axis = 0 ) ), axis = 0 )
    triangles = []
    for i in range( 3 ):
        normal = np.cross( np.array( corners[i + 1] ) - np.array( corners[i] ), np.array( apex ) - np.array( corners[i + 1] ) )

        triangles.append({
            'corners': np.array( [corners[i], corners[i + 1], apex] ),
            'normal': normal / np.linalg.norm( normal ),
        })

    for i in range( len( triangles ) ):
        d_int = moller_trumbore( ray_origin, ray_vector, triangles[i] )
        if( type( d_int ) != type( None ) ):
            return {
                'collision_point': ray_origin + ray_vector * d_int,
                'reflection_vector': ray_vector - 2 * np.dot( ray_vector, triangles[i]['normal'] ) * triangles[i]['normal']
            }
        
    return {}