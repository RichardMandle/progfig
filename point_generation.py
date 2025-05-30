import numpy as np
from scipy.stats import vonmises, uniform

def generate_regular_points(spacing, size):
    num_points_x = int(size[0] / spacing[0])
    num_points_y = int(size[1] / spacing[1])
    num_points_z = int(size[2] / spacing[2])

    points = []
    for x in range(num_points_x):
        for y in range(num_points_y):
            for z in range(num_points_z):
                points.append([x * spacing[0], y * spacing[1], z * spacing[2]])
                
    return np.array(points)


def define_vectors(points, vector_length, P2):
    if P2 < -0.5 or P2 > 1.0:
        raise ValueError("P2 must be between -0.5 and 1.0")

    vectors = np.empty((len(points), 3))

    if P2 == 1: # perfect alignment - easy peasy
        vectors[:] = np.array([0, 0, 1]) * vector_length
        return vectors

    if P2 == 0: # isotropic; just a uniform distribution on the sphere 
        for i in range(len(points)):
            phi = np.arccos(2 * uniform.rvs() - 1) 
            theta = uniform.rvs(0, 2 * np.pi)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            vectors[i] = np.array([x, y, z]) * vector_length
        return vectors
    
    if P2 > 0: # normal positive P2; use the vonmises dist. in scipy:
        kappa = 3 * P2 / (1 - P2)
        for i in range(len(points)):
            theta = uniform.rvs(0, 2 * np.pi)
            phi = vonmises.rvs(kappa)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            vectors[i] = np.array([x, y, z]) * vector_length
        return vectors

    if P2 < 0: # wierdo negative P2; 
        for i in range(len(points)):
            theta = uniform.rvs(0, 2 * np.pi)
            if P2 == -0.5:
                phi = np.pi / 2  # most likely use case; perfect antinematic order
            else:
                phi = np.arccos(2 * uniform.rvs() - 1)  # yeah. this STILL doesn't really work (TO DO)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            vectors[i] = np.array([x, y, z]) * vector_length
        return vectors

    return vectors

def add_tilt(points, vectors, tilt_angle_x=0, tilt_angle_y=0):
    if tilt_angle_x == 0 and tilt_angle_y == 0:
        return points, vectors

    if tilt_angle_x != 0:
        tilt_rad_x = np.radians(tilt_angle_x)
        tilt_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(tilt_rad_x), -np.sin(tilt_rad_x)],
            [0, np.sin(tilt_rad_x), np.cos(tilt_rad_x)]
        ])
        vectors = np.dot(vectors, tilt_matrix_x.T)

    if tilt_angle_y != 0:
        tilt_rad_y = np.radians(tilt_angle_y)
        tilt_matrix_y = np.array([
            [np.cos(tilt_rad_y), 0, np.sin(tilt_rad_y)],
            [0, 1, 0],
            [-np.sin(tilt_rad_y), 0, np.cos(tilt_rad_y)]
        ])
        vectors = np.dot(vectors, tilt_matrix_y.T)

    return points, vectors

def hexatic_offset(points, plane_offset=False):
    # if we have hexatic (HCP) packing/order, we need to do something a bit different
    spacing = np.sum(points[1] - points[0])
    offset_points = points.copy()
    points_per_length = int(np.sqrt(np.unique(points[:, 0], return_counts=True)[1][0]))

    for n in range(len(offset_points)):
        if np.mod((offset_points[n, 0] + spacing * 2), spacing * 2) == 0:
            offset_points[n, 1] = offset_points[n, 1] + (spacing / 2)

    if plane_offset:
        for n in range(len(offset_points)):
            if offset_points[n, 2] % (spacing * 2) == 0:
                offset_points[n, 0] = offset_points[n, 0] + spacing / 2
                offset_points[n, 1] = offset_points[n, 1] + spacing / 2

    return offset_points

def add_randomness(points, randomness_x=0.0, randomness_y=None, randomness_z=None):
    if randomness_y is None:
        randomness_y = randomness_x
    if randomness_z is None:
        randomness_z = randomness_y

    displacements_x = np.random.uniform(-randomness_x*0.5, randomness_x*0.5, size=points.shape[0])
    displacements_y = np.random.uniform(-randomness_y*0.5, randomness_y*0.5, size=points.shape[0])
    displacements_z = np.random.uniform(-randomness_z*0.5, randomness_z*0.5, size=points.shape[0])

    displacements = np.column_stack((displacements_x, displacements_y, displacements_z))
    random_points = points + displacements

    return random_points

def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.arccos(dot_product)

def convert_to_0_90_range(angles):
    angles_array = np.array(angles)
    converted_angles = (angles_array + np.pi/2) % np.pi
    converted_angles = np.abs(converted_angles - np.pi/2)
    return converted_angles

def compute_director(vectors_array):
    # idea borrowed from mdtraj order.py <https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/order.py>
    norm_vectors = vectors_array / np.linalg.norm(vectors_array, axis=1)[:, np.newaxis]

    Q = np.zeros((3, 3))
    for vec in norm_vectors:
        Q += np.outer(vec, vec) - np.eye(3) / 3.0
    Q /= len(norm_vectors)
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    director = eigenvectors[:, np.argmax(eigenvalues)] # our director is the eigenvector corresponding to the largest eigenvalue
    
    return director

def calculate_average_angle(vectors_array):
    angles = []
    director = compute_director(vectors_array)
    for vector in vectors_array:
        angles.append(angle_between(vector, director))

    average_angle = np.mean(np.degrees(convert_to_0_90_range(angles)))
    
    P1, P1_array = calculate_P1(np.array(angles))
    P2, P2_array = calculate_P2(np.array(angles))
    
    return P1, P1_array, P2, P2_array, average_angle, angles

def calculate_P1(angles):
    P1_array = np.cos(angles)
    P1 = np.mean(P1_array)
    return P1, P1_array

def calculate_P2(angles):
    P2_array = (3 * np.cos(angles) ** 2 - 1) / 2
    P2 = np.mean(P2_array)
    return P2, P2_array
