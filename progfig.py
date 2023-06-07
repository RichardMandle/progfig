import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
Core functions for programatic generation of images of nematic LCs

Functions:
generate_regular_points(spacing, size)      - generates regular points in a cubic grid of points
add_randomness(points,                      - adds random displacement to a grid of points
                randomness_x=None,            displacement in x, y, z can be controled seperately
                randomness_y=None,            entering a single value makes x=y=z 
                randomness_z=None)            entering two values makes (y=z)!=x
define_vectors(points, vector_length, P2)   - generates vectors of length "vector_length" starting at each point, which are rotated to give the supplied value of P2
unit_vector(vector)                         - returns the unit vector of a vector
angle_between(v1, v2)                       - returns the angle between two vectors
convert_to_0_90_range(angles)               - converts angles to be in the range 0-90
calculate_average_angle(vectors_array)      - returns the average angle between vectors in an array of vectors
calculate_P2(angles)                        - calculates <P2> - the nematic order parameter, given a list of angles
calculate_P1(angles)                        - calculates <P1> - the polar nematic order parameter
make_apolar(points, vectors)                - nulifies <P1> by flipping half of the vectors head-to-tail

plotting functions:
plot_points(points)                         - plot an array of points in 3D (scatter)

plot_vectors(points,                        - plot vecetors in 3D to reveal nematic order
            vectors,                          vectors defined by "points" and "vectors"
            fig = False,                      plots to "fig"; creates new figure if False
            ax = False,                       plots to "ax"; creates new axis if False
            colors='p1',                      colouring according to 'p1' or 'p2'
            apolar=False)                     creates apolar nematic if False, polar if True

plot_ellipsoid(points,                      - plot ellipsoids in 3D to reveal nematic order
                vectors,                      ellipsoids centred at points, aligned with length along "vectors"
                fig=False,                    plots to "fig"; creates new figure if Fals
                ax=False,                     plots to "ax"; creates new axis if False
                color=True,                   colouring according to polar or apolar order
                apolar=False,                 creates apolar nematic if False, polar if True
                aspect_ratio=3.0,             controls aspect ratio of ellipsoids
                overlap_threshold=3.0)        checks for overlap of ellipsoids and prunes contacts

inspect_angles                              - produces a histogram of angles between vectors and resulting P1,P2 values.

Example usage:
points = progfig.generate_regular_points(0.5,5)
points = progfig.add_randomness(points,0.5)
vectors = progfig.define_vectors(points,0.5,0.6)

progfig.plot_ellipsoid(points, vectors,aspect_ratio=4,overlap_threshold=0.75,apolar=False)

'''

def generate_regular_points(spacing, size):
    """
    Generates regular points in a cubic grid with a given spacing and size.
    
    Parameters:
    spacing (float): Spacing between adjacent points.
    size (float): Size of the cubic grid.
    
    Returns:
    numpy.ndarray: An array of regular points in a cubic grid.
    """
    
    x = np.arange(0, size, spacing)
    y = np.arange(0, size, spacing)
    z = np.arange(0, size, spacing)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
   
    return points


def hexatic_offset(points):
    """
    Offsets the points by half the spacing between points in alternating rows.
    
    Parameters:
    points (numpy.ndarray): Array of regular points in a cubic grid.
    
    Returns:
    numpy.ndarray: Array of offset points.
    """
    spacing = np.sum(points[1] - points[0])
    print(spacing)
    offset_points = points.copy()

    # determine how many points per length were used in the generate_regular_points call
    points_per_length = int(np.sqrt(np.unique(points[:,0],return_counts=True)[1][0]))  
        
    #offset_points[::points_per_length, 1] += spacing
    
    #works but fragile; needs to include the spacing in the mod call.
    for n in range(len(offset_points)):
        if np.mod((offset_points[n,0]+spacing*2),spacing*2) == 0:
            offset_points[n,1] = offset_points[n,1]+(spacing/2)           
            
    return offset_points


def add_randomness(points, randomness_x=0.0, randomness_y=None, randomness_z=None):
    """
    Adds random displacements to points based on the specified randomness in each dimension.

    Parameters:
    points (numpy.ndarray): An array of points.
    randomness_x (float): The range of random displacements in the x-dimension.
    randomness_y (float): The range of random displacements in the y-dimension.
    randomness_z (float): The range of random displacements in the z-dimension.

    Returns:
    numpy.ndarray: An array of points with added random displacements.
    """

    # Check if any randomness value is None
    if randomness_y is None:
        randomness_y = randomness_x
        
    if randomness_z is None:
        randomness_z = randomness_y
        
    # Generate random displacements for each dimension
    displacements_x = np.random.uniform(-randomness_x, randomness_x, size=points.shape[0])
    displacements_y = np.random.uniform(-randomness_y, randomness_y, size=points.shape[0])
    displacements_z = np.random.uniform(-randomness_z, randomness_z, size=points.shape[0])

    # Create a matrix of random displacements
    displacements = np.column_stack((displacements_x, displacements_y, displacements_z))

    # Add the displacements to the points
    random_points = points + displacements

    return random_points


def define_vectors(points, vector_length, P2):
    """
    Defines random vectors based on the specified points, vector length, and P2 value.

    Parameters:
    points (numpy.ndarray): An array of points.
    vector_length (float): The desired length of the vectors.
    P2 (float): P2 value of the system.
    randomness_factor (float): A factor to control the randomness of the angles. Default is None.

    Returns:
    numpy.ndarray: An array of randomly defined vectors.
    """

    # Calculate the average angle based on P2
    average_angle = np.degrees(np.arccos(np.sqrt((2 * P2 + 1) / 3)))

    # Calculate the standard deviation based on the desired range
    standard_deviation =   (2-P2) **(1-P2) * average_angle
    
    # Generate random vectors
    vectors = np.empty((len(points), 3))

    for i in range(len(points)):
        # Generate random spherical angles around the average angle with Gaussian distribution
        theta = np.random.uniform(-2 *  np.pi, 2 * np.pi)
        # Generate a random angle from a Gaussian distribution
        phi = np.random.normal(loc=0, scale=standard_deviation)

        # Clamp the angle within the valid range
        #phi = np.clip(phi, 
        #              -average_angle, 
        #              average_angle)
#
        # Convert spherical angles to Cartesian coordinates
        x = np.sin(np.radians(phi)) * np.cos(theta)
        y = np.sin(np.radians(phi)) * np.sin(theta)
        z = np.cos(np.radians(phi))

        # Scale the vector to the specified length
        vectors[i] = np.array([x, y, z]) * vector_length

    return vectors
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def convert_to_0_90_range(angles):
    """
    Converts a list of angles to the range [0, 90] degrees.
    
    Parameters:
    angles (list): A list of angles in radians.
    
    Returns:
    numpy.ndarray: An array of converted angles in the range [0, 90] degrees.
    """
    
    angles_array = np.array(angles)  # Convert the list to a NumPy array
    converted_angles = (angles_array + np.pi/2) % np.pi  # Shift the angles by 90 degrees and take modulo 180
    converted_angles = np.abs(converted_angles - np.pi/2)  # Take the absolute difference from 90
    
    return converted_angles


def calculate_average_angle(vectors_array):
    """
    Calculates the average angle between vectors in an array.
    
    Parameters:
    vectors_array (numpy.ndarray): A 2D array of vectors.
    
    Returns:
    tuple: A tuple containing the following values:
        - P1 (float): First percentile of the angle distribution.
        - P1_array (numpy.ndarray): Array of angles in the P1 percentile.
        - P2 (float): Second percentile of the angle distribution.
        - P2_array (numpy.ndarray): Array of angles in the P2 percentile.
        - average_angle (float): Average angle in degrees.
        - angles (list): List of angles between vectors and the director.
    """
    
    angles = []
    
    # Calculate the director vector as the mean of the vectors in the array
    director = np.mean(vectors_array, axis=0) / np.linalg.norm(np.mean(vectors_array, axis=0))
    
    # Calculate the angles between each vector and the director
    for vector in vectors_array:
        angles.append(angle_between(vector, director))
        
    # Calculate the first percentile and its corresponding array
    P1, P1_array = calculate_P1(angles)

    # Calculate the average angle between vectors in the range [0, 90]
    average_angle = np.mean(np.degrees(convert_to_0_90_range(angles)))
    
    # Calculate the second percentile and its corresponding array
    P2, P2_array = calculate_P2(angles)

    return P1, P1_array, P2, P2_array, average_angle, angles


def calculate_P2(angles):
    """
    Calculates the P2 value and an array of P2 values for a given list of angles.
    
    Parameters:
    angles (list): A list of angles, in radians.
    
    Returns:
    tuple: A tuple containing the following values:
        - P2 (float): The calculated P2 value, i.e. nematic order parameter
        - P2_array (list): A list of P2 values corresponding to each angle.
    """
    P2_array = []
    average_angle = np.mean(angles)
    
    # Calculate P2 for each angle
    for angle in angles:
        P2_array.append((3 * (np.cos(angle)**2) - 1) / 2)
    
    # Calculate the average P2 value
    P2 = np.mean(P2_array)
            
    return P2, P2_array


def calculate_P1(angles):
    """
    Calculates the P1 value and an array of P1 values for a given list of angles.
    
    Parameters:
    angles (list): A list of angles, in radians.
    
    Returns:
    tuple: A tuple containing the following values:
        - P1 (float): The calculated P1 value.
        - P1_array (list): A list of P1 values corresponding to each angle.
    """
    
    P1_array = []
    average_angle = np.mean(angles)
    
    # Calculate P1 for each angle
    for angle in angles:
        P1_array.append(np.cos(angle))
    
    # Calculate the average P1 value
    P1 = np.mean(P1_array)
            
    return P1, P1_array

    
def plot_points(points):
    """
    Plots the given points in a 3D scatter plot.
    
    Parameters:
    points (numpy.ndarray): An array of points with shape (N, 3).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')

    # Show the plot
    plt.show()

def make_apolar(points, vectors):
    """
    Modifies the given vectors and points to remove polar order, resulting in random orientation.
    
    Parameters:
    points (numpy.ndarray): An array of points with shape (N, 3).
    vectors (numpy.ndarray): An array of vectors with shape (N, 3).
    
    Returns:
    numpy.ndarray: Modified points with polar order removed.
    numpy.ndarray: Modified vectors with polar order removed.
    """
    modified_vectors = vectors.copy()
    modified_points = points.copy()
    
    for i in range(len(modified_points)):
        if np.random.choice([True, False]):
            modified_vectors[i] = -modified_vectors[i]
            modified_points[i] = modified_points[i] - modified_vectors[i]
        
    return modified_points, modified_vectors
   
 
def plot_vectors(points, vectors, fig = False, ax = False, colors='p1', apolar=False):
    """
    Plots the given vectors originating from the given points in a 3D quiver plot.
    
    Parameters:
    points (numpy.ndarray): An array of points with shape (N, 3).
    vectors (numpy.ndarray): An array of vectors with shape (N, 3).
    colors (str): Specifies the color scheme. 'p1' for P1-based colors, 'p2' for P2-based colors. Default is 'p1'.
    apolar (bool): Indicates whether the system is apolar. If True, the vectors and points are modified to be apolar. Default is False.
    """
    if not fig:
        fig = plt.figure()
        
    if not ax:
        ax = fig.add_subplot(111, projection='3d')
        
    local_points, local_vectors = points, vectors
    
    _, _, p2, p2_array, _, _ = calculate_average_angle(local_vectors)
        
    if apolar:
        local_points, local_vectors = make_apolar(local_points, local_vectors)
    
    p1, p1_array, _, _, _, _  = calculate_average_angle(local_vectors)
    
    if colors:
        color_array = p2_array if apolar else p1_array
        color_label = 'P2' if apolar else 'P1'
        print('Colouring according to', color_label, 'values')

    vector_colors = plt.cm.magma_r(color_array)

    ax.quiver(local_points[:, 0], 
              local_points[:, 1], 
              local_points[:, 2],
              local_vectors[:, 0], 
              local_vectors[:, 1], 
              local_vectors[:, 2],
              colors=vector_colors,
              cmap='magma')

    ax.grid(False)  # Remove grid
    ax.set_axis_off()  # Remove axes
    ax.set_box_aspect([1, 1, 1])  # Remove box
    
    # add colorbar
    if colors:
        colormap = cm.get_cmap('magma_r')
        cbar_label = r'$\mathregular{P1}$' if not apolar else r'$\mathregular{P2}$'  # add label
        cbar_ticks = np.linspace(-0.5, 1, 6) if not apolar else np.linspace(0, 1, 6) #set limit
        cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), 
                            ax=ax, 
                            ticks=cbar_ticks,
                            shrink=0.5)
        cbar.ax.set_ylabel(cbar_label, rotation=0, labelpad=10, fontstyle='italic')

    plt.show()
    
    print("P1 = " + str(np.round(p1, 3)))    
    print("P2 = " + str(np.round(p2, 3)))
    

def plot_ellipsoid(points, vectors, fig=False, ax=False, color=True, apolar=False, aspect_ratio=3.0, overlap_threshold=3.0):
    """
    Plots ellipsoids defined by the given points and vectors in a 3D plot.
    
    Parameters:
    points (numpy.ndarray): An array of points with shape (N, 3).
    vectors (numpy.ndarray): An array of vectors with shape (N, 3).
    color (bool): Specifies whether to color the ellipsoids. Default is True.
    apolar (bool): Indicates whether the system is apolar. If True, the vectors and points are modified to be apolar. Default is False.
    aspect_ratio (float): The aspect ratio of the ellipsoids. Default is 3.0.
    overlap_threshold (float): The distance threshold for considering ellipsoids as overlapping. Default is 3.0.
    """
    if not fig:
        fig = plt.figure()
    
    if not ax:
        ax = fig.add_subplot(111, projection='3d')
        
    local_points, local_vectors = points, vectors
    
    _,_,p2,p2_array,_,_ = calculate_average_angle(local_vectors)
        
    if apolar:
        local_points, local_vectors = make_apolar(local_points, local_vectors)
    
    p1,p1_array,_,_,_,_  = calculate_average_angle(local_vectors)

    if color:
        color_array = p2_array if apolar else p1_array
        color_label = 'P2' if apolar else 'P1'
        print('Colouring according to', color_label, 'values')

        colormap = cm.get_cmap('magma_r')

        ellipsoid_centers = []
        
    for point, vector, c in zip(local_points, local_vectors, color_array):
        ellipsoid_center = point
        ellipsoid_length = np.linalg.norm(vector)
        ellipsoid_radius = ellipsoid_length / aspect_ratio
        
        # Normalize the vector for rotation
        normalized_vector = vector / ellipsoid_length
        
        # Calculate rotation matrix using the normalized vector
        u, v = normalized_vector[:2]
        rotation_angle = np.arctan2(v, u)
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                    [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                    [0, 0, 1]])
        
        # Generate the ellipsoid surface points
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        
        x = ellipsoid_length * np.outer(np.cos(u), np.sin(v))
        y = ellipsoid_radius * np.outer(np.sin(u), np.sin(v))
        z = ellipsoid_radius * np.outer(np.ones_like(u), np.cos(v))
        
        # Apply rotation and translation to the ellipsoid surface points
        for i in range(len(x)):
            for j in range(len(x[i])):
                point = np.array([x[i, j], y[i, j], z[i, j]])
                rotated_point = np.dot(rotation_matrix, point) + ellipsoid_center
                x[i, j], y[i, j], z[i, j] = rotated_point
        
        # Perform sanity check for overlapping ellipsoids
        skip_ellipsoid = False
        for center in ellipsoid_centers:
            distance = np.linalg.norm(center - ellipsoid_center)
            if distance < overlap_threshold:
                skip_ellipsoid = True
                break
        
        if not skip_ellipsoid:
            # Get color from colormap based on the normalized color value
            ellipsoid_color = colormap(c)
            ax.plot_surface(x, y, z, color=ellipsoid_color, alpha=0.8)
            ellipsoid_centers.append(ellipsoid_center)

    ax.grid(False)  # Remove grid
    ax.set_axis_off()  # Remove axes
    ax.set_box_aspect([1, 1, 1])  # Remove box
    
    # add colorbar
    if color:
        cbar_label = r'$\mathregular{P1}$' if not apolar else r'$\mathregular{P2}$'
        cbar_ticks = np.linspace(-0.5, 1, 6) if not apolar else np.linspace(0, 1, 6)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), 
                            ax=ax, 
                            ticks=cbar_ticks,
                            shrink=0.5)
        cbar.ax.set_ylabel(cbar_label, rotation=0, labelpad=10, fontstyle='italic')

    plt.show()
    
    print("P1 = " + str(np.round(p1, 3)))  
    print("P2 = " + str(np.round(p2, 3)))
    
    return
    
def inspect_angles(vectors):
    """
    Takes a series of vectors, prints the P1 and P2 order parameters,
    plots a histogram of the angles between vectors and the director (average vector)
    
    Parameters:
    vectors (numpy.ndarray): An array of vectors with shape (N, 3).
    """
    p1,p1_array,p2,p2_array,average_angle,angle_array = calculate_average_angle(vectors)
    
    print("P1 = " + str(np.round(p1, 3)))  
    print("P2 = " + str(np.round(p2, 3)))
    
    plt.hist(np.degrees(angle_array),bins=np.arange(0,180))
    plt.xlabel('Angle / Deg.')
    plt.ylabel('Count')
    plt.show()
    
    return p2
