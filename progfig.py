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

def plot_vectors(points,                    - plot vecetors in 3D to reveal nematic order
vectors,                                      vectors defined by "points" and "vectors"
fig = False,                                  plots to "fig"; creates new figure if False
ax = False,                                   plots to "ax"; creates new axis if False
box=False,                                    colouring according to 'p1' or 'p2'
colors='p1',                                  creates apolar nematic if False, polar if True
apolar=False,                                 box controls wether or not to draw a box around it all
color_map='magma_r'):                         color_map allows to select the colormap used.


plot_ellipsoid(points,                      - plot ellipsoids in 3D to reveal nematic order
                vectors,                      ellipsoids centred at points, aligned with length along "vectors"
                fig=False,                    plots to "fig"; creates new figure if Fals
                ax=False,                     plots to "ax"; creates new axis if False
                color=True,                   colouring according to polar or apolar order
                apolar=False,                 creates apolar nematic if False, polar if True
                aspect_ratio=3.0,             controls aspect ratio of ellipsoids
                overlap_threshold=3.0)        checks for overlap of ellipsoids and prunes contacts

inspect_angles                              - produces a histogram of angles between vectors and resulting P1,P2 values.
make_a_box(points,                          - used to plot a 3D box around vector or ellipspoid plots.
            fig=False,
            ax=False):

Example usage:
points = progfig.generate_regular_points(0.5,5)
points = progfig.add_randomness(points,0.5)
vectors = progfig.define_vectors(points,0.5,0.6)

progfig.plot_ellipsoid(points, vectors,aspect_ratio=4,overlap_threshold=0.75,apolar=False)

'''

def make_apolar_nematic_vectors(spacing=0.5,size=5,randomness=0.375,vector_length=0.5,p2=0.6,color='p2'):

    points = generate_regular_points(spacing,size)
    points = add_randomness(points,
                            randomness_x=randomness,
                            randomness_y=randomness,
                            randomness_z=randomness)

    vectors = define_vectors(points,vector_length,p2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors=color,
             apolar=True,
             box=True,
             color_map='bwr')
             
def make_polar_nematic_vectors(spacing=0.5,size=5,randomness=0.375,vector_length=0.5,p2=0.5,color='p1'):

    points = generate_regular_points(spacing,size)
    points = add_randomness(points,
                            randomness_x=randomness,
                            randomness_y=randomness,
                            randomness_z=randomness)

    vectors = define_vectors(points,vector_length,p2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors=color,
             apolar=False,
             box=True,
             color_map='bwr')

def make_apolar_smectic_vectors(spacing=0.5,size=5,randomness=0.375,vector_length=0.5,p2=0.6,color='p2'):

    points = generate_regular_points(spacing,size)
    points = add_randomness(points,
                            randomness_x=randomness,
                            randomness_y=randomness,
                            randomness_z=randomness/25)

    vectors = define_vectors(points,vector_length,p2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors=color,
             apolar=True,
             box=True,
             color_map='bwr')
             
def make_polar_smectic_vectors(spacing=0.5,size=5,randomness=0.375,vector_length=0.5,p2=0.5,color='p1'):

    points = generate_regular_points(spacing,size)
    points = add_randomness(points,
                            randomness_x=randomness,
                            randomness_y=randomness,
                            randomness_z=randomness/25)

    vectors = define_vectors(points,vector_length,p2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors=color,
             apolar=False,
             box=True,
             color_map='bwr')    
    
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


def hexatic_offset(points,plane_offset=False):
    """
    Offsets the points by half the spacing between points in alternating rows.
    
    You can combine with random-disorder to create hexatic lamellar structures.
    
    Parameters:
    points (numpy.ndarray): Array of regular points in a cubic grid.
    plane_offset: if True, each layer is offset in x and y by 1/2 of the spacing, mimicing FCP packing (roughly).
                  if False, its more of a pure hexagonal packing.
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
    
    if plane_offset==True:
            for n in range(len(offset_points)):
                if offset_points[n,2]%(spacing*2) == 0:
                    offset_points[n,0] = offset_points[n,0] + spacing/2
                    offset_points[n,1] = offset_points[n,1] + spacing/2

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


def add_tilt(points, vectors, tilt_angle=20, rotation_axis=[1, 0, 0]):
    # Convert vectors to numpy array
    vectors = np.array(vectors)
    
    # Normalize rotation axis
    rotation_axis = np.array(rotation_axis)
    rotation_axis = rotation_axis.astype(float) / np.linalg.norm(rotation_axis)
    
    # Convert tilt angle to radians
    tilt_angle_rad = np.radians(tilt_angle)
    
    # Create rotation matrix
    c = np.cos(tilt_angle_rad)
    s = np.sin(tilt_angle_rad)
    rotation_matrix = np.array([
        [c + rotation_axis[0]**2 * (1 - c), rotation_axis[0] * rotation_axis[1] * (1 - c) - rotation_axis[2] * s, rotation_axis[0] * rotation_axis[2] * (1 - c) + rotation_axis[1] * s],
        [rotation_axis[1] * rotation_axis[0] * (1 - c) + rotation_axis[2] * s, c + rotation_axis[1]**2 * (1 - c), rotation_axis[1] * rotation_axis[2] * (1 - c) - rotation_axis[0] * s],
        [rotation_axis[2] * rotation_axis[0] * (1 - c) - rotation_axis[1] * s, rotation_axis[2] * rotation_axis[1] * (1 - c) + rotation_axis[0] * s, c + rotation_axis[2]**2 * (1 - c)]
    ])
    
    # Apply rotation to vectors
    new_vectors = np.dot(rotation_matrix, vectors.T).T
    
    return points, new_vectors
    

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
            0.i
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
    ax = plt.axes(projection='3d')
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
    modified_points: Modified points with polar order removed.
    modified_vectors: Modified vectors with polar order removed.
    """
    modified_vectors = vectors.copy()
    modified_points = points.copy()
    
    for i in range(len(modified_points)):
        if np.random.choice([True, False]):
            modified_vectors[i] = -modified_vectors[i]
            modified_points[i] = modified_points[i] - modified_vectors[i]
        
    return modified_points, modified_vectors
   
 
def plot_vectors(points, vectors, fig = False, ax = False, box=False, colors='p1', apolar=False,color_map='magma_r'):
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
        ax = plt.axes(projection='3d')
        
    local_points, local_vectors = points, vectors
    
    _, _, p2, p2_array, _, _ = calculate_average_angle(local_vectors)
        
    if apolar:
        local_points, local_vectors = make_apolar(local_points, local_vectors)
    
    p1, p1_array, _, _, _, _  = calculate_average_angle(local_vectors)
    

    if colors.lower() is 'p1' or 'p2':
        color_array = p1_array if (colors.lower() == 'p1') else p2_array
        color_label =  'P1' if (colors.lower() == 'p1') else 'P2'
    
    else:    
        color_array = p2_array if apolar else p1_array
        color_label =  'P2' if apolar else 'P1'
        print(color_label)
        #color_label = 'P2' if apolar else 'P1'   
        
    print('Colouring according to', color_label, 'values')

    vector_colors = getattr(plt.cm, color_map)(color_array)

    ax.quiver(local_points[:, 0], 
              local_points[:, 1], 
              local_points[:, 2],
              local_vectors[:, 0], 
              local_vectors[:, 1], 
              local_vectors[:, 2],
              colors=vector_colors,
              cmap=color_map)

    ax.grid(False)  # Remove grid
    ax.set_box_aspect([1, 1, 1])  # Remove box
    
    ax.set_axis_off()  # Remove axes
    
    if box == True:
        make_a_box(local_points,fig=fig,ax=ax)
        
    add_color_bar(color_map,color_label,apolar,fig=fig,ax=ax)     # add colorbar
    
    print("P1 = " + str(np.round(p1, 3)))    
    print("P2 = " + str(np.round(p2, 3)))
    
    plt.show()
    
    
def plot_ellipsoid(points, vectors, fig=False, ax=False, box=False, colors='p1', apolar=False, aspect_ratio=3.0, overlap_threshold=1.0,color_map='magma_r'):
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
    
    #colouring
    if colors.lower() is 'p1' or 'p2':
        color_array = p1_array if (colors.lower() == 'p1') else p2_array
        color_label =  'P1' if (colors.lower() == 'p1') else 'P2'
    
    else:    
        color_array = p2_array if apolar else p1_array
        color_label =  'P2' if apolar else 'P1'
        print(color_label)
    colormap = cm.get_cmap(getattr(plt.cm, color_map))#(color_array))    

    ellipsoid_centers = []
    
    print('Colouring according to', color_label, 'values')

    vector_colors = getattr(plt.cm, color_map)(color_array)
    
    #if color:
    #    color_array = p2_array if apolar else p1_array
    #    color_label = 'P2' if apolar else 'P1'
    #    print('Colouring according to', color_label, 'values')
    
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
    
    if box == True:
        make_a_box(local_points,fig=fig,ax=ax)
        
    ax.set_box_aspect([1, 1, 1])  
    
    add_color_bar(color_map,color_label,apolar,fig=fig,ax=ax)     # add colorbar

    print("P1 = " + str(np.round(p1, 3)))    
    print("P2 = " + str(np.round(p2, 3)))
    
    plt.show()
    
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

def make_a_box(points,fig=False,ax=False):
    '''
    simply makes a box around some 3d data.
    '''

    max_val = np.round(np.max([points]),1)  # Replace with your measured value
    box_vertices = [
        [0, 0, 0],
        [0, 0, max_val],
        [0, max_val, max_val],
        [0, max_val, 0],
        [max_val, 0, 0],
        [max_val, 0, max_val],
        [max_val, max_val, max_val],
        [max_val, max_val, 0]
    ]

    # Connect vertices to form the box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]

    # Plot the box edges
    for edge in edges:
        x_values = [box_vertices[edge[0]][0], box_vertices[edge[1]][0]]
        y_values = [box_vertices[edge[0]][1], box_vertices[edge[1]][1]]
        z_values = [box_vertices[edge[0]][2], box_vertices[edge[1]][2]]
        ax.plot(x_values, y_values, z_values, color='black')

    
def add_color_bar(color_map,color_label,apolar=False,fig=False,ax=False):
    '''
    function that handles drawing a color bar for various plots
    '''

    colormap = cm.get_cmap(color_map)
    cbar_label = r'$\mathregular{P1}$' if (color_label == 'P1') else r'$\mathregular{P2}$'  # add label
    cbar_ticks = np.linspace(-0.5, 1, 6) if not apolar else np.linspace(0, 1, 6) #set limit
    cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), 
                        ax=ax, 
                        ticks=cbar_ticks,
                        shrink=0.5)
    cbar.ax.set_ylabel(cbar_label, rotation=0, labelpad=10, fontstyle='italic')

