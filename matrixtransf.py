import numpy as np
import matplotlib.pyplot as plt

def calculate_transformation_matrix(base_vectors, transformation_type='identity'):
    """
    Calculate the transformation matrix from the standard basis to the given basis.

    Parameters:
    base_vectors (list of 3-tuples): The basis vectors (x, y, z) in the new coordinate system.
    transformation_type (str): The type of linear transformation to apply. Options:
        - 'identity': No transformation (default)
        - 'rotation_x': Rotation around the x-axis
        - 'rotation_y': Rotation around the y-axis
        - 'rotation_z': Rotation around the z-axis
        - 'scaling': Scaling in all directions
        - 'shear_x': Shear in the x-direction
        - 'shear_y': Shear in the y-direction
        - 'shear_z': Shear in the z-direction
        - 'custom': Custom transformation matrix (provide a 3x3 numpy array)

    Returns:
    transformation_matrix (3x3 numpy array): The transformation matrix.
    """
    # Normalize the basis vectors
    base_vectors = [np.array(v) / np.linalg.norm(v) for v in base_vectors]

    # Create the transformation matrix
    if transformation_type == 'identity':
        transformation_matrix = np.eye(3)
    elif transformation_type == 'rotation_x':
        theta = np.pi / 4  # 45-degree rotation
        transformation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif transformation_type == 'rotation_y':
        theta = np.pi / 4  # 45-degree rotation
        transformation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif transformation_type == 'rotation_z':
        theta = np.pi / 4  # 45-degree rotation
        transformation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    elif transformation_type == 'scaling':
        scale_factor = 2  # Scale by a factor of 2
        transformation_matrix = np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, scale_factor]
        ])
    elif transformation_type == 'shear_x':
        shear_factor = 0.5  # Shear by a factor of 0.5
        transformation_matrix = np.array([
            [1, shear_factor, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    elif transformation_type == 'shear_y':
        shear_factor = 0.5  # Shear by a factor of 0.5
        transformation_matrix = np.array([
            [1, 0, 0],
            [shear_factor, 1, 0],
            [0, 0, 1]
        ])
    elif transformation_type == 'shear_z':
        shear_factor = 0.5  # Shear by a factor of 0.5
        transformation_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [shear_factor, 0, 1]
        ])
    elif transformation_type == 'custom':
        transformation_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example custom matrix
    else:
        raise ValueError("Invalid transformation type")

    # Combine the basis vectors and transformation matrix
    transformation_matrix = np.dot(transformation_matrix, np.array(base_vectors).T)

    return transformation_matrix

def plot_transformation_matrix(transformation_matrix):
    """
    Plot the transformation matrix as a 3D coordinate system.

    Parameters:
    transformation_matrix (3x3 numpy array): The transformation matrix.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the standard basis vectors (x, y, z)
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='x')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='z')

    # Plot the transformed basis vectors
    ax.quiver(0, 0, 0, *transformation_matrix[:, 0], color='r', linestyle='--')
    ax.quiver(0, 0, 0, *transformation_matrix[:, 1], color='g', linestyle='--')
    ax.quiver(0, 0, 0, *transformation_matrix[:, 2], color='b', linestyle='--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Transformation Matrix')
    ax.legend()

    plt.show()

# Example usage:
base_vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Standard basis
# base_vectors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]  # Non-standard basis

transformation_matrix = calculate_transformation_matrix(base_vectors, transformation_type='rotation_x')
print("Transformation Matrix:")
print(transformation_matrix)

plot_transformation_matrix(transformation_matrix)