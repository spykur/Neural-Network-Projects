import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



GRID_WIDTH = 10      
GRID_HEIGHT = 20     

EPOCHS = 10
LEARNING_RATE = 1
RADIUS = max(GRID_WIDTH, GRID_HEIGHT) / 2



def generate_sphere_points(n_points=2000):

    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.column_stack((x, y, z))



def initialize_grid(width, height):

    grid = np.random.uniform(-1, 1, (height, width, 3))

    norms = np.linalg.norm(grid, axis=2, keepdims=True)
    grid = grid / norms

    return grid



def find_bmu(grid, sample):

    distances = np.linalg.norm(grid - sample, axis=2)

    return np.unravel_index(np.argmin(distances), distances.shape)


def update_grid(grid, sample, bmu, lr, radius):

    height, width = grid.shape[:2]

    for i in range(height):
        for j in range(width):

            dist_to_bmu = np.sqrt((i - bmu[0])**2 +(j - bmu[1])**2)

            if dist_to_bmu < radius:

                influence = np.exp(-(dist_to_bmu**2) / (2 * radius**2))

                grid[i, j] += (influence *lr *(sample - grid[i, j]))

                grid[i, j] /= np.linalg.norm(grid[i, j])


def train_som(
    grid,
    data,
    epochs,
    initial_lr,
    initial_radius
):

    for epoch in range(epochs):

        lr = initial_lr * np.exp(-epoch / epochs)

        radius = (initial_radius *np.exp(-epoch / epochs))

        np.random.shuffle(data)

        for sample in data:

            bmu = find_bmu(grid, sample)

            update_grid(grid,sample,bmu,lr,radius)

        print(f"Epoka {epoch+1}/{epochs}")

    return grid


def plot_som(grid, data):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        s=2,
        alpha=0.2
    )

    height, width = grid.shape[:2]
    
    for i in range(height):
        for j in range(width):

            x, y, z = grid[i, j]

            ax.scatter(x, y, z, color='red')
            
            if j < width - 1:

                x2, y2, z2 = grid[i, j + 1]

                ax.plot([x, x2],[y, y2],[z, z2],color='blue')

            if i < height - 1:

                x2, y2, z2 = grid[i + 1, j]

                ax.plot(
                    [x, x2],
                    [y, y2],
                    [z, z2],
                    color='blue'
                )

    ax.set_title(
        f"Prostokątna SOM na sferze "
        f"({GRID_HEIGHT}x{GRID_WIDTH})"
    )

    plt.show()



data = generate_sphere_points()

grid = initialize_grid(GRID_WIDTH, GRID_HEIGHT)

trained_grid = train_som(grid, data, EPOCHS, LEARNING_RATE, RADIUS)

plot_som(trained_grid, data)