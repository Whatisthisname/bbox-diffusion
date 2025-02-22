import numpy as np
import torch
import matplotlib.pyplot as plt


def draw_points_on_image(image, points):
    """
    Draws 2D points on the given image.

    Parameters:
    image (numpy.ndarray): The image on which to draw the points.
    points (torch.Tensor): A tensor of 2D points of size (n, 2).

    Returns:
    numpy.ndarray: The image with points drawn on it.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array")

    if not isinstance(points, torch.Tensor):
        raise TypeError("Points must be a torch tensor")

    if points.size(1) != 2:
        raise ValueError("Points tensor must have size (n, 2)")

    # Convert the tensor to a numpy array
    points_np = points.cpu().numpy()

    # Plot the image
    plt.imshow(image)
    plt.axis("off")

    # Draw each point on the image
    for point in points_np:
        x, y = point[0], point[1]
        plt.plot(x, y, "go", markersize=5)

    # convert image to numpy array
    plt.axis("off")

    as_numpy = plt.gcf()
    as_numpy.canvas.draw()

    return np.array(as_numpy.canvas.renderer.buffer_rgba())
