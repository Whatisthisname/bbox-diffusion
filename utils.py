import numpy as np
import torch
import matplotlib.pyplot as plt


def draw_points_on_image(image, points, colors: list = None):
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
        raise ValueError(f"Points tensor must have size (n, 2), {points.shape} given")

    if colors is None:
        colors = ["red"] * len(points)

    # Convert the tensor to a numpy array
    points_np = points.cpu().numpy()

    # Plot the image
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")

    # Draw each point on the image
    for point, color in zip(points_np, colors):
        x, y = point[0], point[1]
        ax.scatter(x, y, color=color)

    # convert image to numpy array
    fig.canvas.draw()
    # close the plot
    plt.close(fig)

    return np.array(fig.canvas.renderer.buffer_rgba())
