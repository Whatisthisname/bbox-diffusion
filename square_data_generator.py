from PIL import Image
from collections import namedtuple
import numpy as np

import torch

import matplotlib.pyplot as plt

BBOX = namedtuple("BBOX", ["center_x", "center_y", "width", "height"])


def make_image(shape: tuple[int, int], n_squares: int) -> tuple[np.ndarray, list[BBOX]]:
    empty_image = torch.rand(shape) * 0.5
    empty_image_pil = Image.fromarray((empty_image.numpy() * 255).astype(np.uint8))

    bboxes = []
    colors = []
    for _ in range(n_squares):
        x = np.random.randint(0, shape[0])
        y = np.random.randint(0, shape[1])
        size = np.random.randint(10, 20)
        angle = np.random.randint(0, 90)

        # Create a square with random color
        color = tuple(np.random.randint(0, 256, size=3))
        square = Image.new("RGBA", (size, size), color + (255,))

        # Create an alpha channel for the square
        alpha = Image.fromarray((np.ones((size, size)) * 200).astype(np.uint8))
        square.putalpha(alpha)

        square = square.rotate(angle, expand=True)

        # Calculate the position to paste the rotated square
        paste_x = x - square.size[0] // 2
        paste_y = y - square.size[1] // 2

        # Get the bounding box of the square
        bboxes.append(
            BBOX(
                x,
                y,
                square.size[0],
                square.size[1],
            )
        )

        colors.append(color)
        # Pad square to same size as image, but keep paste_x and paste_y
        padded_square = Image.new("RGBA", empty_image_pil.size)
        padded_square.paste(square, (paste_x, paste_y))

        # Composite the square onto the image
        empty_image_pil = Image.alpha_composite(
            empty_image_pil.convert("RGBA"), padded_square.convert("RGBA")
        ).convert("RGB")

    bboxes_sort_idx = label_sort_order_big_to_small(
        torch.tensor(
            [
                [bbox.center_x, bbox.center_y, bbox.width, bbox.height]
                for bbox in bboxes
            ],
            dtype=torch.float32,
        )
    )

    return (
        torch.from_numpy(np.array(empty_image_pil)) / 255.0,
        torch.tensor(bboxes)[bboxes_sort_idx],
        torch.tensor(colors)[bboxes_sort_idx],
    )


def label_sort_order_big_to_small(bboxes: torch.Tensor) -> torch.Tensor:
    # return bboxes  # TODO undo this
    sizes = bboxes[:, 2] * bboxes[:, 3]
    return torch.argsort(sizes, descending=True)


if __name__ == "__main__":
    img, bboxes = make_image((100, 100), 5)
    plt.imshow(img)
    plt.show()
