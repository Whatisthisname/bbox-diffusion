from PIL import Image
from collections import namedtuple
import numpy as np

import torch

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

BBOX = namedtuple("BBOX", ["center_x", "center_y", "width", "height"])


def make_image(shape: tuple[int, int], n_squares: int) -> tuple[np.ndarray, list[BBOX]]:
    empty_image = torch.rand(shape) * 0.5 * 0  # TODO remove
    empty_image_pil = Image.fromarray((empty_image.numpy() * 255).astype(np.uint8))

    bboxes = []
    colors = []
    for _ in range(n_squares):
        x = np.random.randint(0, shape[0])
        y = np.random.randint(0, shape[1])

        # x = int(
        #     shape[0] // 2
        #     + shape[0] // 2 * (np.random.binomial(1, 0.5, 1) * 2 - 1) * 0.5
        # )
        # y = 50

        size = np.random.randint(10, 20)
        angle = np.random.randint(0, 90)

        # Create a square with random color
        color = tuple(np.random.randint(0, 256, size=3) * 0 + 255)  # TODO remove
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

        # make grayscale
        # empty_image_pil = empty_image_pil.convert("L")

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
        torch.tensor(bboxes)[bboxes_sort_idx].float(),
        [tuple(x.tolist()) for x in torch.tensor(colors)[bboxes_sort_idx]],
    )


def label_sort_order_big_to_small(bboxes: torch.Tensor) -> torch.Tensor:
    # return bboxes  # TODO undo this
    sizes = bboxes[:, 2] * bboxes[:, 3]
    return torch.argsort(sizes, descending=True)


class SquareDataset(Dataset):
    def __init__(self, image_size, num_boxes, num_samples):
        self.image_size = image_size
        self.num_boxes = num_boxes
        self.num_samples = num_samples
        self.images = []
        self.bboxes = []

        # precomputing
        for _ in range(num_samples):
            img, bboxes, _ = make_image(self.image_size, self.num_boxes)
            self.images.append(img.permute(2, 0, 1))
            self.bboxes.append(bboxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.bboxes[idx]


if __name__ == "__main__":
    # Parameters
    image_size = (100, 100)
    num_boxes = 5
    num_samples = 1000
    batch_size = 32

    # Create dataset and dataloader
    dataset = SquareDataset(image_size, num_boxes, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Example usage
    for batch_imgs, batch_bboxes in dataloader:
        print(batch_imgs.shape)  # Should be [batch_size, 3, 100, 100]
        print(batch_bboxes.shape)  # Should be [batch_size, num_boxes, 4]
        break

    img, bboxes = make_image((100, 100), 5)
    plt.imshow(img)
    plt.show()
