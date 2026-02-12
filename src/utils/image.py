import os
import math
import matplotlib.pyplot as plt


def save_image_grid(images, out_dir, nrow=10, show=False):
    """
    Save a grid of images exactly as they are (NO normalization).
    """
    images = images.detach().cpu()
    B, C, _, _ = images.shape

    ncol = math.ceil(B / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))

    # Normalize axes into a 2D list for consistent indexing
    if ncol == 1 and nrow == 1:
        axes = [[axes]]
    elif ncol == 1:
        axes = [axes]          # wrap row
    elif nrow == 1:
        axes = [[ax] for ax in axes]  # wrap column

    # Turn all axes off first
    for row in axes:
        for ax in row:
            ax.axis("off")

    # Fill the grid
    for i, img in enumerate(images):
        r, c = divmod(i, nrow)
        ax = axes[r][c]

        img = img.permute(1, 2, 0).numpy()

        if C == 1:
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img)

        ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(out_dir, bbox_inches="tight", pad_inches=0.05)

    if show:
        plt.show()

    plt.close(fig)