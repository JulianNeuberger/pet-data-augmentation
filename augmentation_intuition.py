from datasets import load_dataset
import numpy as np
import PIL.PngImagePlugin
import matplotlib.pyplot as plt
import scipy.ndimage as scimage
import typing
import random
import tensorflow as tf


def rotate(min_angle: float, max_angle: float):
    def inner(image: np.ndarray):
        return scimage.rotate(image, angle=min_angle + random.random() * (max_angle - min_angle), reshape=False)

    return inner


def scale(min_factor: float, max_factor: float):
    def inner(image: np.ndarray):
        return scimage.zoom(image, zoom=min_factor + random.random() * (max_factor - min_factor))

    return inner


def translate(min_amount: typing.Tuple[float, float], max_amount: typing.Tuple[float, float]):
    def inner(image: np.ndarray):
        shift_x = int(image.shape[0] * (min_amount[0] + random.random() * (max_amount[0] - min_amount[0])))
        shift_y = int(image.shape[1] * (min_amount[1] + random.random() * (max_amount[1] - min_amount[1])))

        return scimage.shift(image, (shift_x, shift_y), mode="constant", cval=0.0)

    return inner


def flip():
    def inner(image: np.ndarray):
        return np.flip(image, axis=1 if random.random() > .5 else 0)

    return inner


def brightness(min_value: float, max_value: float):
    def inner(image: np.ndarray):
        value = min_value + random.random() * (max_value - min_value)
        return np.clip(image + value, 1, 0)

    return inner


def noise(min_amount: float, max_amount: float):
    def inner(image: np.ndarray):
        amount = min_amount + random.random() * (max_amount - min_amount)
        return np.clip(image + np.random.random(image.shape) * amount, 0, 1)

    return inner


def main():
    ds = load_dataset("ylecun/mnist")
    image: PIL.PngImagePlugin.PngImageFile = ds["train"][22]["image"]
    original = np.asarray(image)
    original = original / 255

    fig, axs = plt.subplots(2, 4)

    axs[0, 0].imshow(original, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axs[1, 0].imshow(original, cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    axs[0, 1].imshow(rotate(-30, -30)(original), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axs[1, 1].imshow(rotate(180, 180)(original), cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    axs[0, 2].imshow(translate((.2, 0), (.2, 0))(original), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axs[1, 2].imshow(translate((.5, 0), (.5, 0))(original), cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    axs[0, 3].imshow(noise(.3, .3)(original), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axs[1, 3].imshow(noise(1, 1)(original), cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    plt.savefig("figures/intuition/mnist.pdf")
    plt.savefig("figures/intuition/mnist.png")


main()
