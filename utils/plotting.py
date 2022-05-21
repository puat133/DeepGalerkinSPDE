import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as np


def set_plot(use_latex: bool = True):
    sns.set_theme(style="white", font_scale=3)
    if use_latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    lw = 1.
    marker_size = 6
    fig_size = (15, 10)
    return fig_size, lw, marker_size


fig_size, lw, marker_size = set_plot(use_latex=True)


def plot_image_and_save(image: np.ndarray, file_path: str, origin='lower', cmap=plt.cm.seismic_r):
    fig = plt.figure(figsize=fig_size)
    im = plt.imshow(image, cmap=cmap, origin=origin)
    fig.colorbar(im)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()
