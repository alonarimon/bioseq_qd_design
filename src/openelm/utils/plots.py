import os
from typing import Dict, List
import numpy as np
from matplotlib import pyplot as plt
import wandb
from matplotlib.colors import Normalize
import matplotlib.cm as cm



def plot_learning_curves(
        learning_stats: Dict[str, List[float]],
        save_dir: str,
        model_id: int,
        title: str = "Learning Curve",
        x_label: str = "Minibatch",
        y_label: str = "Score",
        legend_loc: str = "upper right",
        show_legend: bool = True,
        save_fig: bool = True,
        fig_name: str = "learning_curve",
):
    """
    Plot the learning curve from the learning stats.
    Args:
        learning_stats (Dict[str, List[float]]): Learning stats to plot.
        save_dir (str): Directory to save the plot.
        model_id (int): Model ID for saving the plot.
        title (str): Title of the plot.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        legend_loc (str): Location of the legend.
        show_legend (bool): Whether to show the legend.
        save_fig (bool): Whether to save the figure.
        fig_name (str): Name of the figure file.
    """
    plt.figure(figsize=(10, 6))
    for key, values in learning_stats.items():
        plt.plot(values, label=key)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if show_legend:
        plt.legend(loc=legend_loc)
    wandb.log({title: wandb.Image(plt)})
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{fig_name}_{model_id}.png"))
    # plt.show()


def plot_distance_histograms(
        all_distances,
        topk_distances,
        downsampled_distances,
        title,
        save_path
):
    """Plot histograms of distances from real, ref, and top-k solutions."""
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, max(np.max(all_distances), np.max(topk_distances), np.max(downsampled_distances)), 30)

    plt.hist(all_distances, bins=bins, alpha=0.5, label='All Solutions', color='blue', edgecolor='black')
    plt.hist(topk_distances, bins=bins, alpha=0.5, label=f'Top K Solutions', color='orange', edgecolor='black')
    plt.hist(downsampled_distances, bins=bins, alpha=0.5, label='Downsampled Solutions', color='green', edgecolor='black')

    plt.title(title)
    plt.xlabel('Pairwise Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    wandb.log({title: wandb.Image(plt)})
    plt.savefig(save_path)
    plt.close()

def plot_distance_histograms_only_k(
    
        topk_distances,
        downsampled_distances,
        title,
        save_path
):
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, max(np.max(topk_distances), np.max(downsampled_distances)), 30)

    plt.hist(topk_distances, bins=bins, alpha=0.5, label=f'Top K Solutions', color='orange', edgecolor='black')
    plt.hist(downsampled_distances, bins=bins, alpha=0.5, label='Downsampled Solutions', color='green', edgecolor='black')

    plt.title(title)
    plt.xlabel('Pairwise Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    wandb.log({title: wandb.Image(plt)})
    plt.savefig(save_path)
    plt.close()


def plot_centroid_map(phenotypes, 
                      scores, 
                      centroids, 
                      behavior_space_bounds, 
                      title, 
                      save_path=None,
                      is_3d=False, 
                      wandb_name=None,
                      min_score=None,
                      max_score=None):
    """
    Plots phenotype points on top of centroids.
    Empty cells (NaNs in scores) are not plotted.
    """

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")  # Treat NaN as white

    if is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(*centroids.T, s=20, marker="x", c=[[0.5, 0.5, 0.5, 0.2]])
        sc = ax.scatter(
            *phenotypes.T, 
            c=scores, 
            cmap=cmap, 
            s=20, 
            marker=".", 
            vmin=min_score, 
            vmax=max_score
        )
        fig.colorbar(sc, ax=ax, label="Score")
        for i, (min_b, max_b) in enumerate(zip(behavior_space_bounds[0], behavior_space_bounds[1])):
            getattr(ax, f"set_{'xyz'[i]}lim")((min_b, max_b))
    else:
        plt.figure()
        plt.scatter(centroids[:, 0], centroids[:, 1], s=20, marker="x", c=[[0.5, 0.5, 0.5, 0.2]])
        sc = plt.scatter(
            phenotypes[:, 0], 
            phenotypes[:, 1], 
            c=scores, 
            cmap=cmap, 
            s=20, 
            marker=".", 
            vmin=min_score, 
            vmax=max_score
        )
        plt.colorbar(sc, label="Score")
        plt.xlim(behavior_space_bounds[0, 0], behavior_space_bounds[1, 0])
        plt.ylim(behavior_space_bounds[0, 1], behavior_space_bounds[1, 1])
        plt.title(title)

    if wandb_name:
        import wandb
        wandb.log({wandb_name: wandb.Image(plt)})
    if save_path:
        plt.savefig(save_path)
    plt.close()

from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

def plot_voronoi_score_grid(centroids, scores, bounds, title="Voronoi Grid", save_path=None, wandb_name=None, vmin=0, vmax=1):
    """
    Plot a 2D Voronoi diagram where each region is colored by score.

    Args:
        centroids (np.ndarray): (N, 2) array of centroid coordinates.
        scores (np.ndarray): (N,) array of scores, with NaN for unfilled niches.
        bounds (np.ndarray): (2, 2) array of behavior space bounds (min and max).
        title (str): Plot title.
        save_path (str): Optional path to save the figure.
        wandb_name (str): Optional wandb image key.
        vmin (float): Minimum score for colormap.
        vmax (float): Maximum score for colormap.
    """

    # Crop to bounding box
    min_x, min_y = bounds[0]
    max_x, max_y = bounds[1]

    vor = Voronoi(centroids)
    patches = []
    patch_colors = []

    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not region or -1 in region:
            continue  # skip open (infinite) regions
        polygon = [vor.vertices[v] for v in region]
        poly = Polygon(polygon, closed=True)
        patches.append(poly)

        # Set color based on score
        color = scores[i]
        patch_colors.append(color)

    # Create color map
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array(patch_colors)
    face_colors = cmap(norm(colors))

    # Replace NaN with white
    face_colors[np.isnan(colors)] = [1, 1, 1, 1]

    # Plot
    fig, ax = plt.subplots()
    p = PatchCollection(patches, facecolors=face_colors, edgecolor='black', linewidth=0.2)
    ax.add_collection(p)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Score")

    if wandb_name:
        import wandb
        wandb.log({wandb_name: wandb.Image(plt)})

    if save_path:
        plt.savefig(save_path)

    plt.close()
