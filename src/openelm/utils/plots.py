import os
from typing import Dict, List

from matplotlib import pyplot as plt


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
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{fig_name}_{model_id}.png"))
    plt.show()