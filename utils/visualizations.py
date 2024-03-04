from typing import List
import matplotlib.pyplot as plt
import numpy as np
import os


def visual_results(previous: List, actual: List, predicted: List, start_point: int, end_point: int, save_path: str):
    """
    Visualize and save comparison plots of actual and predicted sequences.

    Args:
        previous: List of previous sequences.
        actual: List of actual sequences.
        predicted: List of predicted sequences.
        start_point: Starting index for visualization.
        end_point: Ending index for visualization.
        save_path: Path to save the visualization plots.

    Returns:
        None
    """

    with plt.style.context('seaborn-v0_8'):
        fig = plt.figure(figsize=(14, 6), dpi=300)
        layout = (1, 1)
        ax = plt.subplot2grid(layout, (0, 0))

        for i in range(start_point, end_point):
            prev_act = np.concatenate((previous[i][:, -1], actual[i][:, -1]), axis=0)
            prev_pred = np.concatenate((previous[i][:, -1], predicted[i][:, -1]), axis=0)

            plt.plot(prev_pred, label='Predicted')
            plt.plot(prev_act, label='Actual')

            plt.legend()
            plt.savefig(os.path.join(save_path, '{}.png'.format(i)))
            plt.clf()
        #    plt.show()
