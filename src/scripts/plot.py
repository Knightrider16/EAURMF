import torch
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.checkpoint import CheckpointManager



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    args, remaining_args = parser.parse_known_args()

    assert remaining_args == [], remaining_args
    assert args.checkpoint_dir is not None, "checkpoint directory cannot be None"

    folder = Path(args.checkpoint_dir)
    plot_dir = f'plots/{"_".join(folder.parts[-2:])}'
    data = CheckpointManager.load(args.checkpoint_dir, metrics_only=True)

    os.makedirs(plot_dir, exist_ok=True)

    for metric in ['acc', 'f1', 'loss']:
        
        plt.clf()
        
        for mode in ['train', 'test', 'val']:
            
            metric_data = [dat[mode][metric] for dat in data]
            plt.plot(range(len(metric_data)), metric_data, label=mode)
            
        plt.title(f"{metric} metric")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, f"{metric}.jpg"), dpi=300)
            
