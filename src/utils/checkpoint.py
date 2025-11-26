import os
import datetime
import logging
import torch
import json
from datetime import datetime, timezone, timedelta
from src.utils.utils import autodetect_device
from src.utils.config import Config
from pathlib import Path

logger = logging.getLogger(__name__)

class CheckpointManager:
    
    
    def __init__(self, config: Config):
        
        IST = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime.now(IST).strftime('%d-%m-%Y_%H-%M-%S')
        
        self.checkpoint_dir = os.path.join(config.checkpoint_dir, timestamp)
        config.checkpoint_dir = self.checkpoint_dir
        
        
        logger.info(f"Creating checkpoint directory at {self.checkpoint_dir}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config.get_dict(), f, indent=4)
            
        logger.info(f"Initialized checkpoint manager")
        
    def save(self, checkpoint_name, checkpoint_data):
        torch.save(checkpoint_data, os.path.join(self.checkpoint_dir, checkpoint_name))
        
        
    @staticmethod
    def load(checkpoint_dir, file_name=None, metrics_only=False):
        device = autodetect_device()
        checkpoint_data = None
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
        
        if file_name:
            file_path = os.path.join(checkpoint_dir, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Checkpoint file {file_path} not found")
        
    
                checkpoint_data = torch.load(file_path, weights_only=True, map_location=device)

                if metrics_only:
                    checkpoint_data = checkpoint_data['metrics']
        else:
            checkpoint_data = []
            folder = Path(checkpoint_dir)
            files = [p for p in folder.iterdir() if p.suffix in [".pt", ".pth"]]
                
            for i, file in enumerate(files):
                print(f"Loading checkpoint no. {i + 1}")
                data = torch.load(file, weights_only=False, map_location=device)
                
                if metrics_only:
                    data = data['metrics']
                checkpoint_data.append(data)
        
                
        return checkpoint_data
    
