import os
import datetime
import logging
import torch
import json
from datetime import datetime, timezone, timedelta
from utils.utils import autodetect_device
from utils.config import Config

logger = logging.getLogger(__name__)

class CheckpointManager:
    
    
    def __init__(self, config: Config):
        
        IST = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime.now(IST).strftime('%d-%m-%Y_%H-%M-%S')
        
        self.checkpoint_dir = os.path.join(config.checkpoint_dir, timestamp)
        
        
        logger.info(f"Creating checkpoint directory at {self.checkpoint_dir}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config.get_dict(), f, indent=4)
            
        logger.info(f"Initialized checkpoint manager")
        
    def save(self, checkpoint_name, checkpoint_data):
        torch.save(checkpoint_data, os.path.join(self.checkpoint_dir, checkpoint_name))
        
        
    @staticmethod
    def load(checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} not found")
    
        checkpoint_data = torch.load(checkpoint_path, weights_only=True, map_location=autodetect_device)
        
        return checkpoint_data