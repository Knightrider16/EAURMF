from dataclasses import dataclass, asdict


@dataclass
class Config:
    
    description: str = ""
    checkpoint_dir: str = "checkpoints"
    
    
    def get_dict(self):
        return asdict(self)