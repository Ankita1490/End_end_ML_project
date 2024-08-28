from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class Serialization(ABC):
    @abstractmethod
    def load(self, file_path:str):
        """Load file from the artifact folder"""
        pass
    
    @abstractmethod
    def serialize(self, data:Any, file_path:str):
        """Serialize any type of data in to the artifacts folder"""

