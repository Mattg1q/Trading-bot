import torch
import torch.nn as nn
from transformers import pipeline
import logging
import warnings

# Suppress some transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class SpatialLOBModel(nn.Module):
    """
    A Spatial Neural Network (SNN) in PyTorch with 1D-CNN layers 
    to capture liquidity clusters from 20x4 LOB tensor.
    """
    def __init__(self):
        super(SpatialLOBModel, self).__init__()
        # Input shape: (Batch, Channels, Length) -> (1, 4, 20)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # 32 channels * 20 sequence length = 640
        self.fc1 = nn.Linear(32 * 20, 64)
        self.relu3 = nn.ReLU()
        # Output: Trading signal (-1, 0, 1) prediction, continuous map to Tanh
        self.fc2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x is expected to be [batch, 20, 4]
        # Transpose to [batch, channels, length] for Conv1d -> [batch, 4, 20]
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x

class SentimentModel:
    """
    Sentiment Model: A transformers pipeline using ProsusAI/finbert 
    to score Finnhub headlines.
    """
    def __init__(self):
        logger.info("Loading FinBERT sentiment model...")
        try:
            self.pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            logger.info("FinBERT model loaded.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.pipeline = None

    def score_headlines(self, headlines):
        """
        Scores a list of headlines and returns an aggregate sentiment score.
        Positive: +1, Negative: -1, Neutral: 0
        """
        if not headlines or self.pipeline is None:
            return 0.0
            
        try:
            results = self.pipeline(headlines)
            score = 0.0
            for res in results:
                label = res['label']
                if label == 'positive':
                    score += 1.0
                elif label == 'negative':
                    score -= 1.0
            
            # Normalize score against the number of valid headlines
            avg_score = score / len(headlines)
            return avg_score
        except Exception as e:
            logger.error(f"Error scoring sentiment: {e}")
            return 0.0
