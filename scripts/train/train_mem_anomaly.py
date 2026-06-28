import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llava.train.train_anomaly import train

if __name__ == "__main__":
    train()
