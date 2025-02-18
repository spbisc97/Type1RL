from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class LearningRateScheduler(BaseCallback):
    """
    Callback for scheduling learning rate based on training progress.
    Can use either linear or exponential decay.
    """
    def __init__(self, initial_lr=1e-4, min_lr=1e-5, decay_type='linear', total_timesteps=None, verbose=0):
        """
        Args:
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            decay_type: Type of decay ('linear' or 'exponential')
            total_timesteps: Total number of timesteps for training
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_type = decay_type
        self.total_timesteps = total_timesteps
        
    def _init_callback(self) -> None:
        """
        Initialize callback attributes before training starts
        """
        if self.total_timesteps is None:
            self.total_timesteps = self.model._total_timesteps
        
    def _on_step(self):
        # Calculate current progress
        progress = self.num_timesteps / self.total_timesteps
        
        # Update learning rate based on decay type
        if self.decay_type == 'linear':
            new_lr = self.initial_lr * (1 - progress) + self.min_lr * progress
        else:  # exponential
            decay = np.exp(-5 * progress)  # -5 controls decay speed
            new_lr = self.min_lr + (self.initial_lr - self.min_lr) * decay
            
        # Set the new learning rate
        self.model.learning_rate = new_lr
        
        if self.verbose > 0 and self.n_calls % 1000 == 0:  # Reduced logging frequency
            print(f"Learning rate set to {new_lr:.2e}")
        
        return True 