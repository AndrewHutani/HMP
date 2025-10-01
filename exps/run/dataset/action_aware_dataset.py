import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import re
from typing import Dict, List, Optional

class ActionAwareDataset(Dataset):
    def __init__(self, data_file: str, specific_action: Optional[str] = None):
        """
        Dataset that groups existing preprocessed data by action based on filenames
        
        Args:
            data_file: Path to the existing pickled data (e.g., 'h36m_test_50.pkl')
            specific_action: If specified, only load data from this action
        """
        # Load existing preprocessed data
        with open(data_file, 'rb') as f:
            self.all_data_paths = pickle.load(f)
        
        # Group by action based on filename patterns
        self.action_groups = self._group_by_action(self.all_data_paths)
        
        # Filter by specific action if requested
        if specific_action:
            if specific_action not in self.action_groups:
                available_actions = list(self.action_groups.keys())
                raise ValueError(f"Action '{specific_action}' not found. Available actions: {available_actions}")
            
            self.data_paths = self.action_groups[specific_action]
            self.action_labels = [specific_action] * len(self.data_paths)
        else:
            # Use all actions
            self.data_paths = []
            self.action_labels = []
            for action, paths in self.action_groups.items():
                self.data_paths.extend(paths)
                self.action_labels.extend([action] * len(paths))
        
        self._print_summary(specific_action)
    
    def _extract_action_from_path(self, file_path: str) -> str:
        """
        Extract action name from preprocessed file path
        
        Examples:
        - '/path/to/S1_waiting_1_vid_0.npy' -> 'waiting'
        - '/path/to/S5_walking_2_vid_50.npy' -> 'walking'
        """
        # Get the filename without extension
        filename = os.path.basename(file_path).replace('.npy', '')
        
        # Pattern: Subject_Action_Trial_vid_Frame
        # e.g., 'S1_waiting_1_vid_0' -> extract 'waiting'
        parts = filename.split('_')
        
        if len(parts) >= 4:
            # Remove subject (S1), trial number (1), 'vid', and frame number
            # Keep everything in between as the action
            subject_removed = parts[1:]  # Remove 'S1'
            
            # Find 'vid' and remove everything from 'vid' onwards
            vid_index = None
            for i, part in enumerate(subject_removed):
                if part == 'vid':
                    vid_index = i
                    break
            
            if vid_index is not None:
                action_parts = subject_removed[:vid_index]
                
                # Remove the last part if it's a number (trial number)
                if action_parts and action_parts[-1].isdigit():
                    action_parts = action_parts[:-1]
                
                action = '_'.join(action_parts)
            else:
                # Fallback: assume the action is the second part
                action = subject_removed[0] if subject_removed else 'unknown'
        else:
            action = 'unknown'
        
        return action
    
    def _group_by_action(self, data_paths: List[str]) -> Dict[str, List[str]]:
        """Group data paths by action"""
        action_groups = {}
        
        for path in data_paths:
            action = self._extract_action_from_path(path)
            
            if action not in action_groups:
                action_groups[action] = []
            action_groups[action].append(path)
        
        return action_groups
    
    def _print_summary(self, specific_action: Optional[str]):
        """Print dataset summary"""
        print(f"Loaded dataset with {len(self.data_paths)} samples")
        
        if specific_action:
            print(f"Filtered to action: {specific_action}")
        else:
            print("Action distribution:")
            action_counts = {}
            for action in self.action_labels:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            for action, count in sorted(action_counts.items()):
                print(f"  {action}: {count} samples")
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # Load the data (add .npy extension if missing)
        file_path = self.data_paths[idx]
        if not file_path.endswith('.npy'):
            file_path = file_path + '.npy'

        data = np.load(file_path)
        
        # Split into components
        q = data[:, :63]  # Joint angles and translation
        shape = data[:, 63:73]  # Shape parameters  
        gender_id = data[:, 73:74]  # Gender
        
        action = self.action_labels[idx]
        
        return {
            'q': torch.from_numpy(q).float(),
            'shape': torch.from_numpy(shape).float(), 
            'gender_id': torch.from_numpy(gender_id).float(),
            'action': action,
            'file_path': self.data_paths[idx]
        }
    
    def get_actions(self) -> List[str]:
        """Get list of all available actions"""
        return list(self.action_groups.keys())
    
    def get_action_count(self, action: str) -> int:
        """Get number of samples for a specific action"""
        return len(self.action_groups.get(action, []))

class ActionSampler:
    """Helper class to sample specific actions from the dataset"""
    
    def __init__(self, dataset: ActionAwareDataset):
        self.dataset = dataset
        self.action_indices = self._build_action_indices()
    
    def _build_action_indices(self) -> Dict[str, List[int]]:
        """Build mapping from action to sample indices"""
        action_indices = {}
        
        for i, action in enumerate(self.dataset.action_labels):
            if action not in action_indices:
                action_indices[action] = []
            action_indices[action].append(i)
        
        return action_indices
    
    def sample_action(self, action: str, num_samples: int = 1) -> List[Dict]:
        """Sample specific number of samples from an action"""
        if action not in self.action_indices:
            raise ValueError(f"Action '{action}' not found. Available: {list(self.action_indices.keys())}")
        
        import random
        available_indices = self.action_indices[action]
        sampled_indices = random.sample(available_indices, min(num_samples, len(available_indices)))
        
        return [self.dataset[idx] for idx in sampled_indices]
    
    def get_action_dataloader(self, action: str, batch_size: int = 1, shuffle: bool = True):
        """Create a DataLoader for a specific action"""
        from torch.utils.data import DataLoader, Subset
        
        if action not in self.action_indices:
            raise ValueError(f"Action '{action}' not found")
        
        action_subset = Subset(self.dataset, self.action_indices[action])
        return DataLoader(action_subset, batch_size=batch_size, shuffle=shuffle)