"""
Local Dataset Management
========================

Handles local data storage and buffering for on-device training.

Features:
- Memory-efficient circular buffer
- Disk-based overflow for large datasets
- Class balancing for non-IID mitigation
- Data augmentation hooks

Storage strategy:
1. Primary: Keep recent samples in RAM (configurable limit)
2. Overflow: Write older samples to disk (numpy files)
3. Training: Load from RAM first, then disk as needed
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Iterator
from collections import deque
import numpy as np


class LocalDataset:
    """
    Local dataset buffer for edge device training.

    Manages a rolling buffer of training samples with
    automatic disk overflow when memory limits are reached.

    Attributes:
        max_memory_samples: Max samples to keep in RAM
        overflow_dir: Directory for disk overflow
        samples: In-memory sample deque
    """

    def __init__(
        self,
        max_memory_samples: int = 1000,
        overflow_dir: str = "data/local_buffer",
        enable_disk_overflow: bool = True,
        balance_classes: bool = True
    ):
        """
        Initialize the local dataset.

        Args:
            max_memory_samples: Maximum samples in RAM
            overflow_dir: Directory for overflow files
            enable_disk_overflow: Whether to use disk storage
            balance_classes: Whether to balance classes when sampling
        """
        self.max_memory_samples = max_memory_samples
        self.overflow_dir = overflow_dir
        self.enable_disk_overflow = enable_disk_overflow
        self.balance_classes = balance_classes

        # In-memory buffer
        self.samples: deque = deque(maxlen=max_memory_samples)

        # Class tracking for balancing
        self.class_counts: Dict[str, int] = {}

        # Disk overflow tracking
        self.overflow_files: List[str] = []
        self.overflow_count = 0

        self.logger = logging.getLogger(__name__)

        # Create overflow directory
        if enable_disk_overflow:
            os.makedirs(overflow_dir, exist_ok=True)

    def add(self, sample: Dict) -> None:
        """
        Add a sample to the dataset.

        If memory is full and disk overflow is enabled,
        oldest samples are written to disk.

        Args:
            sample: Sample dict with 'spectrogram' and 'label'
        """
        # Update class counts
        label = sample.get('label', 'unknown')
        self.class_counts[label] = self.class_counts.get(label, 0) + 1

        # Add timestamp if not present
        if 'timestamp' not in sample:
            sample['timestamp'] = time.time()

        # Check if we need to overflow
        if len(self.samples) >= self.max_memory_samples:
            if self.enable_disk_overflow:
                self._overflow_to_disk()

        # Add to buffer
        self.samples.append(sample)

    def add_batch(self, samples: List[Dict]) -> None:
        """Add multiple samples."""
        for sample in samples:
            self.add(sample)

    def _overflow_to_disk(self, num_samples: int = 100) -> None:
        """
        Write oldest samples to disk.

        Args:
            num_samples: Number of samples to overflow
        """
        if not self.samples:
            return

        # Get samples to overflow
        overflow_samples = []
        for _ in range(min(num_samples, len(self.samples))):
            if self.samples:
                overflow_samples.append(self.samples.popleft())

        if not overflow_samples:
            return

        # Save to disk
        filename = f"overflow_{self.overflow_count:06d}.npz"
        filepath = os.path.join(self.overflow_dir, filename)

        # Separate data for efficient storage
        spectrograms = np.array([s['spectrogram'] for s in overflow_samples])
        labels = [s['label'] for s in overflow_samples]

        np.savez_compressed(
            filepath,
            spectrograms=spectrograms,
            labels=labels
        )

        self.overflow_files.append(filepath)
        self.overflow_count += 1

        self.logger.debug(
            f"Overflowed {len(overflow_samples)} samples to {filepath}"
        )

    def get_training_data(
        self,
        num_samples: Optional[int] = None,
        include_overflow: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data as numpy arrays.

        Args:
            num_samples: Max samples to return (None = all)
            include_overflow: Whether to include disk overflow

        Returns:
            Tuple of (X, y) where X is spectrograms, y is labels
        """
        # Collect samples
        all_samples = list(self.samples)

        # Add overflow samples if requested
        if include_overflow and self.overflow_files:
            for filepath in self.overflow_files:
                try:
                    data = np.load(filepath, allow_pickle=True)
                    for i in range(len(data['labels'])):
                        all_samples.append({
                            'spectrogram': data['spectrograms'][i],
                            'label': data['labels'][i]
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to load {filepath}: {e}")

        # Balance classes if enabled
        if self.balance_classes:
            all_samples = self._balance_samples(all_samples)

        # Limit samples if requested
        if num_samples and len(all_samples) > num_samples:
            indices = np.random.choice(
                len(all_samples), num_samples, replace=False
            )
            all_samples = [all_samples[i] for i in indices]

        # Convert to arrays
        if not all_samples:
            return np.array([]), np.array([])

        X = np.array([s['spectrogram'] for s in all_samples])
        labels = [s['label'] for s in all_samples]

        # Encode labels
        unique_labels = sorted(set(labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_to_idx[l] for l in labels])

        # Add channel dimension
        if X.ndim == 3:
            X = X[..., np.newaxis]

        return X, y

    def _balance_samples(self, samples: List[Dict]) -> List[Dict]:
        """
        Balance samples across classes.

        Uses undersampling to match minority class.
        """
        if not samples:
            return samples

        # Group by class
        class_samples: Dict[str, List[Dict]] = {}
        for s in samples:
            label = s['label']
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(s)

        # Find minimum class size
        min_count = min(len(v) for v in class_samples.values())

        # Undersample each class
        balanced = []
        for label, class_list in class_samples.items():
            if len(class_list) > min_count:
                indices = np.random.choice(
                    len(class_list), min_count, replace=False
                )
                balanced.extend([class_list[i] for i in indices])
            else:
                balanced.extend(class_list)

        return balanced

    def get_batch_generator(
        self,
        batch_size: int = 8,
        shuffle: bool = True
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generator for training batches.

        Memory efficient - generates batches on demand.

        Args:
            batch_size: Samples per batch
            shuffle: Whether to shuffle data

        Yields:
            Tuples of (X_batch, y_batch)
        """
        X, y = self.get_training_data()

        if shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

    def clear(self, include_disk: bool = True) -> int:
        """
        Clear all samples.

        Args:
            include_disk: Also delete disk overflow files

        Returns:
            Number of samples cleared
        """
        count = len(self.samples)
        self.samples.clear()
        self.class_counts.clear()

        if include_disk:
            for filepath in self.overflow_files:
                try:
                    os.remove(filepath)
                    count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {filepath}: {e}")
            self.overflow_files.clear()
            self.overflow_count = 0

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'memory_samples': len(self.samples),
            'overflow_files': len(self.overflow_files),
            'class_distribution': dict(self.class_counts),
            'total_samples': len(self.samples) + sum(
                1 for f in self.overflow_files
            ) * 100  # Estimate
        }

    def __len__(self) -> int:
        return len(self.samples)
