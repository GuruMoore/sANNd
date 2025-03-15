from typing import Iterable, Callable, Any, Optional, List
from itertools import cycle, islice

class Mould:
    """
    Mould: A trainable, modulating iterator that applies transformations lazily.
    
    - Wraps one or more iterables.
    - Applies a function dynamically during iteration.
    - Ensures shorter iterables cycle to match the longest one.
    - Supports backpropagation-style updates via `train()`.
    """
    
    def __init__(
        self,
        *iterables: Iterable,
        func: Optional[Callable[..., Any]] = None,
        train: bool = True,
        train_func: Optional[Callable[..., Any]] = None,
        parent: Optional["Mould"] = None,
        learning_rate: float = 0.5,
        momentum: float = 1.5,
        batch_size: int = 1,
        gradient_clip: Optional[float] = None
    ):
        self.iterables: List[List[Any]] = [*iterables]  # Store as lists for cycling
        if not self.iterables:
            raise ValueError("At least one iterable must be provided.")
        
        self.func = func or (lambda x: x)  # Transformation function

        if train:
            self.train_func = train_func or (lambda grad, item, lr: item - lr * grad)

            # Training parameters
            self.parent = parent  # Parent Mould for backpropagation --not needed? auto bp in train method
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.batch_size = batch_size
            self.gradient_clip = gradient_clip
            self.velocity: List[float] = [] # Velocity for momentum updates
        
        self._iterator = None  # Internal iterator state

    def __iter__(self):
        self._iterator = self.form()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = self.form()
        return next(self._iterator)
    
    def __len__(self):
        return len(self.iterables)
    
    def form(self):
        # Update output size
        self.output_dim = max(len(it) for it in self.iterables)
        if self.output_dim == 0:
            raise ValueError("Iterables cannot be empty.")

        # Match longest iterable length to cycle shorter ones
        cycled_iterables = [cycle(it) for it in self.iterables]
        for items in zip(*(islice(it, self.output_dim) for it in cycled_iterables)):
            yield self.func(*items)
    
    def train(self, gradients: List[Any]):
        """
        Backpropagates the gradients to update the underlying iterables.
        Applies learning rate, momentum, and optional gradient clipping.
        """
        if hasattr(self, "train_func"):
            # Ensure velocity list is ready for training
            self.velocity.extend([0.0] * (self.output_dim - len(self.velocity)))

        num_batches = max(1, len(gradients) // self.batch_size)
        batched_gradients = [sum(gradients[i::num_batches]) / num_batches for i in range(num_batches)]
        
        if self.gradient_clip:
            batched_gradients = [max(min(g, self.gradient_clip), -self.gradient_clip) for g in batched_gradients]
        
        back_prop = []
        new_iterables = []
        iter_idx = 0
        for iterable in self.iterables:
            if isinstance(iterable, Mould):
                if hasattr(iterable, "train_func"):
                    back_prop.append(iterable)
                new_values = iterable
            else:
                new_values = []
                for j, item in enumerate(iterable):
                    if isinstance(item, Mould):
                        if hasattr(item, "train_func"):
                            back_prop.append(item)
                        new_values.append(item)
                    elif hasattr(self, "train_func"):
                        grad = batched_gradients[j % len(batched_gradients)]  # Cycle gradients
                        self.velocity[iter_idx] = self.momentum * self.velocity[iter_idx] - self.learning_rate * grad
                        new_values.append(self.train_func(grad, item, self.learning_rate))
                    else:
                        new_values.append(item)
            
            new_iterables.append(new_values)
            iter_idx += 1
        
        self.iterables = new_iterables  # Store updated iterables

        for bp_iter in back_prop:
            bp_iter.train(gradients)  # Backpropagate

    def adjust_learning_rate(self, factor: float):
        """Dynamically adjusts the learning rate."""
        self.learning_rate *= factor

    def __repr__(self):
        return f"Mould(iterables={self.iterables}, func={self.func})"
