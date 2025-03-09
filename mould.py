from typing import Iterable, Callable, Any, Optional, List
from itertools import cycle, islice, zip_longest

class Mould:
    """
    Mould: A trainable, modulating iterator that shapes data flow in a structured framework.

    - Applies transformations during iteration.
    - Supports momentum-based training with adjustable learning rates.
    - Maintains linkage to a parent Mould for backpropagation.
    - Implements batch updates and gradient clipping for stability.
    """
    
    def __init__(
        self,
        *iterables: Iterable,
        func: Optional[Callable[..., Any]] = None,
        train_func: Optional[Callable[..., Any]] = None,
        length: Optional[int] = None,
        parent: Optional["Mould"] = None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        batch_size: int = 1,
        gradient_clip: Optional[float] = None
    ):
        self.iterables = [list(iterable) for iterable in iterables]
        self.func = func or (lambda *x: x)  # Default: identity function
        self.train_func = train_func or (lambda grad, item, lr: item - lr * grad)
        self.length = length or max(len(it) for it in self.iterables)  # Match longest iterable
        self.parent = parent or self.iterables[1] if len(self.iterables) > 1 and isinstance(self.iterables[1], Mould) else None # Link to previous Mould in the chain
        
        # Training parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        
        # Velocity for momentum-based updates
        self.velocity = [0.0 for _ in range(len(self.iterables))]
    
    def __iter__(self):
        """Forward pass: Apply func to each zipped set of items from the iterables."""
        cycled_iterables = [cycle(it) for it in self.iterables]  # Cycle shorter iterables
        zipped = zip(*(islice(it, self.length) for it in cycled_iterables))
        return (self.func(*items) for items in zipped)
    
    def train(self, gradients: List[Any]):
        """
        Backward pass: Applies gradients using momentum, learning rate, and optional gradient clipping.
        """
        if self.parent:
            self.parent.train(gradients)  # Backpropagate first
        
        num_batches = max(1, len(gradients) // self.batch_size)
        batched_gradients = [sum(gradients[i::num_batches]) / num_batches for i in range(num_batches)]
        
        if self.gradient_clip:
            batched_gradients = [max(min(g, self.gradient_clip), -self.gradient_clip) for g in batched_gradients]
        
        new_iterables = []
        for i, iterable in enumerate(self.iterables):
            new_values = []
            for j, item in enumerate(iterable):
                grad = batched_gradients[j % len(batched_gradients)]  # Cycle gradients
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
                new_values.append(self.train_func(grad, item, self.learning_rate))
            new_iterables.append(new_values)
        
        self.iterables = new_iterables
    
    def adjust_learning_rate(self, factor: float):
        """Dynamically adjust the learning rate."""
        self.learning_rate *= factor
