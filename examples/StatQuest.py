from mould import Mould
import math

"""
    Reproduce simple network example demonstrated in StatQuest with Josh Starmer:
        https://www.youtube.com/watch?v=CqOfi41LfDw
"""

# Callables
def scale(x, y):
    print("scale", x, y)
    return x * y

def add(x, y):
    print("add", x, y)
    return x + y

def softplus(x):
    print("softplus", x)
    return math.log1p(math.exp(x))

def softplus_derivative(x):
    return 1 / (1 + math.exp(-x))

# Bee - beep - boop - bee - boop
input = [0.5] 
hw = Mould([-34.4, -2.52], input, func=scale) # Weights of connection from input to the hidden layer
hb = Mould([2.14, 1.29], hw, func=add) # Biases of the Nodes in the hidden layer
ha = Mould(hb, func=softplus) # Softplus activation

ow = Mould([-1.30, 2.28], ha, func=scale) # Weights to output layer
oc = Mould([sum(ow)], func=lambda x: x) # Sum hidden outputs
ob = Mould([-0.58], oc, func=add) # Apply output bias

# Bam!
print(list(ob))
