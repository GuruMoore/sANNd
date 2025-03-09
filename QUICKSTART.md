# 🚀 sANNd Quickstart Guide  

Welcome to **sANNd (sandbox artificial neural network design)**!  
This guide will show you how to **quickly build and train models** using `Mould`.  

---

## **1️⃣ Install sANNd**
```sh
git clone https://github.com/GuruMoore/sANNd.git
cd sANNd
```

---

## **2️⃣ The Basics: Moulds are the Building Blocks**  

In sANNd, networks are built from **Moulds**—trainable, iterable units that flow data and gradients.  
### **🔹 Example: Simple Forward Pass**
```python
from mould import Mould

# Define a transformation function
def scale(x, y):
    return x * y

# Create Moulds
input_data = [0.5]
weight = Mould([2.0])  # Initial weight
output = Mould(weight, input_data, func=scale)  # Multiply weight * input

print(list(output))  # Output: [1.0]
```
🔥 **Each Mould transforms data as it flows!**

---

## **3️⃣ Training a Simple Model**
Let’s train a **basic neural network** to match a target output.

```python
import random
import math
from mould import Mould

# Activation & Gradient Functions
def scale(x, y): return x * y
def add(x, y): return x + y
def softplus(x): return math.log1p(math.exp(min(x, 50)))
def compute_gradient(output, target): return [(o - t) * 0.01 for o, t in zip(output, target)]
def apply_gradient(grad, param, lr): return param - lr * grad

in_data = [0.5]  # Input data
target_output = [1.0348]  # Target output

# Initialize Moulds
hw = Mould([-random.uniform(1, 5)], in_data, func=scale, train_func=apply_gradient)  # Hidden layer weight
hb = Mould([0.0], hw, func=add, train_func=apply_gradient)  # Hidden layer bias
ha = Mould(hb, func=softplus, parent=hb) # Softplus activation

ow = Mould([-random.uniform(1, 5)], ha, func=scale, train_func=apply_gradient)  # Output weight
oc = Mould([sum(ow)], func=lambda x: x, parent=ow) # Sum hidden outputs
ob = Mould([0.0], oc, func=add, train_func=apply_gradient)  # Output bias

# Train the Model
for epoch in range(2000):
    # Forward Pass
    final_output = list(ob)  # Convert iterator to list

    # Compute Loss and Gradients
    loss = sum((o - t) ** 2 for o, t in zip(final_output, target_output)) / len(final_output)
    gradients = compute_gradient(final_output, target_output)

    # Backpropagation (Train Moulds)
    ob.train(gradients)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Output: {final_output}, Loss: {loss}")

# Final Prediction
print(f"\n Final Output: {final_output}, Target: {target_output}")
```
🔥 **A fully trainable network—no static layers, just flowing iterators!**

---

## **4️⃣ Residual Connections**
sANNd makes **skip connections trivial**:
```python
residual = Mould(prev_layer, transformed_layer, func=lambda h, f_h: h + f_h)
```
💡 **ResNets work naturally in sANNd!**

---

## **5️⃣ Building an LSTM**
LSTMs **flow naturally** with `Mould`—just chain cell states together.

```python
hidden, cell = Mould(input_data, prev_h, prev_c, w_h, w_x, w_c, b, func=lstm_cell)
```
🔥 **Recurrent models are just different `Mould` chains!**

---

## **6️⃣ Next Steps**
📚 **Explore More**:
- 🔗 **[Full sANNd Documentation](https://github.com/GuruMoore/sANNd)**
- 🚀 **[GitHub Issues: Contribute Ideas & Features](https://github.com/GuruMoore/sANNd/issues)**
- 💬 **Join the Discussion: [sANNd GitHub Discussions](https://github.com/GuruMoore/sANNd/discussions)**  

💡 **Start experimenting and shape the future of ML with sANNd!**

---
