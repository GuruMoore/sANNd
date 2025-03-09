# 🏗️ sANNd: The Sandbox for Neural Networks 🏖️  

### **A novel, iterator-based approach to deep learning**  
sANNd (**sandbox artificial neural network design**) is an **open-source machine learning framework** that takes a radically different approach:  
- **No computational graphs**  
- **No tensors**  
- **No rigid layer structures**  
Instead, sANNd is built **entirely on trainable iterators** that **flow data and gradients** through the network, just like sand through a child's hands in a sandbox.  

🔥 **sANNd lets you build and train deep learning models with simple, composable `Mould` units.**  

---

## **🚀 Why sANNd?**
✅ **Iterator-Based Learning** → No static graphs, just **flowing iterables**.  
✅ **Flexible & Modular** → Networks **compose like LEGO**, making **residuals, LSTMs, and CNNs trivial**.  
✅ **Efficient Backpropagation** → Uses **parent-linked `Moulds`** to propagate gradients **automatically**.  
✅ **Lightweight & Fast** → No deep dependency trees, just **pure Python and NumPy (soon JAX/CUDA support!)**.  

> _Imagine neural networks built like a child's sandbox:  
> The **buckets, sifters, and dump trucks** are your transformations,  
> The **sand grains** are your iterables,  
> The **pivoting hourglass** is your training pipeline._  

💡 **sANNd is an experimental playground for AI research.**  

---

## **🛠️ Install sANNd**
```sh
git clone https://github.com/GuruMoore/sANNd.git
cd sANNd
```

---

## **🎨 Example: A Simple Neural Network**
A **basic neural network** in sANNd is just a **chain of `Moulds`** that **modulate data flow**.  
Here’s how you can **define and train a simple model**:

```python
import random
import math
from mould import Mould

# Define activation functions
def scale(x, y):
    return x * y

def add(x, y):
    return x + y

def softplus(x):
    return math.log1p(math.exp(min(x, 50)))  # Prevent overflow

# Gradient functions
def compute_gradient(output, target):
    return [(o - t) * 0.01 for o, t in zip(output, target)]  # Simple derivative

def apply_gradient(grad, param, lr):
    return param - lr * grad  # Learning rate-based update

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

    if loss == 0: break

# Final Prediction
print(f"\n ({epoch}) Final Output: {final_output}, Target: {target_output}")
```
🔥 **This is a fully functional trainable model—built without traditional layers!**  

---

## **📌 Residual Networks (ResNet) in One Line**
In sANNd, residual connections **are just `Moulds` with additive identity connections**:
```python
residual = Mould(prev_layer, transformed_layer, func=lambda h, f_h: h + f_h)
```
💡 **Residuals, skip connections, and complex architectures are now trivial!**

---

## **📌 LSTMs Are Just Recurrent `Moulds`**
Unlike other ML frameworks, **LSTMs don’t require special treatment**.  
Just define your **memory cell as a `Mould`**, and everything **flows naturally**.
```python
hidden, cell = Mould(input_data, prev_h, prev_c, w_h, w_x, w_c, b, func=lstm_cell)
```
💡 **Recurrent models are as simple as stacking Moulds!**

---

## **🤝 Contributing**
We **welcome contributions** to improve sANNd!  
1️⃣ **Fork the repo**  
2️⃣ **Create a new branch** (`feature-xyz`)  
3️⃣ **Submit a pull request** 🚀  

🔥 **Join the discussion!** Open an issue or start a GitHub Discussion.  

---

## **📚 Roadmap**
🛠 **Upcoming Features**:
- ✅ **Multi-Layer Architectures** (MLPs, CNNs)  
- ✅ **Gradient Clipping & Adaptive Learning Rates**  
- ✅ **JAX/CUDA Acceleration**  
- 🚀 **Transformer Support**  
- 🚀 **Meta-Learning with Differentiable Programming**  

> **🌍 Let's push AI research forward—together.**  
> If **information isn't free, then neither are we.**  

---

## **📜 License**
**MIT License** – Free to use, modify, and share.  

---

## **🌍 Join the sANNd Community**
📢 **Share your experiments, insights, and ideas!**  
💬 Twitter, Reddit, Hacker News, Dev.to, Medium  
🚀 **Let’s build something amazing together!**  

---