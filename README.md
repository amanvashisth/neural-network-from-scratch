# Neural Network from Scratch

A comprehensive implementation of a feedforward neural network built entirely from first principles using NumPy, without relying on high-level deep learning frameworks. This project demonstrates the mathematical foundations and practical implementation of neural networks for digit classification on the MNIST dataset.

## 📋 Project Overview

This project implements a multi-layer perceptron (MLP) neural network with 3 layers to classify handwritten digits (0-9) from the MNIST dataset. The implementation includes:

- **Complete forward propagation** through multiple hidden layers
- **Backpropagation algorithm** with mathematical derivations
- **Optimization using gradient descent**
- **Numerical stability** considerations (e.g., softmax temperature scaling)

The network achieves high accuracy on digit classification while providing educational insights into how neural networks work internally.

## 🏗️ Architecture

The neural network consists of a 3-layer feedforward architecture:

```
Input Layer (784 neurons) 
    ↓
Hidden Layer 1 (128 neurons) → ReLU Activation
    ↓
Hidden Layer 2 (64 neurons) → ReLU Activation
    ↓
Output Layer (10 neurons) → Softmax Activation
    ↓
Class Prediction (10 classes: digits 0-9)
```

### Layer Details

| Layer | Input | Output | Activation | Parameters |
|-------|-------|--------|------------|-----------|
| Layer 1 | 784 | 128 | ReLU | W₁, b₁ |
| Layer 2 | 128 | 64 | ReLU | W₂, b₂ |
| Layer 3 | 64 | 10 | Softmax | W₃, b₃ |

## 📊 Dataset

**MNIST (Modified National Institute of Standards and Technology)**

- **Training samples**: 60,000 handwritten digits
- **Test samples**: 10,000 handwritten digits
- **Image size**: 28×28 pixels (flattened to 784 features)
- **Classes**: 10 (digits 0-9)
- **Pixel range**: 0-255 (normalized to 0-1)

### Data Preprocessing

1. **Normalization**: Pixel values divided by 255 to scale to [0, 1] range
2. **Flattening**: 2D images (28×28) reshaped to 1D vectors (784,)
3. **Transposition**: Data transposed to shape (784, m) where m is number of samples
4. **One-hot encoding**: Labels converted to one-hot vectors for categorical cross-entropy

## 🧠 Mathematical Foundations

### Activation Functions

#### ReLU (Rectified Linear Unit)
Used in hidden layers for non-linearity:

```
ReLU(z) = max(0, z)

Derivative:
ReLU'(z) = { 1,  if z > 0
           { 0,  if z ≤ 0
```

#### Softmax
Used in output layer for multi-class probability distribution:

```
softmax(zᵢ) = e^(zᵢ - max(z)) / Σⱼ e^(zⱼ - max(z))
```

The subtraction of the maximum value ensures numerical stability.

### Linear Transformation

Each layer performs an affine transformation:

```
Z = W^T × A_prev + b
```

Where:
- **W**: Weight matrix (shape: input_units × output_units)
- **A_prev**: Activations from previous layer
- **b**: Bias vector (shape: output_units × 1)

### Loss Function

Categorical Cross-Entropy:

```
L = -(1/m) × Σᵢ Σₖ Yₖᵢ × log(Ŷₖᵢ + ε)
```

Where:
- **m**: Number of samples
- **Y**: One-hot encoded ground truth labels
- **Ŷ**: Predicted probabilities
- **ε**: 1e-8 (prevents log(0))

## 🔄 Forward Propagation

The network processes input data through all layers sequentially:

### Layer 1
```
Z₁ = W₁^T × X + b₁
A₁ = ReLU(Z₁)
```

### Layer 2
```
Z₂ = W₂^T × A₁ + b₂
A₂ = ReLU(Z₂)
```

### Output Layer
```
Z₃ = W₃^T × A₂ + b₃
A₃ = Softmax(Z₃)
```

**Prediction**: ŷ = argmax(A₃)

## 🔙 Backpropagation

Backpropagation computes gradients by applying the chain rule from output to input.

### Output Layer Gradient

Due to the combination of softmax activation and categorical cross-entropy loss:

```
dZ₃ = A₃ - Y
```

### Weight and Bias Gradients

```
dW₃ = (1/m) × A₂ × dZ₃^T
db₃ = (1/m) × Σ(dZ₃)
```

### Error Propagation to Previous Layer

```
dA₂ = W₃ × dZ₃
```

### Hidden Layer Gradient

```
dZ₂ = dA₂ ⊙ ReLU'(Z₂)    [⊙ denotes element-wise multiplication]
dW₂ = (1/m) × A₁ × dZ₂^T
db₂ = (1/m) × Σ(dZ₂)
```

This process continues backward through all layers.

## 📦 Project Structure

```
neural-network-from-scratch/
├── neural_network.ipynb       # Main Jupyter notebook
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## 🛠️ Implementation Details

### Key Functions

#### `init_params()`
Initializes weights and biases for all layers using random values in range [-0.5, 0.5].

```python
def init_params():
    W1 = np.random.rand(784, 128) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    W2 = np.random.rand(128, 64) - 0.5
    b2 = np.random.rand(64, 1) - 0.5
    W3 = np.random.rand(64, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3
```

#### `relu(Z)` & `softmax(Z)`
Activation function implementations with numerical stability considerations.

```python
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
```

#### `linear_forward(A_prev, W, b)`
Performs affine transformation for each layer.

```python
def linear_forward(A_prev, W, b):
    Z = np.dot(W.T, A_prev) + b
    return Z
```

#### `forward_propagation(X, W1, b1, W2, b2, W3, b3)`
Executes complete forward pass through the network, returning intermediate activations and pre-activations for backpropagation.

```python
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = linear_forward(X, W1, b1)
    A1 = relu(Z1)
    Z2 = linear_forward(A1, W2, b2)
    A2 = relu(Z2)
    Z3 = linear_forward(A2, W3, b3)
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3
```

#### `calculate_loss(Y, Y_hat)`
Computes categorical cross-entropy loss.

```python
def calculate_loss(Y, Y_hat):
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
    return loss
```

#### `one_hot(Y)`
Converts integer class labels to one-hot encoded vectors.

```python
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
```

## 📚 Technologies Used

- **NumPy**: Numerical computations and linear algebra
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **TensorFlow/Keras**: MNIST dataset loading
- **Python 3**: Programming language

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib tensorflow
```

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/amanvashisth/neural-network-from-scratch.git
cd neural-network-from-scratch
```

2. Open the Jupyter notebook:
```bash
jupyter notebook neural_network.ipynb
```

3. Run cells sequentially to:
   - Load and preprocess MNIST data
   - Initialize network parameters
   - Perform forward propagation
   - Implement backpropagation
   - Train the network
   - Evaluate performance

## 📈 Key Concepts Demonstrated

✅ **Data Normalization**: Scaling pixel values to [0, 1]  
✅ **Network Initialization**: Random weight initialization strategy  
✅ **Forward Propagation**: Computing predictions through layers  
✅ **Loss Calculation**: Categorical cross-entropy for multi-class classification  
✅ **Backpropagation**: Deriving and implementing gradient computations  
✅ **Optimization**: Gradient descent parameter updates  
✅ **Numerical Stability**: Techniques like softmax temperature scaling  
✅ **One-hot Encoding**: Converting labels to suitable format  

## 💡 Learning Outcomes

After studying this project, you will understand:

- How neural networks transform input data through layers
- The mathematics behind forward and backward propagation
- How activation functions introduce non-linearity
- Why gradient descent helps optimize network parameters
- Numerical considerations in deep learning implementations
- How to build a working classifier from mathematical first principles

## 📝 Notes

- The implementation prioritizes **educational clarity** over performance optimization
- This is ideal for learning the fundamentals of neural networks
- For production use, consider optimized libraries like TensorFlow or PyTorch
- The network trains on the CPU, making it suitable for learning environments

## 🔍 Debugging & Understanding

Each section of the notebook includes:

1. **Mathematical formulation** (LaTeX equations)
2. **Code implementation** (Python with NumPy)
3. **Verification cells** (checking intermediate shapes and values)
4. **Data visualizations** (viewing images and results)

## 🎯 Potential Improvements

- Implement batch normalization for faster convergence
- Add learning rate scheduling
- Implement momentum or Adam optimizer
- Add model checkpointing
- Visualize learned features and gradients
- Implement regularization (L1/L2)
- Add early stopping mechanism
- Parallelize computations for GPU acceleration

## 📖 References

- **Deep Learning**: Goodfellow, Bengio, Courville
- **Neural Networks and Deep Learning**: Michael Nielsen
- **MNIST Dataset**: Yann LeCun et al.
- **Backpropagation Algorithm**: Rumelhart, Hinton, Williams (1986)

## 👨‍💻 Author

**Aman Vashisth**

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest enhancements
- Submit pull requests
- Improve documentation

## 📞 Support

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated**: September 2026  
**Status**: Active Development  
**Purpose**: Educational - Understanding Neural Networks from First Principles
