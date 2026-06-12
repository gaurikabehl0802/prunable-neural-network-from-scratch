# Custom Algorithmic Optimization: Prunable Neural Network From Scratch

An empirical research project demonstrating the mathematical mechanics of **Model Compression** and **Magnitude-Based Weight Pruning** to build sustainable, energy-efficient machine learning architectures. 
This entire repository is implemented in raw Python using **NumPy**, bypassing heavy deep learning frameworks like PyTorch or TensorFlow to prove foundational algorithmic competencies.

---

## 🔬 Core Research Objective
Modern deep learning architectures scale exponentially, leading to severe memory bottlenecks and massive carbon footprints during hardware-level execution. 
This project designs a custom Multi-Layer Perceptron (MLP) containing an integrated, post-training structural pruning pipeline. 

By systematically dropping negligible weight metrics, this framework maps out the trade-off boundaries between categorical classification precision and downstream microsecond computational latency.

---

## 🧮 Mathematical Framework & Implementation

### 1. Forward Propagation Telemetry
Data flows through the linear and non-linear layer transformations using optimized matrix operations. To enforce structural compression barriers, an element-wise binary mask matrix ($M$) intercepts the standard parameter tensors using the Hadamard product ($\odot$):

$$Z^{[1]} = X \cdot (W^{[1]} \odot M^{[1]}) + b^{[1]}$$
$$A^{[1]} = \text{ReLU}(Z^{[1]})$$

$$Z^{[2]} = A^{[1]} \cdot (W^{[2]} \odot M^{[2]}) + b^{[2]}$$
$$A^{[2]} = \text{Softmax}(Z^{[2]})$$

### 2. Manual Backpropagation via the Chain Rule
To calculate optimization gradients without automatic differentiation engines, partial derivatives are mapped out manually across matrix dimensions. The gradient calculations for our hidden layer weights ($dW^{[1]}$) strictly apply the structural masks to ensure that connections zeroed out by pruning receive zero active gradient updates:

$$dZ^{[2]} = A^{[2]} - Y_{\text{true}}$$

$$dW^{[2]} = \frac{1}{m} (A^{[1]})^T \cdot dZ^{[2]}$$

$$dA^{[1]} = dZ^{[2]} \cdot (W^{[2]} \odot M^{[2]})^T$$

$$dZ^{[1]} = dA^{[1]} \odot \text{ReLU}'(Z^{[1]})$$

$$dW^{[1]} = \frac{1}{m} X^T \cdot dZ^{[1]} \odot M^{[1]}$$

---

## 🟢 The Magnitude-Based Pruning Mechanism
The philosophy of weight pruning rests on the concept of parameter redundancy. Many weights inside trained networks hover close to absolute zero and contribute marginally to the final inference output. 



### Structural Optimization Protocol:
1. **Dense Training:** The model initializes weights using **He Normal Initialization** and trains to convergence under an uninhibited $0\%$ sparsity constraint.
2. **Global Absolute Sorting:** All system matrices are flattened and combined into a singular absolute magnitude distribution array:
   $$\text{Weights}_{\text{flattened}} = \text{sort}(|W^{[1]}| \cup |W^{[2]}|)$$
3. **Threshold Masking:** Given a target sparsity ratio $\kappa \in [0, 1]$, a threshold value is extracted at the $\kappa \times 100\text{th}$ percentile. Tensors are updated according to the condition:
   $$M_{i,j} = \begin{cases} 1 & \text{if } |W_{i,j}| > \text{Threshold} \\ 0 & \text{otherwise} \end{cases}$$

---

## 📊 Empirical Findings & Hardware Analysis
The performance evaluation pipeline benchmarks network metrics across isolated sparsity thresholds ($0\% \rightarrow 90\%$) using the Scikit-Learn Digits classification dataset.

### Key Observations:
* **The Accuracy Plateau:** Model classification accuracy remains highly stable even when stripping away up to $40\%$ of the network's internal connections. This confirms massive algorithmic redundancy, verifying that model compression can drastically reduce parameter size without destroying task performance.
* **The Latency Profile:** In generic CPU environments, dense matrix layouts populated with zero parameters do not automatically speed up processing times. Standard hardware registers calculate zero multiplications at identical speeds to non-zero values.
* **Bridging to Research Realities:** To convert mathematical sparsity into real-world energy and power savings ($Joules$), deployments utilize **Sparse Linear Algebra Kernels** or specialized hardware execution cores (such as Google TPUs or NVIDIA Sparse Tensor Cores). These architectures completely skip calculating operations when encountering a `0` in a mask matrix ($M$), minimizing processor switching activity, lowering memory bandwidth strain, and unlocking true hardware-level energy efficiency.

---

## 🛠️ Repository Directory & Setup

### File Structure
* `mlp.py`: The core object-oriented engine containing initialization states, mathematical activation primitives, forward/backward loops, and the pruning logic.
* `benchmark.py`: The evaluation pipeline that handles data preparation, model training, performance tracking, microsecond latency profiling, and graph generation.

### Execution Instructions
To run this project inside a standard local environment or Jupyter setup, clone this repository and execute the driver file:

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/Prunable-Neural-Network-From-Scratch.git](https://github.com/YOUR_USERNAME/Prunable-Neural-Network-From-Scratch.git)
cd Prunable-Neural-Network-From-Scratch

# Run the training, pruning, and benchmarking pipeline
python benchmark.py
