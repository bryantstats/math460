import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --------------------------
# Activation functions
# --------------------------
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def linear(x): return x

activations = {
    "ReLU": relu,
    "Sigmoid": sigmoid,
    "Tanh": tanh,
    "Linear": linear
}

# --------------------------
# Page layout
# --------------------------
st.set_page_config(page_title="Neural Network Visualizer", layout="wide")
st.title("🧠 Two-Layer Neural Network — Adjustable Connections")

st.markdown("""
Interactively explore a **two-hidden-layer neural network**.  
You can:
- Adjust **every connection’s weight** and **each neuron’s bias**
- Choose activation functions for each layer
- See both the **input–output curve** and **network graph with labeled weights**
""")

# --------------------------
# Sidebar configuration
# --------------------------
st.sidebar.header("Network Configuration")

# Layer 1 settings
st.sidebar.subheader("Layer 1 (Hidden 1)")
n1 = st.sidebar.slider("Neurons in Layer 1", 1, 3, 2)
act1_name = st.sidebar.selectbox("Activation for Layer 1", list(activations.keys()), key="act1")

# Layer 2 settings
st.sidebar.subheader("Layer 2 (Hidden 2)")
n2 = st.sidebar.slider("Neurons in Layer 2", 1, 3, 2)
act2_name = st.sidebar.selectbox("Activation for Layer 2", list(activations.keys()), key="act2")

# Output layer
st.sidebar.subheader("Output Layer")
act_out_name = st.sidebar.selectbox("Output Activation", list(activations.keys()), key="act_out")

# --------------------------
# Individual weights and biases
# --------------------------
st.sidebar.markdown("### 🔗 Connection Weights")

# Weights from input → layer1
w1 = {}
for i in range(n1):
    w1[i] = st.sidebar.slider(f"w₁{i+1} (x → h1_{i+1})", -5.0, 5.0, 1.0, 0.1)

# Biases layer1
b1 = {}
st.sidebar.markdown("### ⚙️ Biases Layer 1")
for i in range(n1):
    b1[i] = st.sidebar.slider(f"b₁{i+1}", -5.0, 5.0, 0.0, 0.1)

# Weights layer1 → layer2
w2 = {}
st.sidebar.markdown("### 🔗 Weights Layer 1 → Layer 2")
for i in range(n1):
    for j in range(n2):
        w2[(i, j)] = st.sidebar.slider(f"w₂{i+1}{j+1} (h1_{i+1} → h2_{j+1})", -5.0, 5.0, 1.0, 0.1)

# Biases layer2
b2 = {}
st.sidebar.markdown("### ⚙️ Biases Layer 2")
for j in range(n2):
    b2[j] = st.sidebar.slider(f"b₂{j+1}", -5.0, 5.0, 0.0, 0.1)

# Weights layer2 → output
w_out = {}
st.sidebar.markdown("### 🔗 Weights Layer 2 → Output")
for j in range(n2):
    w_out[j] = st.sidebar.slider(f"w₃{j+1} (h2_{j+1} → y)", -5.0, 5.0, 1.0, 0.1)

# Bias output
b_out = st.sidebar.slider("Bias (output layer)", -5.0, 5.0, 0.0, 0.1)

# --------------------------
# Forward propagation for plotting
# --------------------------
x = np.linspace(-5, 5, 200).reshape(-1, 1)
act1 = activations[act1_name]
act2 = activations[act2_name]
act_out = activations[act_out_name]

# Layer 1
a1 = np.zeros((x.shape[0], n1))
for i in range(n1):
    z = w1[i] * x + b1[i]
    a1[:, i] = act1(z).flatten()

# Layer 2
a2 = np.zeros((x.shape[0], n2))
for j in range(n2):
    z = sum(a1[:, i] * w2[(i, j)] for i in range(n1)) + b2[j]
    a2[:, j] = act2(z).flatten()

# Output
z_out = sum(a2[:, j] * w_out[j] for j in range(n2)) + b_out
y = act_out(z_out)

# --------------------------
# Plot input–output curve
# --------------------------
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(x, y, color="purple", linewidth=2)
ax1.set_title(f"Output Curve (Activation: {act_out_name})")
ax1.set_xlabel("Input x")
ax1.set_ylabel("Output y")
ax1.axhline(0, color="gray", linewidth=1)
ax1.axvline(0, color="gray", linewidth=1)
ax1.grid(True)

# --------------------------
# Plot network graph with weights
# --------------------------
fig2, ax2 = plt.subplots(figsize=(6, 4))
G = nx.DiGraph()

# Nodes
G.add_node("x")
for i in range(n1):
    G.add_node(f"h1_{i+1}")
for j in range(n2):
    G.add_node(f"h2_{j+1}")
G.add_node("y")

# Edges
for i in range(n1):
    G.add_edge("x", f"h1_{i+1}", weight=w1[i])
for i in range(n1):
    for j in range(n2):
        G.add_edge(f"h1_{i+1}", f"h2_{j+1}", weight=w2[(i, j)])
for j in range(n2):
    G.add_edge(f"h2_{j+1}", "y", weight=w_out[j])

# Node positions
pos = {"x": (-2, 0)}
for i in range(n1):
    pos[f"h1_{i+1}"] = (-1, i - n1 / 2)
for j in range(n2):
    pos[f"h2_{j+1}"] = (0, j - n2 / 2)
pos["y"] = (1.5, 0)

# Draw graph
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="#cce5ff", arrows=True, ax=ax2)
edge_labels = nx.get_edge_attributes(G, "weight")
edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax2)

ax2.set_title("Network Graph (Weights Labeled)")

# --------------------------
# Layout
# --------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Input–Output Curve")
    st.pyplot(fig1)
with col2:
    st.subheader("🧩 Neural Network Graph")
    st.pyplot(fig2)

# --------------------------
# Step-by-step numeric output
# --------------------------
st.subheader("🔢 Step-by-Step Calculation")
input_val = st.number_input("Enter a test input x:", value=1.0, step=0.1)

# Layer 1
a1_val = []
for i in range(n1):
    z = w1[i] * input_val + b1[i]
    a = act1(z)
    a1_val.append(a)
    st.write(f"h1_{i+1}: z = {w1[i]:.2f}×{input_val:.2f} + {b1[i]:.2f} = {z:.3f} → a₁ = {act1_name}(z) = {a:.3f}")

# Layer 2
a2_val = []
for j in range(n2):
    z = sum(a1_val[i] * w2[(i, j)] for i in range(n1)) + b2[j]
    a = act2(z)
    a2_val.append(a)
    st.write(f"h2_{j+1}: z = Σ(h1_i×w₂ij)+b₂ = {z:.3f} → a₂ = {act2_name}(z) = {a:.3f}")

# Output
z_out_val = sum(a2_val[j] * w_out[j] for j in range(n2)) + b_out
y_val = act_out(z_out_val)
st.write(f"Output: z = Σ(h2_j×w₃j)+b₃ = {z_out_val:.3f} → y = {act_out_name}(z) = {y_val:.3f}")

st.markdown("---")
st.caption("Developed for teaching: visualize each connection, activation, and numerical output in a two-layer neural network.")
