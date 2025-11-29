# Predicting Effective Young's Modulus with Neural Networks from Parametric FEA

This project automates the prediction of the **effective elastic properties** of a perforated 2D plate using a hybrid Finite Element Analysis (FEA) and Machine Learning approach.

The goal is to train a Feedforward Neural Network (FNN) to predict the **effective Young's modulus** ($E_{\text{eff}}$) of a plate with two random elliptical holes directly from geometric parameters, bypassing computationally expensive FEM simulations for new geometries.

---

## Project Status

- [x] **Phase 1:** Parametric mesh generation with GMSH (Completed)
- [ ] **Phase 2:** FEM simulation & Homogenization with CalculiX (In Progress)
- [ ] **Phase 3:** Dataset assembly & Neural Network training (Planned)

---

## Project Pipeline

The workflow integrates three main stages:

### 1. Parametric Mesh Generation (GMSH)

Automated generation of 2D rectangular plates with two elliptical holes.

**Geometric Parameters (per hole):**
- Position: $x, y$ (center coordinates)
- Semi-axes: $r_x, r_y$ (radii)
- Orientation: $\theta$ (rotation angle)

Each configuration is randomized and stored with an associated mesh file (.msh).

### 2. FEM Simulation (CalculiX)

For each geometry, CalculiX solves the linear elastic problem:

**Problem Setup:**
- Material: Isotropic, constant $E_{\text{mat}}$, constant Poisson's ratio $\nu$
- Boundary Condition: Prescribed displacement $u$ on plate edges
- Solver: Assembles global stiffness matrix $\mathbf{K}$, solves $\mathbf{K}\mathbf{u} = \mathbf{F}$

From the displacement field, stresses and strains are evaluated at Gauss integration points across all elements.

### 3. Homogenization & Data Extraction

**Volume Averaging:**
- Average stress: $\bar{\boldsymbol{\sigma}} = \frac{1}{V}\int_V \boldsymbol{\sigma} \, dV$
- Average strain: $\bar{\boldsymbol{\varepsilon}} = \frac{1}{V}\int_V \boldsymbol{\varepsilon} \, dV$

**Constitutive Identification:**
The effective stiffness matrix is identified from:

$$\bar{\boldsymbol{\sigma}} = \mathbf{C}_{\text{eff}} \cdot \bar{\boldsymbol{\varepsilon}}$$

The effective Young's modulus $E_{\text{eff}}$ is extracted from $\mathbf{C}_{\text{eff}}$.

### 4. Neural Network Training

**Input Feature Vector** (10-dimensional):

$$\mathbf{x} = [x_1, y_1, r_{x1}, r_{y1}, \theta_1, x_2, y_2, r_{x2}, r_{y2}, \theta_2]$$

**Target Output:**

$$E_{\text{eff}}$$

A regression FNN learns the mapping from geometry to effective Young's modulus. The model is trained on a dataset of approximately 5000 FEM simulations.

---

## Dataset & Training

- **Data Set (target):** ~5000 samples
- **Input:** 10-dimensional geometric feature vector
- **Output:** Single scalar $E_{\text{eff}}$
- **Train/Test Split:** (to be defined)
- **Baseline:** Comparison against direct FEM evaluation


## Mathematical Foundations

### Global Stiffness Matrix ($\mathbf{K}$)

The global stiffness matrix is the assembled collection of all element stiffness matrices. Its role:
- Relates applied forces to resulting displacements: $\mathbf{F} = \mathbf{K} \mathbf{u}$
- Encodes the full mechanical response of the structure
- Boundary conditions are enforced by modifying rows/columns corresponding to constrained DOFs
- Once assembled and boundary conditions applied, solving $\mathbf{K} \mathbf{u} = \mathbf{F}$ yields the full displacement solution

### From Element to Global: FEM Pipeline

1. **Element Level:** Each triangular element computes $\mathbf{K}^{(e)}$ using strain-displacement matrix $\mathbf{B}^{(e)}$ and material stiffness $\mathbf{C}$.
2. **Global Assembly:** All element matrices are summed into $\mathbf{K}$ at global DOF locations.
3. **Solution:** With boundary conditions applied, the system is solved for $\mathbf{u}$.
4. **Post-Processing:** Strains $\boldsymbol{\varepsilon} = \mathbf{B} \mathbf{u}$ and stresses $\boldsymbol{\sigma} = \mathbf{C} \boldsymbol{\varepsilon}$ are computed at Gauss points.
5. **Homogenization:** Volume-averaged stresses and strains feed into the constitutive identification.

---

## Next Steps

- [ ] Implement batch execution of CalculiX simulations.
- [ ] Develop post-processing script for volume averaging (Homogenization).
- [ ] Parallelization for faster mesh/FEM generation.
- [ ] Validation against FEM ground truth (MSE/R² metrics).
- [ ] Jupyter notebook example for reproducibility.

---

## Requirements
- **Python 3.x**
- **Gmsh SDK** (Mesh generation)
- **CalculiX (ccx)** (FEM Solver)
- **NumPy & Pandas** (Data handling)
- **PyTorch** (Planned for ML)

---

## Author

**Andrea Vinarš**  
Email: [andrea.vinars3@gmail.com]

---

## License

MIT License



