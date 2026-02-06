# Predicting Effective Young's Modulus with Neural Networks from Parametric FEA

This project automates the prediction of the **effective elastic properties** of a perforated 2D plate using a hybrid Finite Element Analysis (FEA) and Machine Learning approach.

The goal is to train a Feedforward Neural Network (FNN) to predict the **effective Young's modulus** ($E_{\text{eff}}$) of a plate with two random elliptical holes directly from geometric parameters, bypassing computationally expensive FEM simulations for new geometries.

---

## Project Status

- [x] **Phase 1:** Parametric mesh generation with GMSH (Completed)
- [x] **Phase 2:** FEM simulation & Homogenization with CalculiX (Completed)
- [ ] **Phase 3:** Neural Network training (In progress)
- [ ] **Phase 4:** Results evaluation

---

## Project Pipeline

The workflow integrates three main stages:

### 1. Parametric Mesh Generation (GMSH)

Automated generation of 2D rectangular plates with two elliptical holes.

**Geometric Parameters (per hole):**
- Position: $x, y$ (center coordinates)
- Semi-axes: $r_x, r_y$ (semi-major and semi-minor axis lengths)
- Orientation: $\theta$ (rotation angle)

Each configuration is randomized and stored with an associated mesh file (.msh).

### 2. FEM Simulation (CalculiX)

For each geometry, CalculiX solves the linear elastic problem:

**Problem Setup:**
- Material: Isotropic, constant $E_{\text{mat}}$, constant Poisson's ratio $\nu$
- Boundary Condition: Prescribed displacement $u$ on plate edges
- Solver: Assembles global stiffness matrix $\mathbf{K}$ from element matrices and solves the linear system $\mathbf{K}\mathbf{u} = \mathbf{F}$ for nodal displacements $\mathbf{u}$.

From the displacement field, stresses and strains are evaluated at Gauss integration points across all elements.

### 3. Homogenization & Data Extraction

**Volume Averaging:**
- Average stress: $\bar{\boldsymbol{\sigma}} = \frac{1}{V}\int_V \boldsymbol{\sigma} \, dV$
- Average strain: $\bar{\boldsymbol{\varepsilon}} = \frac{1}{V}\int_V \boldsymbol{\varepsilon} \, dV$

**Effective Young's Modulus Calculation:**

The effective modulus is obtained from the ratio of homogenized stress to applied strain:

$$E_{eff} = \frac{\bar{\sigma}_{xx}}{\bar{\varepsilon}_{xx}}$$

**Homogenized Stress** (area-weighted average):

$$\bar{\sigma}_{xx} = \frac{\sum_{i} \sigma_{xx,i} \cdot A_i}{\sum_{i} A_i}$$

Where:
- $\sigma_{xx,i}$ = Axial stress of element $i$ (from CalculiX `.dat` file)
- $A_i$ = Area of element $i$ (computed from nodal coordinates in `.inp` file)

**Applied Strain** (prescribed boundary condition):

$$\bar{\varepsilon}_{xx} = \frac{\text{elongation}}{\text{plate width}}$$

**Implementation:**
The `calculate_youngs_modulus()` function parses the mesh and stress files, computes element areas using the cross-product formula, performs area-weighted averaging of Sxx stresses, and divides by the applied strain to obtain $E_{eff}$.

**Assumptions:**
- Linear elastic regime (small strains)
- Plane stress conditions  
- Homogenization: Effective modulus links average stress to applied (nominal) strain
- Uniform boundary displacement (prescribed elongation)

### 4. Neural Network Training

**Input Feature Vector** (10-dimensional):

$$\mathbf{x} = [x_1, y_1, r_{x1}, r_{y1}, \theta_1, x_2, y_2, r_{x2}, r_{y2}, \theta_2]$$

**Target Output:**

$$E_{\text{eff}}$$

A regression FNN learns the mapping from geometry to effective Young's modulus. The model is trained on a dataset of approximately 5000 FEM simulations.

---

## Dataset & Training

- **Target dataset size:** ~5000 samples
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

- [ ] Train NN
- [ ] Validation against FEM ground truth (MSE/R² metrics).
- [ ] Jupyter notebook example for reproducibility.

---

## Requirements
- **Python 3.x**
- **Gmsh 4.15.0** (Mesh generation)
- **CalculiX 2.2x (ccx)** (FEM Solver)
- **NumPy & Pandas** (Data handling)
- **PyTorch** (Planned for ML)

---

## Author

**Andrea Vinarš**  
Email: andrea.vinars3@gmail.com

---

## License

MIT License



