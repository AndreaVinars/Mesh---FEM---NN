# Predicting Effective Young's Modulus with Neural Networks from Parametric FEA

This project automates the prediction of the **effective elastic properties** of a perforated 2D plate using a hybrid Finite Element Analysis (FEA) and Machine Learning approach.

The goal is to train a Feedforward Neural Network (FNN) to predict the **effective Young's modulus ($E_{\text{eff}}$)** of a plate with two random elliptical holes, bypassing computationally expensive FEM simulations for new geometries.

---

## Project Status

Project is currently under active development.

- [x] **Phase 1:** Parametric mesh generation with GMSH (Completed)
- [ ] **Phase 2:** FEM simulation & Homogenization with CalculiX (In Progress)
- [ ] **Phase 3:** Neural Network training & Validation (Planned)

---

## Methodology & Workflow

The pipeline integrates computational homogenization principles with deep learning:

### 1. Parametric Meshing (GMSH)
- Automated generation of 2D plates with two elliptical holes.
- Parameters: Position ($x, y$), Semi-axes ($r_x, r_y$), and Rotation ($\theta$) for each hole are randomized.

### 2. FEM Simulation (CalculiX)

**Boundary Conditions & Solver:**
- A prescribed displacement ($u$) is applied to the plate edges.
- Material Model: Linear elastic isotropic material (Base $E_{\text{mat}}$, $\nu$).
- CalculiX assembles the **Global Stiffness Matrix** ($\mathbf{K}$) from elemental stiffness matrices:

$$\mathbf{K} = \bigcup_{e=1}^{N_{\text{elem}}} \mathbf{K}^{(e)}$$

where each element contributes via the strain-displacement matrix ($\mathbf{B}$) and material constitutive matrix ($\mathbf{C}$):

$$\mathbf{K}^{(e)} = \int_{\Omega^{(e)}} \mathbf{B}^T \mathbf{C} \mathbf{B} \, d\Omega$$

The global system is solved: $\mathbf{K} \mathbf{u} = \mathbf{F}$, yielding the displacement field $\mathbf{u}$ and stresses at all integration points.

### 3. Homogenization (Post-Processing)

- Stresses are extracted from **Gauss integration points** across all elements.
- **Average stress ($\bar{\boldsymbol{\sigma}}$)** and **average strain ($\bar{\boldsymbol{\varepsilon}}$)** are computed via volume integration over the domain ($V$).
- The effective stiffness matrix $\mathbf{C}_{\text{eff}}$ is identified from the homogenized constitutive relation:

$$\bar{\boldsymbol{\sigma}} = \mathbf{C}_{\text{eff}} \cdot \bar{\boldsymbol{\varepsilon}}$$

from which the effective Young's modulus $E_{\text{eff}}$ is extracted.

### 4. Machine Learning

- A regression neural network maps the 10-dimensional geometric vector directly to $E_{\text{eff}}$.
- Training data consists of: (geometry parameters) → (FEM-derived $E_{\text{eff}}$).

---

## Example Workflow

1. **Generate dataset (meshes and parameter CSV):**
    ```
    python mesh_generator.py
    ```
2. **Run FEM for all meshes and build output CSV:**
    ```
    python fem_runner.py
    ```
3. **Build final input-output dataset:**
    ```
    python dataset_builder.py
    ```
4. **Train neural network:**
    ```
    python train_nn.py
    ```

---

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



