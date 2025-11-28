# Predicting Effective Young's Modulus with Neural Networks from Parametric FEA

This project automates the prediction of the **effective elastic properties** of a perforated 2D plate using a hybrid Finite Element Analysis (FEA) and Machine Learning approach.

The goal is to train a Feedforward Neural Network (FNN) to predict the **effective Young's modulus ($E_{eff}$)** of a plate with two random elliptical holes, bypassing computationally expensive FEM simulations for new geometries.

---

## Project Status

This project is currently under active development.

- [x] **Phase 1:** Parametric mesh generation with GMSH (Completed)
- [ ] **Phase 2:** FEM simulation & Homogenization with CalculiX (In Progress)
- [ ] **Phase 3:** Neural Network training & Validation (Planned)

---

## Methodology & Workflow

The pipeline integrates computational homogenization principles with deep learning:

1. **Parametric Meshing (GMSH):** 
   - Automated generation of 2D plates with two elliptical holes.
   - Parameters: Position ($x, y$), Semi-axes ($r_x, r_y$), and Rotation ($\theta$) for each hole are randomized.

2. **FEM Simulation (CalculiX):** 
   - **Boundary Conditions:** A prescribed displacement ($u$) is applied to the plate edges.
   - **Material Model:** Linear elastic isotropic material (Base $E_{mat}$, $\nu$).
   - **Solver:** Solves for the displacement field and stress distribution.

3. **Homogenization (Post-Processing):** 
   - Stresses are extracted from **Gauss integration points** across all elements.
   - **Average stress ($\bar{\sigma}$)** and **average strain ($\bar{\varepsilon}$)** are computed via volume integration over the domain ($V$).
   - The **effective Young's modulus ($E_{eff}$)** is derived from the constitutive relation of the equivalent homogeneous medium.

4. **Machine Learning:** 
   - A regression neural network maps the 10-dimensional geometric vector directly to $E_{eff}$.

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
Email: andrea.vinars3@gmail.com  

---

## License

MIT License


