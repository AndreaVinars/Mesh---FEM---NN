# Predicting Young's Modulus with Feedforward Neural Networks from Parametric FEA

This project aims to train a neural network to predict the effective elastic modulus ($E_{eff}$) of a perforated plate based on hole geometry. Using Computational Homogenization principles, the effective stiffness is derived from the averaged stress-strain response calculated via FEM.

---

## Project Status

This project is currently under active development.

- [x] **Phase 1:** Parametric mesh generation with GMSH (Completed)
- [ ] **Phase 2:** FEM simulation integration with CalculiX (In Progress)
- [ ] **Phase 3:** Neural Network training & Validation (Planned)

---

## Project Workflow

1. **Mesh Generation:** Automated creation and meshing of 2D plates (fixed dimensions) with two randomly parameterized elliptical holes (position, axes, angle) using GMSH Python API.
2. **FEM Simulation:** Batch simulation of all geometries in CalculiX, applying a prescribed displacement to extract global reaction force for each sample.
3. **Dataset Assembly:** For each geometry, the input vector is the 10D set of ellipse parameters; the output is the computed Young's modulus.
4. **Neural Network Training:** Training a feedforward neural network (regression) to predict Young's modulus based only on ellipse parameters (geometry→modulus).
5. **Surrogate Prediction:** Fast evaluation of modulus for new geometries using the trained neural network, bypassing the need for full FEM.

---

## Features

- **Parametric 2D geometry:** Plate with two elliptical holes, with randomizable position, orientation, and size.
- **Mesh automation:** Scriptable generation of unique mesh samples (Target: 5000+ samples).
- **Data Pipeline:** Automated export of parameters to CSV, ready for ML workflows.
- **Surrogate Modeling:** Direct inference of Young's modulus using Deep Learning.

---

## Repository Structure

├── mesh_generator.py # Generates parametric meshes with ellipses
├── fem_runner.py # Runs CalculiX simulations, extracts force/displacement
├── dataset_builder.py # Assembles (input, output) table
├── train_nn.py # Trains neural network for modulus prediction
├── visualize.py # Basic data and result plots
├── data/
│ ├── mesh_parameters.csv # Input: all ellipse/full geometry parameters
│ ├── fem_results.csv # Output: FEM-derived Young's modulus per sample
│ ├── full_dataset.csv # Final dataset combining input and output
│ └── ellipse_scatter.png # Visualization
├── meshes/ # Stored .msh mesh files

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

- [ ] Feeding FEM with models (CalculiX)
- [ ] Parallelization for faster mesh/FEM simulations
- [ ] Hyperparameter tuning for neural network
- [ ] Error analysis and validation
- [ ] Jupyter notebook example for reproducibility

---

## Author

**Andrea Vinarš**  
Email: andrea.vinars3@gmail.com  

---

## License

MIT License


