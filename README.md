# Mesh---FEM---NN
Automated generation of 2D finite element meshes with parametric, rotatable elliptical holes for use in computational mechanics and machine learning workflows.

# Predicting Young's Modulus with Neural Networks from Parametric FEA

This project aims to train a feedforward neural network to predict the modulus of elasticity (Young's modulus) of a 2D plate with two variable, rotatable elliptical holes, based solely on geometric input parameters. The dataset is generated using finite element simulations (FEM) in CalculiX, with plates of fixed dimensions and varying ellipse parameters as inputs.

---

## Project Workflow

1. **Mesh Generation:** Automated creation and meshing of 2D plates (fixed dimensions) with two randomly parameterized elliptical holes (position, axes, angle) using GMSH Python API.
2. **FEM Simulation:** Batch simulation of all geometries in CalculiX, applying a prescribed displacement to extract global reaction force for each sample.
3. **Dataset Assembly:** For each geometry, the input vector is the 10D set of ellipse parameters; the output is the computed Young's modulus (from the F=K·u relation).
4. **Neural Network Training:** Training a feedforward neural network (regression) to predict Young's modulus based only on ellipse parameters (geometry→modulus).
5. **Surrogate Prediction:** Fast evaluation of modulus for new geometries using the trained neural network, bypassing the need for full FEM.

---

## Features

- **Parametric 2D geometry:** Plate with two elliptical holes, with randomizable position, orientation, and size.
- **Mesh and simulation automation:** Generate 5000+ unique samples and automatically simulate them in CalculiX.
- **Comprehensive dataset:** All input parameters and computed outputs are tabulated in a CSV (Excel-compatible), ready for ML workflows.
- **Neural network regression:** Feedforward architecture trained for direct modulus of elasticity inference.

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

---

## Example Data Table

| id |  x1  |  y1  |  rx1  | ry1  | angle1 |  x2  |  y2  |  rx2 |  ry2 | angle2 | E (MPa) |
|----|------|------|-------|------|--------|------|------|------|------|--------|---------|
| 0  | 6.21 | 2.99 | 1.00  | 0.71 |  41.3  | 2.39 | 7.01 | 0.91 | 0.83 |  187.5 |  199000 |
| ...| ...  | ...  |  ...  | ...  |  ...   | ...  | ...  | ...  | ...  |   ...  |   ...   |

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


