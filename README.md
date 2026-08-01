# Effective Elastic Properties of Plates with Elliptical Holes

This project combines finite element homogenization and a feed-forward neural
network (FNN) to estimate the in-plane effective elastic properties of a 2D
periodic plate containing two elliptical holes.

For each geometry, the workflow creates a periodic Gmsh mesh, runs three
CalculiX load cases, evaluates homogenized stresses, and extracts four effective
engineering constants:

- Young's modulus in the x-direction, $E_x$
- Young's modulus in the y-direction, $E_y$
- In-plane shear modulus, $G_{xy}$
- Poisson's ratio, $\nu_{xy}$

The resulting dataset is used to train an FNN surrogate that predicts all four
properties directly from the geometry of the two holes.

## Contents

- [Workflow](#workflow)
- [Mathematical Foundation](#mathematical-foundation)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running FEM Simulations](#running-fem-simulations)
- [Fixed-Geometry Validation](#fixed-geometry-validation)
- [Neural-Network Surrogate](#neural-network-surrogate)
- [Representative Results](#representative-results)
- [Assumptions and Limitations](#assumptions-and-limitations)

## Key Features

- Latin Hypercube Sampling (LHS) of ten independent ellipse parameters
- Parametric geometry and adaptive 2D meshing with Gmsh
- Periodic boundary conditions on opposite plate edges
- Automatic mesh-quality checks and fallback meshing strategies
- Three CalculiX analyses per geometry: `EX`, `EY`, and `XY`
- Two methods for extracting effective elastic properties
- Parallel simulation execution with `joblib`
- Automatic dataset, diagnostic CSV, histogram, and log generation
- A shared 26-feature FNN pipeline for training and inference
- Prediction of $E_x$, $E_y$, $G_{xy}$, and $\nu_{xy}$

## Workflow

```mermaid
flowchart LR
    A["LHS geometry sampling"] --> B["Gmsh geometry and periodic mesh"]
    B --> C["Mesh-quality validation"]
    C --> D["CalculiX EX, EY, and XY analyses"]
    D --> E["Stress and displacement post-processing"]
    E --> F["Effective elastic properties"]
    F --> G["Machine-learning dataset"]
    G --> H["FNN training and evaluation"]
    H --> I["Fast surrogate prediction"]
```

## Mathematical Foundation

### Mechanical Model

The representative volume element is a rectangular, plane-stress plate with
two elliptical holes. Each hole is described by five parameters:

$$
(x, y, r_x, r_y, \theta)
$$

where $(x,y)$ is the center, $r_x$ and $r_y$ are the semi-axes, and $\theta$ is
the rotation angle. The full geometry therefore contains ten sampled
parameters.

The matrix material is isotropic and linearly elastic. Small strains and a
constant plate thickness are assumed.

### Periodic Boundary Conditions

Opposite boundary nodes are paired in Gmsh before mesh generation. After the
final mesh has been accepted, CalculiX `*EQUATION` constraints are assembled
from the actual boundary-node pairs and corner nodes.

In general, the periodic displacement difference follows

$$
\mathbf{u}(\mathbf{x}^{+}) - \mathbf{u}(\mathbf{x}^{-})
= \bar{\boldsymbol{\varepsilon}}
\left(\mathbf{x}^{+} - \mathbf{x}^{-}\right).
$$

The same accepted mesh is reused for all three load cases:

| Load case | Prescribed macroscopic deformation | Primary response |
| --- | --- | --- |
| `EX` | $\bar{\varepsilon}_{xx}=\varepsilon_0$ | $\bar{\sigma}_{xx}$ |
| `EY` | $\bar{\varepsilon}_{yy}=\varepsilon_0$ | $\bar{\sigma}_{yy}$ |
| `XY` | $\bar{\gamma}_{xy}=\gamma_0$ | $\bar{\tau}_{xy}$ |

### Homogenized Stress

CalculiX writes stresses and integration volumes to the `.dat` file. The
homogenized stress is evaluated by volume-weighted averaging:

$$
\bar{\boldsymbol{\sigma}}
=
\frac{\sum_i \boldsymbol{\sigma}_i V_i}{\sum_i V_i}.
$$

### Effective-Property Calculation

The simulation supports two calculation modes through
`simulation.calculation_mode`.

#### `from_c`

The three homogenized stress vectors form the columns of the in-plane stiffness
matrix:

$$
\mathbf{C}=
\begin{bmatrix}
\bar{\boldsymbol{\sigma}}^{EX}/\varepsilon_0 &
\bar{\boldsymbol{\sigma}}^{EY}/\varepsilon_0 &
\bar{\boldsymbol{\sigma}}^{XY}/\gamma_0
\end{bmatrix}.
$$

With $\mathbf{S}=\mathbf{C}^{-1}$, the engineering constants are

$$
E_x=\frac{1}{S_{11}}, \qquad
E_y=\frac{1}{S_{22}}, \qquad
G_{xy}=\frac{1}{S_{33}}, \qquad
\nu_{xy}=-\frac{S_{21}}{S_{11}}.
$$

#### `direct_contraction`

The primary moduli are calculated directly from the corresponding load cases:

$$
E_x=\frac{\bar{\sigma}_{xx}^{EX}}{\varepsilon_0}, \qquad
E_y=\frac{\bar{\sigma}_{yy}^{EY}}{\varepsilon_0}, \qquad
G_{xy}=\frac{\bar{\tau}_{xy}^{XY}}{\gamma_0}.
$$

The transverse displacement measured during `EX` loading is used to determine
$\nu_{xy}$.

## Repository Structure

```text
.
|-- Main/
|   |-- Fixed_elipses_corners_Bez_Kontrakcije.py  Fixed-geometry validation
|   `-- predict_fixed.py                         Surrogate inference for validation
|-- Pipeline/
|   |-- Mesh-FEM/
|   |   |-- simulation.py                 Main parallel FEM pipeline
|   |   |-- simulation_context.py         Configuration and load-case paths
|   |   |-- data_processing.py            CalculiX parsing and homogenization
|   |   |-- equation_block_corners.py     Periodic-equation implementation
|   |   `-- Helper_functions.py           Sampling, logging, and mesh diagnostics
|   `-- FNN/
|       |-- FNN.py                        Four-output surrogate training
|       |-- FNN_shared.py                 Shared features and FNN architecture
|       `-- predict.py                    General inference helpers
|-- config_example.yaml           Example simulation configuration
|-- requirements.txt              Python dependencies
|-- data/                         Datasets and parameter histograms
|-- models/                       Saved model and scaler artifacts
|-- Results/
|   |-- Mesh-FEM/                 Mesh and FEM result images
|   `-- FNN/                      Training and evaluation images
```

The `input_files/`, `output_files/`, and `logs/` directories contain generated
simulation artifacts and can become large during dataset generation.

All commands below assume that the current working directory is the repository
root.

## Requirements

- Python 3.10 or newer
- CalculiX executable (`ccx` or `ccx_static`)
- Python packages listed in `requirements.txt`

The main Python dependencies are Gmsh, NumPy, SciPy, pandas, PyTorch,
scikit-learn, Matplotlib, Joblib, PyYAML, and tqdm.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AndreaVinars/Mesh---FEM---NN.git
cd Mesh---FEM---NN
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure CalculiX and the simulation

Create a local configuration file:

Windows PowerShell:

```powershell
Copy-Item config_example.yaml config.yaml
```

Linux or macOS:

```bash
cp config_example.yaml config.yaml
```

Edit `config.yaml` and provide the path to the CalculiX executable. All
available settings and their units are documented in `config_example.yaml`.

The `CALCULIX_CMD` environment variable can override the executable configured
in YAML:

```powershell
$env:CALCULIX_CMD = "C:\path\to\ccx_static.exe"
```

## Running FEM Simulations

Start with a small number of geometries to verify the installation:

```bash
python Pipeline/Mesh-FEM/simulation.py config.yaml
```

Each geometry is meshed once and analyzed in the `EX`, `EY`, and `XY` load
cases. With `num_simulations: 20`, this produces up to 60 CalculiX analyses.

The main generated outputs are:

```text
data/<calculation_mode>/ml_data_<calculation_mode>_seed_<seed>.csv
data/<calculation_mode>/param_histograms_<calculation_mode>.png
data/rejected_simulations.csv
input_files/sim_XXXX_<load_case>.inp
output_files/sim_XXXX_<load_case>.*
logs/main.log
logs/sim_XXXX.log
```

The dataset uses a semicolon separator and a decimal comma.

## Fixed-Geometry Validation

The main example runs the three FEM load cases for one fixed geometry and
compares the homogenized properties with the trained surrogate:

```bash
python Main/Fixed_elipses_corners_Bez_Kontrakcije.py
```

By default, the script loads `best_silu.pt`, `scaler_X.pkl`, and
`scaler_y.pkl` from the root `models/` directory. Set `CALCULIX_CMD` or
`FNN_MODEL_DIR` to override the corresponding executable or model location.

## Neural-Network Surrogate

The user supplies ten geometric values for two ellipses. These values are
expanded into 26 model features:

- 18 base features, including trigonometric angle encoding, relative position,
  distance, areas, and relative orientation
- 8 derived features describing area, aspect-ratio, and semi-axis relationships

The FNN predicts the following targets simultaneously:

```text
E_x [GPa]
E_y [GPa]
G_xy [GPa]
NU_xy [-]
```

The current architecture contains five hidden layers with 32 neurons and SiLU
activations. Dropout is applied after the first two layers. Training uses a
70/15/15 train-validation-test split, `StandardScaler`, Smooth L1 loss, AdamW,
learning-rate reduction on validation plateaus, and early stopping.

### Training

Set the working directory and dataset explicitly before starting training.
For a `direct_contraction` dataset generated with seed 30:

Windows PowerShell:

```powershell
$env:FNN_WORK_DIR = (Get-Location).Path
$env:FNN_DATASET_PATH = (Resolve-Path ".\data\direct_contraction\ml_data_direct_contraction_seed_30.csv").Path
python Pipeline/FNN/FNN.py
```

Linux or macOS:

```bash
export FNN_WORK_DIR="$PWD"
export FNN_DATASET_PATH="$PWD/data/direct_contraction/ml_data_direct_contraction_seed_30.csv"
python Pipeline/FNN/FNN.py
```

Training creates:

```text
models/best_silu.pt
models/scaler_X.pkl
models/scaler_y.pkl
plots/SILU_<target>.png
plots/<target>_distribution.png
plots/LOSS.png
logs/train_<timestamp>.log
```

### Inference

The reusable inference functions load the model and both scalers from the root
`models/` directory. Point `FNN_MODEL_DIR` to a different artifact directory
when needed:

```powershell
$env:FNN_MODEL_DIR = (Resolve-Path ".\models").Path
```

Example prediction:

```python
from Main.predict_fixed import load_surrogate_model, predict_custom

model, scaler_X, scaler_y, device = load_surrogate_model()

properties = predict_custom(
    model,
    scaler_X,
    scaler_y,
    x1=4.11,
    y1=3.77,
    rx1=1.83,
    ry1=1.16,
    angle1_deg=48.9,
    x2=8.03,
    y2=7.56,
    rx2=1.64,
    ry2=0.87,
    angle2_deg=118.3,
    device=device,
)

for name, value in properties.items():
    print(f"{name}: {value:.6f}")
```

Run this example from the repository root. The complete FEM-versus-FNN
validation workflow is available in
`Main/Fixed_elipses_corners_Bez_Kontrakcije.py`.

The surrogate should only be used within the geometric parameter range covered
by its training dataset.

## Representative Results

The following held-out test metrics were obtained for the currently published
model using a dataset of 5,000 accepted geometries:

| Target | RMSE | MAE |
| --- | ---: | ---: |
| $E_x$ [GPa] | 3.209959 | 2.216892 |
| $E_y$ [GPa] | 3.228733 | 2.032885 |
| $G_{xy}$ [GPa] | 1.789192 | 1.086722 |
| $\nu_{xy}$ [-] | 0.011676 | 0.007045 |

Exact results depend on the sampled geometries, accepted meshes, dataset size,
random seed, and training configuration.

### Periodic Mesh

![Periodic Gmsh mesh](Results/Mesh-FEM/periodic_mesh.png)

### FEM Load Cases

The contour plots show the CalculiX response for the three independent
macroscopic deformation states used during homogenization.

| Tension in $x$ (`EX`) | Tension in $y$ (`EY`) | Shear (`XY`) |
| --- | --- | --- |
| ![FEM result for tensile loading in x](Results/Mesh-FEM/Tensile_X.png) | ![FEM result for tensile loading in y](Results/Mesh-FEM/Tensile_Y.png) | ![FEM result for xy shear loading](Results/Mesh-FEM/Shear_XY.png) |

### FNN Training History

![FNN training and validation loss](Results/FNN/LOSS.png)

### True vs. Predicted Properties

| $E_x$ | $E_y$ |
| --- | --- |
| ![True versus predicted E_x](Results/FNN/SILU_E_x.png) | ![True versus predicted E_y](Results/FNN/SILU_E_y.png) |

| $G_{xy}$ | $\nu_{xy}$ |
| --- | --- |
| ![True versus predicted G_xy](Results/FNN/SILU_G_xy.png) | ![True versus predicted NU_xy](Results/FNN/SILU_NU_xy.png) |

### Target Distributions

| $E_x$ | $E_y$ |
| --- | --- |
| ![Distribution of E_x](Results/FNN/E_x_distribution.png) | ![Distribution of E_y](Results/FNN/E_y_distribution.png) |

| $G_{xy}$ | $\nu_{xy}$ |
| --- | --- |
| ![Distribution of G_xy](Results/FNN/G_xy_distribution.png) | ![Distribution of NU_xy](Results/FNN/NU_xy_distribution.png) |

## Assumptions and Limitations

- The matrix material is isotropic and linearly elastic.
- The analysis uses small strains and plane-stress conditions.
- The current geometry contains exactly two elliptical holes.
- Invalid geometries, poor meshes, solver failures, and failed post-processing
  cases are excluded from the training dataset and recorded separately.
- Prediction accuracy is expected to deteriorate outside the training range.
- CalculiX is an external dependency and must be installed separately.

## Reproducibility

- LHS sampling is controlled by `simulation.seed`.
- Input and target scalers are fitted only on the training subset.
- The trained model and both scalers must be kept together for inference.
- The feature and target order is defined centrally in `FNN_shared.py`.
- Simulation mode is stored in the generated dataset as `dataset_mode`.

## Author

**Andrea Vinarš**  
Email: andrea.vinars3@gmail.com

## License

MIT License
