# Mesh---FEM---NN
Automated generation of 2D finite element meshes with parametric, rotatable elliptical holes for use in computational mechanics and machine learning workflows.

# 2D Parametric FE Meshes with Rotatable Elliptical Holes

Automated generation of 2D finite element meshes for a rectangular plate containing two parametric, rotatable elliptical holes. Developed for computational mechanics and machine learning workflows.

---

## Project Description

This project provides scripts for batch generation of mesh datasets representing plates with variable elliptical holes (location, size, orientation). All mesh and geometry parameters are exported to a CSV table, ready for FEM analysis or surrogate modeling.  
Primary goals are to enable large-scale synthetic data creation for numerical experiments, design optimization, or neural network training.

---

## Project Status

- ✅ Fully operational mesh generation and geometry randomization
- ✅ Automated CSV parameter export (Excel compatible)
- ✅ Visualization of parameter distributions and mesh geometry
- ⏳ FEM simulations and ML model integration planned

---

## Features

- Parametric specification of ellipse center, semi-axes, and rotation for each defect
- Automated batch generation (configurable sample size)
- CSV export with columns for every geometry and mesh quantity (Excel-friendly)
- Scatter plot and histogram visualization of parameter distributions

---

## Repository Structure

