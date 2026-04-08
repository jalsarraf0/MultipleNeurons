# MultipleNeurons

A small Keras neural network that predicts house sale prices from tabular features.

## Overview

This project demonstrates a multi-layer feedforward neural network for regression using Keras. It trains on a housing dataset (`data.csv`) with 8 input features — year built, floor area, living area, bath counts, bedroom count, total rooms, and year sold — and predicts sale price. The model is saved after training and reloaded to verify inference.

## How to Run

**1. Install dependencies**

```bash
pip install keras numpy pandas
```

**2. Prepare data**

Place a `data.csv` file in the project directory. The file must have the following columns in order:

```
YearBuilt, 2ndFlrSF, GrLivArea, FullBath, HalfBath, BedroomAbvGr, TotRmsAbvGrd, YrSold, SalePrice
```

**3. Train and run inference**

```bash
python BrainsAndHouses.py
```

The script will:
- Train a 3-layer network (8 → 8 → 8 → 1) for up to 30 epochs with early stopping
- Print a sale price prediction for a hardcoded test sample
- Save the model to `saved_model.h5`
- Reload the saved model and run the same prediction again to confirm save/load works

## Model Architecture

| Layer | Units | Activation |
|-------|-------|------------|
| Dense | 8     | ReLU       |
| Dense | 8     | ReLU       |
| Dense | 1     | Linear     |

- Optimizer: Adam
- Loss: Mean Squared Error
- Early stopping: patience = 3 epochs

## Dependencies

- keras
- numpy
- pandas
