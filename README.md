GIS-Integrated Adaptive Spatio-Temporal Graph Neural Network for Ward-Level Groundwater Forecasting

A deep learning framework for urban groundwater depth analysis and forecasting using Graph Neural Networks, Transformers, and GIS integration.

This project proposes HydroAST-GNN, an adaptive spatio-temporal model designed to analyze groundwater dynamics at ward-level resolution in dense urban environments such as Chennai, India. 

MiniProject_Report_22MIC0019

The system integrates multi-source geospatial datasets, deep learning models, and an interactive GIS dashboard for real-time groundwater monitoring and forecasting.

Project Overview

Urban groundwater resources fluctuate due to climate variability, urbanization, land-use changes, and extraction patterns.

Traditional hydrological models often fail to capture:

Local spatial heterogeneity

Long-term temporal dependencies

Urban environmental influences

To address this, this project introduces HydroAST-GNN, which combines:

Transformer-based temporal modeling

Spatial attention using Graph Neural Networks

GIS-based environmental features

The model predicts ward-level groundwater depth across 200+ wards in Chennai and provides forecasting for up to 24 months ahead. 

MiniProject_Report_22MIC0019

Key Features
Ward-Level Groundwater Forecasting

Predict groundwater depth for each administrative ward rather than large hydrological zones.

Multi-Source Geospatial Data Integration

Combines satellite, climatic, and hydrological datasets.

Adaptive Spatio-Temporal Modeling

Captures both:

Temporal dependencies (seasonality, lag effects)

Spatial relationships between neighboring wards.

GIS Visualization

Provides interactive maps and spatial analysis.

Real-Time Dashboard

Streamlit-based interface for groundwater analytics and predictions.

Multi-Step Forecasting

Supports predictions for 1–24 months ahead.

System Architecture

The project follows a multi-stage pipeline:

Data Sources
   │
   ▼
Data Harmonization & Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
HydroAST-GNN Model
   ├─ Temporal Transformer Block
   ├─ Spatial Attention Block
   └─ Readout Layer
   │
   ▼
Model Training & Evaluation
   │
   ▼
GIS Dashboard (Streamlit)
   │
   ▼
Groundwater Forecast Visualization

The architecture integrates graph-based spatial reasoning with temporal deep learning to model groundwater dynamics effectively. 

MiniProject_Report_22MIC0019

Methodology

The workflow consists of six major stages:

1 Data Collection

Groundwater and environmental datasets were collected from:

OpenCity Chennai groundwater dataset

Chennai Metropolitan Water Supply and Sewerage Board (CMWSSB)

Sentinel-2 satellite imagery

ERA5 climate data

SRTM elevation data

OpenStreetMap drainage data

2 Data Preprocessing

Steps include:

Temporal alignment

Missing value handling

Spatial aggregation of GIS layers

Feature normalization

3 Feature Engineering

Important features include:

Rainfall

NDVI

Land Use / Land Cover

Temperature

Evapotranspiration

Elevation

Drainage density

Lagged groundwater values

4 Spatial Graph Construction

Each ward is treated as a graph node, and spatial relationships are derived using Haversine distance-based adjacency matrices. 

MiniProject_Report_22MIC0019

5 Model Architecture: HydroAST-GNN

The model integrates:

Temporal Transformer Block

Captures seasonal and long-term patterns.

Spatial Attention Block

Learns interactions between neighboring wards.

Readout Layer

Produces groundwater depth prediction.

6 Deployment

The trained model is deployed using a Streamlit-based GIS dashboard.

Features include:

Ward search

Heatmap visualization

Time-series plots

Forecast generation

QGIS export support

Dataset

The dataset integrates multiple geospatial layers:

Feature	Source	Purpose
Groundwater depth	OpenCity	Target variable
Rainfall	ERA5 / IMD	Recharge driver
NDVI	Sentinel-2	Vegetation influence
LULC	ESA WorldCover	Urban impermeability
Temperature	ERA5	Evapotranspiration
PET	MODIS	Water balance
DEM	SRTM	Terrain slope
Drainage networks	OpenStreetMap	Runoff pathways

All datasets were harmonized to monthly resolution for modeling. 

MiniProject_Report_22MIC0019

Model Training
Input
(N wards, 12 months history, F features)
Output
Groundwater depth prediction for next month
Training Configuration
Parameter	Value
Optimizer	Adam
Learning rate	3e-4
Loss	Mean Squared Error
Batch size	4
Epochs	200
Early stopping	20 epochs
Gradient clipping	1.0
Evaluation Metrics

Model performance is evaluated using standard hydrological metrics:

Metric	Description
RMSE	Root Mean Squared Error
MAE	Mean Absolute Error
SMAPE	Symmetric Mean Absolute Percentage Error
KGE	Kling-Gupta Efficiency
R	Correlation coefficient

These metrics assess accuracy, reliability, and hydrological consistency of predictions. 

MiniProject_Report_22MIC0019

Results

The HydroAST-GNN model demonstrates strong predictive performance.

Example global metrics:

RMSE  ≈ 0.75 m
MAE   ≈ 0.47 m
R     ≈ 0.98
SMAPE ≈ 14.97 %
KGE   ≈ 0.90

The model successfully captures:

Seasonal groundwater fluctuations

Cross-ward spatial influence

Long-term hydrological trends

Visualization

The system provides several visual analytics tools:

Groundwater Trend Plots

Historical vs predicted groundwater depth.

Spatial Error Heatmaps

Ward-wise model accuracy.

Ward-Level Forecast Maps

Future groundwater predictions visualized as GIS choropleths.

Interactive Dashboard

Real-time exploration of groundwater behavior.

Technologies Used
Programming

Python 3.10+

Machine Learning

PyTorch

Scikit-learn

NumPy

Pandas

Geospatial Tools

GeoPandas

Shapely

GDAL

Rasterio

Folium

Visualization

Matplotlib

Plotly

Dashboard

Streamlit

Streamlit-Folium

GIS Software

QGIS

Hardware Requirements

Minimum:

Intel i5 / Ryzen 5

8 GB RAM

10 GB storage

Recommended:

Intel i7 / Ryzen 7

16–32 GB RAM

NVIDIA RTX GPU

Running the Project

Clone the repository:

git clone https://github.com/yourusername/HydroAST-GNN-Groundwater-Forecasting.git
cd HydroAST-GNN-Groundwater-Forecasting

Install dependencies:

pip install -r requirements.txt

Train the model:

python src/training/HydroASTGNN_training.py

Run inference:

python src/inference/Inference.py

Launch the dashboard:

streamlit run dashboard/app.py
Applications

This system can support:

Urban water management

Groundwater sustainability planning

Climate impact assessment

Municipal policy decision-making

Early warning for groundwater depletion

Future Improvements

Incorporate groundwater extraction data

Add uncertainty estimation

Expand to other cities

Integrate real-time IoT groundwater sensors

Improve model interpretability

Author

Rufina M
M.Tech Integrated – Computer Science and Engineering
Vellore Institute of Technology (VIT)

Project Supervisor
Dr. Yuvaraj N
Associate Professor
School of Computer Science and Engineering

License

This project is released for academic and research purposes.
