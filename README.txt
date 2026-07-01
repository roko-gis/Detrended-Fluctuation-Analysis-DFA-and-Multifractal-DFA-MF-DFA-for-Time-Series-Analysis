# Detrended Fluctuation Analysis (DFA) and Multifractal DFA (MF-DFA) for Time Series Analysis

## Overview

This software provides an implementation of Detrended Fluctuation Analysis (DFA) and Multifractal Detrended Fluctuation Analysis (MF-DFA) for investigating scaling properties, long-range correlations, and multifractal characteristics in time series data. The software is implemented as a Python Processing Tool script.

The methodology enables the characterization of complex temporal dynamics and has applications in geosciences, climatology, hydrology, ecology, environmental monitoring, economics, physiology, and other disciplines involving nonlinear processes.

The present implementation was developed for the QGIS environment and allows users to perform fractal analyses directly within a Geographic Information System (GIS), facilitating the integration of temporal and spatial information.

---

## Scientific Background

Many natural and anthropogenic processes exhibit nonstationary behavior and long-range dependence. Conventional statistical approaches are often unable to adequately characterize these properties.

Detrended Fluctuation Analysis (DFA), introduced by Peng et al. (1994), estimates the scaling exponent associated with persistent or anti-persistent behavior in time series.

Multifractal Detrended Fluctuation Analysis (MF-DFA), proposed by Kantelhardt et al. (2002), extends DFA by quantifying multifractal characteristics and evaluating the heterogeneity of scaling behavior across fluctuations of different magnitudes.

These techniques have been widely applied in studies related to:

* climate variability;
* hydrological processes;
* vegetation dynamics;
* ecosystem functioning;
* seismic activity;
* physiological signals;
* economic and financial systems;
* remote sensing time series.

---

## Purpose of the Software

The objective of this software is to provide a reproducible, transparent, and user-friendly framework for performing DFA and MF-DFA analyses without requiring advanced programming expertise.

The software aims to:

* simplify fractal analysis workflows;
* provide scientifically reproducible outputs;
* facilitate integration with GIS-based studies;
* support interdisciplinary research involving temporal datasets.

---

## Target Users

The software is intended for:

* researchers and scientists;
* graduate and postgraduate students;
* environmental analysts;
* geographers and GIS specialists;
* hydrologists and climatologists;
* ecologists and Earth system scientists;
* researchers working with nonlinear and complex systems.

---

## Input Data

The software accepts one-dimensional numerical time series representing temporal observations from a wide range of scientific disciplines.

### Supported Input Sources

Input data can be imported from:

* CSV files (*.csv);
* Microsoft Excel files (*.xlsx);
* text files (*.txt);
* QGIS vector layers and attribute tables.

Supported vector formats include:

* ESRI Shapefile (*.shp);
* GeoPackage (*.gpkg);
* GeoJSON (*.geojson);
* any vector format supported by QGIS.

For vector datasets, the analysis is performed on a user-selected numerical attribute field, which is internally converted into a one-dimensional time series.

### Data Requirements

Input datasets should satisfy the following conditions:

* contain numerical values;
* represent a temporal sequence;
* be arranged in chronological order;
* have missing values removed or appropriately handled before analysis;
* avoid non-numeric entries.

### Expected Structure

Example of a tabular dataset:

| Date    | Value |
| ------- | ----- |
| 2020-01 | 0.43  |
| 2020-02 | 0.51  |
| 2020-03 | 0.47  |

Example of a vector layer attribute table:

| ID | Year | NDVI |
| -- | ---- | ---- |
| 1  | 2020 | 0.43 |
| 2  | 2021 | 0.51 |
| 3  | 2022 | 0.47 |

### Typical Applications

The software can process time series derived from:

#### Environmental observations

* precipitation;
* temperature;
* streamflow;
* groundwater levels;
* evapotranspiration;
* drought indices.

#### Remote sensing products

* NDVI;
* EVI;
* vegetation indices;
* land surface temperature;
* satellite-derived environmental variables.

#### Geophysical measurements

* seismic records;
* atmospheric observations;
* magnetic field measurements.

#### Biological and physiological signals

* heart rate variability;
* EEG recordings;
* biomedical measurements.

#### Socio-economic datasets

* stock market indices;
* economic indicators;
* energy consumption records.

---

## Methodology

### Detrended Fluctuation Analysis (DFA)

The algorithm performs the following steps:

1. Integration of the original time series;
2. Division into non-overlapping segments;
3. Polynomial detrending within each segment;
4. Estimation of the fluctuation function;
5. Determination of the scaling exponent (Hurst exponent).

### Multifractal Detrended Fluctuation Analysis (MF-DFA)

The MF-DFA procedure extends DFA by:

1. Calculating fluctuation functions for multiple moments q;
2. Estimating generalized Hurst exponents;
3. Computing mass exponents τ(q);
4. Deriving singularity spectra;
5. Characterizing multifractal properties.

---

## Software Architecture

The software was developed in Python using:

* Python 3;
* PyQGIS;
* NumPy;
* SciPy;
* Pandas;
* Matplotlib;
* PyQt.

The graphical interface was designed using Qt Designer and integrated into the QGIS environment.

The processing workflow is divided into:

* data import;
* preprocessing;
* DFA computation;
* MF-DFA computation;
* visualization;
* export of results.

---

## Output Products

The software generates:

### DFA Outputs

* Hurst exponent;
* fluctuation functions;
* log-log regression plots.

### MF-DFA Outputs

* generalized Hurst exponents;
* mass exponent τ(q);
* singularity spectrum f(α);
* multifractal spectrum width;
* spectrum asymmetry indicators.

### Exported Products

* CSV tables;
* graphical outputs;
* publication-ready figures.

---

## Applications

The software can support studies related to:

* climate change;
* drought assessment;
* ecosystem resilience;
* vegetation dynamics;
* hydrological variability;
* environmental monitoring;
* nonlinear dynamics;
* complexity science.

---

## Reproducibility

All calculations are deterministic and reproducible. Identical input data and parameter settings produce identical outputs.

The software follows transparent scientific computing principles and facilitates reproducible research.

---

## References

Peng, C. K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., & Goldberger, A. L. (1994). *Mosaic organization of DNA nucleotides*. Physical Review E, 49, 1685-1689.

Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S., Bunde, A., & Stanley, H. E. (2002). *Multifractal detrended fluctuation analysis of nonstationary time series*. Physica A, 316, 87-114.

---

## License

MIT License

---

## Citation

If you use this software in scientific research, please cite the software using the DOI provided by Zenodo.