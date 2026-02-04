# VietNam PM2.5 Forecasting

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Data-pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/Modeling-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-0171CE?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Open--Meteo-0080FF?logo=cloudflare&logoColor=white)](https://open-meteo.com/)
[![Trello](https://img.shields.io/badge/Project-Trello-0079BF?logo=trello&logoColor=white)](https://trello.com/invite/b/690653218d9e920d842ae003/ATTI57b16ae720553776d120c333a5cf0f09C269FA10/project-nmkhdl)
[![Status](https://img.shields.io/badge/Status-Completed-22C55E?logoColor=white)](https://github.com)

**A comprehensive machine learning project for short-term PM2.5 concentration forecasting across Vietnam using meteorological and air quality data**

---

**Course:** Introduction to Data Science  
**Faculty:** Data Science

</div>

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
  - [Abstract](#abstract)
  - [Methodology](#methodology)
  - [Objectives](#objectives)
- [2. Team Information](#2-team-information)
- [3. Dataset Source & Description](#3-dataset-source--description)
  - [3.1 Dataset Source](#31-dataset-source)
  - [3.2 Dataset Description](#32-dataset-description)
  - [3.3 Key Features](#33-key-features)
- [4. Research Questions](#4-research-questions)
  - [4.1 Investigating the Impact of Human Activities](#41-investigating-the-impact-of-human-activities)
  - [4.2 Investigating Weather and Regional Climate Conditions](#42-investigating-weather-and-regional-climate-conditions)
  - [4.3 Pollutants Used in AQI Calculation](#43-pollutants-used-in-aqi-calculation)
- [5. Key Findings Summary](#5-key-findings-summary)
  - [5.1 Human Activity Impact on Air Quality](#51-human-activity-impact-on-air-quality)
  - [5.2 Pollution Sources](#52-pollution-sources)
  - [5.3 PM2.5/PM10 Ratio as a Diagnostic for Local vs. External Pollution](#53-pm25pm10-ratio-as-a-diagnostic-for-local-vs-external-pollution)
  - [5.4 Model Performance](#54-model-performance)
- [6. File Structure](#6-file-structure)
  - [6.1 Directory Descriptions](#61-directory-descriptions)
- [7. Installation & Setup](#7-installation--setup)
  - [7.1 Prerequisites](#71-prerequisites)
  - [7.2 Setup](#72-setup)
  - [7.3 Usage](#73-usage)
- [8. Dependencies](#8-dependencies)
- [9. Model Performance Comparison](#9-model-performance-comparison)
- [10. Limitations & Future Work](#10-limitations--future-work)
- [11. Contact](#11-contact)
- [12. References](#12-references)

---

## 1. Project Overview

### Abstract

This project presents a comprehensive machine learning approach for forecasting PM2.5 (particulate matter with diameter ≤ 2.5 micrometers) concentration levels across major cities in Vietnam. PM2.5 poses significant health risks in urban areas, and accurate short-term forecasting is crucial for public health protection and early warning systems.

### Methodology

The project implements a complete data science pipeline:

1. **Data Collection**: Automated data retrieval from the [Open-Meteo API](https://open-meteo.com/) covering approximately **34 provinces and cities** from January 2023 to present with **hourly temporal resolution**

2. **Data Preprocessing**: Comprehensive cleaning, feature engineering (temporal features, lag variables, rolling statistics), and regional classifications

3. **Exploratory Data Analysis (EDA)**: Investigate the factors affecting air quality in Vietnamese cities, including temporal and seasonal patterns, and use modeling techniques to examine multicollinearity among features related to PM2.5 concentrations. The EDA also focuses on analyzing AQI (Air Quality Index) patterns and their relationships with various environmental factors. 

4. **Machine Learning Modeling**: Comparative analysis of multiple approaches:
   - Baseline linear models (OLS, Ridge, PCR, PLS)
   - Tree-based methods
   - Advanced boosting algorithms (XGBoost with Optuna hyperparameter optimization)

### Objectives

- Identify key meteorological drivers influencing PM2.5 fluctuations across different regions
- Understand seasonal patterns and pollution threshold exceedance frequencies
- Develop accurate forecasting models for short-term horizons (**1-24 hours ahead**)
- Support early warning systems and environmental monitoring across Vietnam's diverse climatic regions

---

## 2. Team Information

| # | Name                  | Student ID |
|:--|:----------------------|:-----------|
| 1 | Phạm Thành Nam        | 23120301   |
| 2 | Trương Quang Phát     | 23120318   |
| 3 | Huỳnh Tấn Phước       | 23120334   |
| 4 | Trần Nguyễn Minh Quân | 23120342   |

**Project Management:** [Trello Board](https://trello.com/invite/b/690653218d9e920d842ae003/ATTI57b16ae720553776d120c333a5cf0f09C269FA10/project-nmkhdl) - Team collaboration and task tracking

---

## 3. Dataset Source & Description

### 3.1 Dataset Source

The dataset is collected from the **[Open-Meteo API](https://open-meteo.com/)**, a comprehensive open-source weather and air quality data service that provides historical and real-time environmental data. The project utilizes two main API endpoints:

- **[Air Quality API](https://open-meteo.com/en/docs/air-quality-api)**: Provides hourly air pollution measurements including PM2.5, PM10, and various gaseous pollutants
- **[Weather Archive API](https://open-meteo.com/en/docs/historical-weather-api)**: Supplies historical meteorological data including temperature, humidity, precipitation, wind conditions, and atmospheric pressure

**Data Collection Details:**
- **Time Range**: From `2023-01-01` to `2025-11-26`
- **Temporal Resolution**: Hourly data points
- **Timezone**: `Asia/Ho_Chi_Minh` (UTC+7) to match Vietnam's local time
- **Geographic Coverage**: Approximately 34 provinces and major cities across Vietnam
- **Location Coordinates**: Sourced from `data/raw/vietnam_locations.csv`, which contains latitude and longitude coordinates for each city
- **Raw Data File**: `data/raw/vietnam_air_quality.csv` (generated via automated data crawling script)

### 3.2 Dataset Description

The dataset is a time-series collection integrating meteorological and air pollution variables. Each record represents an hourly observation for a specific city, suitable for time-series forecasting and spatial-temporal analysis.

**Data Characteristics:**
- **Combined Data**: Merges weather variables (temperature, humidity, precipitation, pressure, wind, cloud cover) with air pollution indicators (PM2.5, PM10, CO, NO2, SO2, O3)
- **US AQI**: Includes the US Air Quality Index as a composite measure
- **Pollution Labels**: Derived classification labels (`pollution_level`, `pollution_class`) based on AQI thresholds
- **Geographic Structure**: Fixed latitude/longitude coordinates per city throughout the time series

### 3.3 Key Features

| Feature Name | Description | Unit |
|:------------|:------------|:-----|
| `timestamp` | Date and time of observation (hourly granularity) | UTC+7 format |
| `city` | Name of the city/province | Text |
| `lat`, `lon` | Geographic coordinates (latitude, longitude) | Decimal degrees |
| `aqi` | US Air Quality Index (composite measure) | Index (0-500+) |
| `pollution_level` | Categorical pollution level (Good, Moderate, Unhealthy, etc.) | Text |
| `pollution_class` | Numerical pollution classification (0-5) | Integer |
| `pm2_5` | Fine particulate matter (diameter ≤ 2.5 µm) | µg/m³ |
| `pm10` | Coarse particulate matter (diameter ≤ 10 µm) | µg/m³ |
| `co` | Carbon monoxide concentration | µg/m³ |
| `no2` | Nitrogen dioxide concentration | µg/m³ |
| `so2` | Sulfur dioxide concentration | µg/m³ |
| `o3` | Ozone concentration | µg/m³ |
| `temp` | Air temperature at 2 meters above ground | °C |
| `humidity` | Relative humidity at 2 meters | % |
| `pressure` | Atmospheric surface pressure | hPa |
| `rain` | Precipitation amount | mm |
| `wind_speed` | Wind speed at 10 meters above ground | m/s |
| `wind_dir` | Wind direction (0-360°) | degrees |
| `cloud` | Cloud cover percentage | % |

---

## 4. Research Questions

### 4.1 Investigating the Impact of Human Activities

**Q1:** Does air quality significantly improve on weekends compared to weekdays?

**Q2:** How about the impact of population density and industrial scale on air quality (AQI)?

### 4.2 Investigating Weather and Regional Climate Conditions

**Q3:** What is the relationship between regional climate characteristics and air pollution levels (AQI) across different climate zones in Vietnam?

**Q4:** Do the alarming AQI levels in the Northern region primarily stem from internal emission sources or from external pollution?

### 4.3 Pollutants Used in AQI Calculation

**Q5:** Multicollinearity Handling Strategy between PM2.5 and PM10 to Optimize Regression Models

---

## 5. Key Findings Summary

### 5.1 Human Activity Impact on Air Quality

Human activities directly influence air quality in Vietnamese cities. Analysis reveals that air quality significantly improves on weekends compared to weekdays, with AQI typically peaking around **Thursday** and showing sharp decreases on **Saturday and Sunday** due to reduced industrial and traffic activities. Cities with high population density and intensive industrial activity (e.g., Hà Nội, Bắc Ninh, Hồ Chí Minh, Biên Hòa) show substantially higher AQI levels (mean AQI: **87.83**) compared to green cities with lower density and limited industrial concentration (mean AQI: **56.85**), demonstrating a **~31% difference** in pollution levels.

### 5.2 Pollution Sources

High AQI in Northern regions primarily stems from **local emission sources** rather than transboundary pollution. Wind speed negatively correlates with AQI (correlation ≈ **-0.12**), with PM2.5 decreasing noticeably when wind speed exceeds **3 m/s**.

### 5.3 PM2.5/PM10 Ratio as a Diagnostic for Local vs. External Pollution

In the most polluted northern cities (e.g., Hà Nội, Bắc Ninh, Hưng Yên), high AQI levels come from both **local emissions** and a strong **regional background** of pollution. The average PM2.5/PM10 ratio is about **0.79**, meaning most particles are fine dust and secondary aerosol, which is typical for **pollution transported from other regions** rather than only local road or construction dust.  
For Hà Nội, PM2.5 during rush hours is almost the same as at night (rush/night ratio ≈ **1.0**), and rain only reduces PM2.5 by around **26%**. This flat daily pattern and weak cleaning effect of rain show that pollution is **constantly replenished from outside the city**, so severe episodes cannot be explained by local sources alone.

### 5.4 Model Performance

**XGBoost** achieves superior results:
- **RMSE:** 2.7315
- **MAE:** 1.4011
- **R²:** 0.9766

This represents a **19.41% reduction in RMSE** and **20.49% reduction in MAE** compared to linear regression. Notably, in XGBoost, PM10 emerges as a **top-3 important feature** despite multicollinearity issues in linear models, as tree-based models effectively leverage the physical relationship between PM2.5 and PM10.

---

## 6. File Structure

```
VietNam_PM2.5_Forecasting/
│
├── data/                           # Data directory
│   ├── raw/                        # Raw data from API
│   │   ├── vietnam_air_quality.csv
│   │   └── vietnam_locations.csv   # Location coordinates for cities
│   ├── processed/                  # Processed and cleaned data
│   │   └── processed_data.csv
│   └── model/                      # Train/test/validation splits
│       ├── train.csv
│       ├── test.csv
│       └── val.csv
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── data_preprocessing.ipynb    # Data cleaning and feature engineering
│   ├── data_exploration.ipynb      # Exploratory Data Analysis (EDA)
│   ├── data_modeling.ipynb         # Model training and evaluation
│   └── summary.ipynb               # Project summary and key findings
│
├── src/                            # Source code modules
│   ├── create_locations.py         # Generate location coordinates
│   ├── crawl_data.py               # Fetch data from Open-Meteo API
│   ├── preprocessing.py            # Data preprocessing utilities
│   └── visualization.py            # Visualization helper functions
│
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

### 6.1 Directory Descriptions

- **`data/raw/`**: Contains raw hourly air quality and meteorological data collected from Open-Meteo API, and location coordinates file (`vietnam_locations.csv`) with latitude/longitude for each city
- **`data/processed/`**: Contains cleaned and feature-engineered datasets ready for modeling
- **`data/model/`**: Contains train/test/validation dataset splits (train.csv, test.csv, val.csv) for model training and evaluation
- **`notebooks/`**: Jupyter notebooks organized by workflow stages (preprocessing → exploration → modeling)
- **`src/`**: Python modules for data collection, preprocessing, and utility functions

## 7. Installation & Setup

### 7.1 Prerequisites
- Python 3.10 (recommended)
- pip package manager

### 7.2 Setup

#### Step 1: Clone Repository
```bash
git clone https://github.com/TrNguyenMQuan/VietNam_PM2.5_Forecasting
cd VietNam_PM2.5_Forecasting
```

#### Step 2: Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# MacOS/Linux
source .venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### 7.3 Usage

The project follows a sequential workflow from data collection to model training. Follow these steps in order:

#### Step 1: Generate Location Coordinates
```bash
python src/create_locations.py
```
**Output:** `data/raw/vietnam_locations.csv` containing latitude and longitude coordinates for approximately 34 provinces and cities across Vietnam.

#### Step 2: Collect Air Quality and Weather Data
```bash
python src/crawl_data.py
```
**Output:** `data/raw/vietnam_air_quality.csv` with hourly air quality and meteorological data from January 1, 2023 to 5 days before the current date.

#### Step 3: Understand Raw Dataset
Open `notebooks/data_exploration.ipynb` and run **Section I: Data Understanding about Raw Dataset** (all cells from the beginning through Section I).

**Purpose:** Get insights about raw dataset structure, size, variable meanings, and data quality before preprocessing.

#### Step 4: Initial Data Cleaning
Open `notebooks/data_preprocessing.ipynb` and run **Section I: Data Cleaning** (all cells through the end of Section I).

**Output:** `data/processed/processed_data.csv` - cleaned dataset ready for EDA analysis.

#### Step 5: Exploratory Data Analysis (EDA)
Return to `notebooks/data_exploration.ipynb` and run:
- **Section II: Data Understanding about Processed Data** - Verify cleaned data structure
- **Section III: Columns Analysis** - Analyze value distributions and visualizations
- **Section IV: Relationships & Correlations** - Investigate correlations between features and AQI
- **Section V: Meaningful Questions** - Answer research questions about human activities, climate impact, and pollution sources

**Purpose:** Deep dive into processed data to understand patterns, distributions, and relationships before final feature engineering.

#### Step 6: Complete Preprocessing for Modeling
Return to `notebooks/data_preprocessing.ipynb` and run:
- **Section II: Data Reduction** - Remove analyzed columns (aqi, pollution_level, pollution_class)
- **Section III: Feature Engineering** - Create temporal features, lag variables, rolling statistics
- **Section IV: Data Splitting** - Split data into train/validation/test sets

**Output:** 
- `data/model/train.csv` - Training dataset
- `data/model/val.csv` - Validation dataset  
- `data/model/test.csv` - Test dataset

#### Step 7: Model Training and Evaluation
Open and run `notebooks/data_modeling.ipynb`:
- Train baseline linear models (OLS, Ridge, PCR, PLS)
- Train and optimize XGBoost model using Optuna
- Compare model performance (RMSE, MAE, R²)
- Analyze feature importance and model interpretability

## 8. Dependencies

### Core Libraries

The following Python libraries are required to run this project:

- **Python** (3.10+)
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **TensorFlow** - Deep learning framework
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **Jupyter/IPython** - Interactive notebook environment

### Installation

All required libraries with specific versions are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

> **Note:** It is recommended to use a virtual environment to avoid conflicts with other projects.

---

## 9. Model Performance Comparison

### 9.1 Performance Metrics

The following table summarizes the performance of different models on the test set:

| Model | RMSE | MAE | R² | Improvement vs Baseline | Notes |
|:------|:-----|:----|:---|:------------------------|:------|
| **OLS (PM2.5 + PM10)** | 3.4013 | 1.7648 | - | Baseline | Multicollinearity issues |
| **OLS (PM2.5 only)** | 3.3891 | 1.7623 | - | -0.4% RMSE | Improved stability |
| **Ridge Regression** | 3.4011 | 1.7648 | - | -0.01% RMSE | L2 regularization |
| **PCR** | 3.4013 | 1.7648 | - | 0% | Dimensionality reduction |
| **PLS** | 3.4058 | 1.7972 | - | +0.13% RMSE | Underperformed |
| **XGBoost** | **2.7315** | **1.4011** | **0.9766** | **-19.41% RMSE** | **Best performance** |

### 9.2 Detailed Analysis

#### Linear Models Performance

All linear models (OLS, Ridge, PCR) achieve similar performance with RMSE around **3.40** and MAE around **1.77**, indicating:

- The relationship between features and PM2.5 is predominantly **linear**
- **Multicollinearity** between PM2.5 and PM10 does not significantly degrade predictive performance
- **Regularization** (Ridge) and **dimensionality reduction** (PCR) provide minimal improvement over simple OLS
- Removing PM10 improves **model stability** and **interpretability** without sacrificing accuracy

#### XGBoost Superior Performance

**XGBoost** demonstrates significant improvements:

- **RMSE Reduction**: 2.7315 vs 3.3891 (baseline) = **19.41% improvement**
- **MAE Reduction**: 1.4011 vs 1.7623 (baseline) = **20.49% improvement**
- **R² Score**: 0.9766, explaining **97.7%** of variance in PM2.5 concentrations
- **Feature Interactions**: XGBoost effectively captures non-linear relationships and complex feature interactions
- **PM10 Importance**: Despite multicollinearity in linear models, PM10 emerges as a top-3 important feature in XGBoost, demonstrating tree-based models' ability to leverage physical relationships

### 9.3 Model Selection Recommendation

For **production deployment**, **XGBoost** is recommended due to:
- Superior predictive accuracy
- Ability to handle non-linear patterns
- Feature importance interpretability
- Robust performance across different cities and time periods

For **baseline comparison** and **interpretability**, **OLS (PM2.5 only)** provides:
- Simple, interpretable coefficients
- Stable performance
- Fast training and inference
- Clear understanding of feature contributions

---

## 10. Limitations & Future Work

### 10.1 Limitations

- **Data Coverage**: The dataset covers 34 major cities, but may not fully represent rural or remote areas
- **Temporal Scope**: Data collection starts from January 2023, limiting long-term trend analysis
- **API Dependency**: Reliance on external API (Open-Meteo) for data collection
- **Feature Engineering**: Current features focus on meteorological and pollution variables; additional features (e.g., traffic data, industrial activity) could improve predictions
- **Model Interpretability**: While XGBoost provides feature importance, deep interpretability of complex interactions remains challenging

### 10.2 Future Work

- **Deep Learning Models**: Explore LSTM, GRU, or Transformer-based architectures for time-series forecasting
- **Ensemble Methods**: Combine multiple models (stacking, blending) to further improve accuracy
- **Real-time Forecasting**: Implement a production-ready system for real-time PM2.5 predictions
- **Extended Features**: Incorporate additional data sources such as traffic patterns, industrial emissions, and land use data
- **Multi-step Forecasting**: Extend prediction horizon beyond 24 hours
- **Regional Customization**: Develop region-specific models to account for local climate and emission patterns
- **Uncertainty Quantification**: Implement prediction intervals and confidence bounds for forecasts

---

## 11. Contact

For questions, suggestions, or collaborations, please contact team members via email:

| Name | Student ID | Email |
|:-----|:-----------|:------|
| Phạm Thành Nam | 23120301 | [23120301@student.hcmus.edu.vn](mailto:23120301@student.hcmus.edu.vn) |
| Trương Quang Phát | 23120318 | [23120318@student.hcmus.edu.vn](mailto:23120318@student.hcmus.edu.vn) |
| Huỳnh Tấn Phước | 23120334 | [23120334@student.hcmus.edu.vn](mailto:23120334@student.hcmus.edu.vn) |
| Trần Nguyễn Minh Quân | 23120342 | [23120342@student.hcmus.edu.vn](mailto:23120342@student.hcmus.edu.vn) |

---

## 12. References

[1] Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*, 29-39.

[2] Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice*. OTexts. Available at: https://otexts.com/fpp2/

[3] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: with applications in R*. Springer Science & Business Media. Official PDF available at: https://www.statlearning.com/

[4] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. Preprint available at: https://arxiv.org/abs/1603.02754

[5] Open-Meteo. (n.d.). Air Quality API Documentation. Retrieved from: https://open-meteo.com/

[6] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623-2631. Preprint available at: https://arxiv.org/abs/1907.10902

---

## License

This project is developed for academic purposes as part of the Introduction to Data Science course.

## Acknowledgments

- **Open-Meteo** for providing comprehensive air quality and weather data APIs
- **XGBoost** and **scikit-learn** communities for excellent machine learning libraries
- All contributors and team members for their valuable work

---

<div align="center">

**Made for better air quality forecasting in VietNam**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/TrNguyenMQuan/VietNam_PM2.5_Forecasting)

</div>
