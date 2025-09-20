# Data Analysis and Visualization Tools

This repository provides interactive Python applications built with **Tkinter**, **Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-learn**.  
It includes GUI-based tools for **data analysis**, **statistical modeling**, and **data visualization**.

---

## 📂 Files in the Repository

### 1. `data_analysis_app_v1.py`
- A Tkinter-based GUI for:
  - Loading CSV files
  - Performing:
    - **K-Means Clustering** (on 2 columns)
    - **Linear Regression** (1 independent, 1 dependent variable)
    - **Multiple Linear Regression** (multiple independent, 1 dependent variable)
  - Visualization of results using **Matplotlib**.

---

### 2. `data_analysis_app_v2.py`
- A **refactored and documented version** of the first file.
- Features include:
  - Cleaner code with docstrings
  - Same analysis functionalities (**K-Means, Linear Regression, Multiple Linear Regression**)
  - Embedded interactive plots within the Tkinter window.

---

### 3. `data_visualizer_app.py`
- A **flexible data visualization tool** with a Tkinter GUI.
- After uploading a CSV, users can generate a variety of plots:
  - **2D Plots**: Histogram, Bar, Box, Scatter, Line, Area, Pie, Violin
  - **Multivariate Analysis**: Pair Plot, Correlation Heatmap
  - **Time Series Analysis**: Line plot, Rolling average, Seasonal decomposition
  - **3D Visualizations**: 3D Scatter, 3D Line, 3D Surface plots
- Powered by **Seaborn**, **Matplotlib**, and **Statsmodels**.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed, then install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### 3. Run the Applications
Run any of the apps using:
```bash
python data_analysis_app_v1.py
```
or
```bash
python data_analysis_app_v2.py
```
or
```bash
python data_visualizer_app.py
```

---

## 🛠️ Features Overview

- **Data Loading**: Upload CSV files easily via a file dialog.
- **Data Analysis**:
  - K-Means clustering with visualization
  - Simple linear regression with regression line
  - Multiple regression with predicted vs actual plots
- **Data Visualization**:
  - Quick plots for exploratory data analysis
  - Time series and decomposition for trend analysis
  - Advanced 3D plots for multivariate data

---

## 📊 Example Use Cases
- Explore and visualize datasets without coding.
- Perform quick clustering or regression analysis.
- Generate professional plots for reports and presentations.
- Understand relationships and correlations in datasets.

---

## 📝 Notes
- Ensure your CSV has proper column names for analysis.
- For regression:
  - Independent variables must be **numeric**.
  - Dependent variable should be **continuous numeric data**.
- For 3D plots, provide exactly three numeric columns.

---

## 👨‍💻 Author
Developed as an educational tool to simplify **data analysis and visualization** for non-programmers and beginners in Python.
