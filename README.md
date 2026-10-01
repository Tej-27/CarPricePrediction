# 🚗 Car Price Prediction using Machine Learning

A machine learning web application that predicts the **Manufacturer's Suggested Retail Price (MSRP)** of a car based on its specifications.

The project uses a **Decision Tree Regression** model with categorical feature encoding and is deployed as an interactive **Streamlit web application**.

## 🌐 Live Application

🔗 **Try the Car Price Predictor:**  
https://car-price-predictor-cpp.streamlit.app/

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Machine Learning Model](#-machine-learning-model)
- [Dataset](#-dataset)
- [Features Used](#-features-used)
- [Data Preprocessing](#-data-preprocessing)
- [Categorical Feature Encoding](#-categorical-feature-encoding)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Training](#-model-training)
- [Saved Model Files](#-saved-model-files)
- [Streamlit Application](#-streamlit-application)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Running the Application Locally](#-running-the-application-locally)
- [How to Use](#-how-to-use)
- [Example Prediction Workflow](#-example-prediction-workflow)
- [Model Evaluation](#-model-evaluation)
- [Deployment](#-deployment)
- [Git and GitHub](#-git-and-github)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Troubleshooting](#-troubleshooting)
- [Author](#-author)
- [License](#-license)

---

# 📌 Project Overview

Car prices depend on several factors such as the manufacturer, model, manufacturing year, engine specifications, transmission, drivetrain, vehicle type, fuel economy, and popularity.

This project uses **Machine Learning** to learn the relationship between these vehicle attributes and their corresponding **Manufacturer's Suggested Retail Price (MSRP)**.

The trained model takes the specifications of a vehicle as input and predicts an estimated MSRP.

The project demonstrates an end-to-end machine learning workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Categorical Encoding
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Web Application
   ↓
Cloud Deployment
