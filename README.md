# 🚗 Car Price Prediction using Machine Learning

A complete end-to-end machine learning project that predicts the **Manufacturer's Suggested Retail Price (MSRP)** of a vehicle from its specifications.

The project includes data preprocessing, categorical encoding, model training, evaluation, model serialization, an interactive Streamlit application, Git/GitHub version control, and cloud deployment.

## 🌐 Live Application

**Car Price Predictor:**  
https://car-price-predictor-cpp.streamlit.app/

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Project Objectives](#-project-objectives)
- [Features](#-features)
- [Dataset](#-dataset)
- [Target Variable](#-target-variable)
- [Input Features](#-input-features)
- [Data Preprocessing](#-data-preprocessing)
- [Feature Types](#-feature-types)
- [Categorical Encoding](#-categorical-encoding)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Decision Tree Regressor](#-decision-tree-regressor)
- [Model Evaluation](#-model-evaluation)
- [Model Serialization](#-model-serialization)
- [Saved Files](#-saved-files)
- [Streamlit Application](#-streamlit-application)
- [Dynamic Make-to-Model Selection](#-dynamic-make-to-model-selection)
- [Application Workflow](#-application-workflow)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Virtual Environment](#-virtual-environment)
- [Run Locally](#-run-locally)
- [Using the Application](#-using-the-application)
- [Git and GitHub](#-git-and-github)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Learning Outcomes](#-learning-outcomes)
- [Author](#-author)
- [License](#-license)

---

# 📌 Project Overview

Car prices depend on many factors, including the manufacturer, model, year, engine specifications, transmission, drivetrain, vehicle size, vehicle style, fuel economy, and popularity.

This project applies machine learning to learn the relationship between these vehicle attributes and their corresponding MSRP values.

The trained model is integrated into a Streamlit web application so that users can enter vehicle specifications and receive an estimated MSRP.

The project demonstrates a complete machine learning deployment workflow:

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Categorical Encoding
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
Streamlit Application
     ↓
Cloud Deployment
```

---

# 🌐 Live Demo

The trained model is available through an interactive Streamlit application.

### 🚗 Car Price Predictor

**https://car-price-predictor-cpp.streamlit.app/**

Users can enter vehicle specifications and obtain an estimated MSRP.

---

# 🎯 Project Objectives

The main objectives of this project are:

1. Understand a real-world regression problem.
2. Clean and preprocess vehicle data.
3. Handle missing values.
4. Separate numerical and categorical features.
5. Encode categorical variables.
6. Build a machine learning pipeline.
7. Train a Decision Tree Regressor.
8. Evaluate the regression model.
9. Save the trained model using Joblib.
10. Build an interactive Streamlit application.
11. Create a dynamic Make → Model selection system.
12. Host the project on GitHub.
13. Deploy the application using Streamlit Community Cloud.

---

# ✨ Features

## 🚘 Car Information

The application accepts:

- Make
- Model
- Year

The Model dropdown dynamically changes according to the selected Make.

## ⚙️ Engine Information

The application accepts:

- Engine Fuel Type
- Engine HP
- Engine Cylinders
- Transmission Type

## 🚙 Vehicle Information

The application accepts:

- Driven Wheels
- Number of Doors
- Vehicle Size
- Vehicle Style

## ⛽ Fuel Economy

The application accepts:

- Highway MPG
- City MPG

## 📊 Other Information

The application also accepts:

- Popularity

## 🔮 MSRP Prediction

After entering the required vehicle information, users can click:

```text
🔮 Predict Car Price
```

The trained model processes the input and returns an estimated MSRP.

---

# 📊 Dataset

The original vehicle dataset contains information about vehicle specifications and their MSRP.

The original dataset included the following fields:

```text
Make
Model
Year
Engine Fuel Type
Engine HP
Engine Cylinders
Transmission Type
Driven_Wheels
Number of Doors
Market Category
Vehicle Size
Vehicle Style
highway MPG
city mpg
Popularity
MSRP (Manufacturer's suggested retail Price)
```

The final version of the project does **not** use `Market Category`.

The final model therefore uses 14 input features.

---

# 🎯 Target Variable

The target variable is:

```text
MSRP (Manufacturer's suggested retail Price)
```

Since MSRP is a continuous numerical value, the problem is treated as a **regression problem**.

---

# 📋 Input Features

The final model uses these 14 input features:

| Feature | Type | Used by Model |
|---|---|---|
| Make | Categorical | Yes |
| Model | Categorical | Yes |
| Year | Numerical | Yes |
| Engine Fuel Type | Categorical | Yes |
| Engine HP | Numerical | Yes |
| Engine Cylinders | Numerical | Yes |
| Transmission Type | Categorical | Yes |
| Driven_Wheels | Categorical | Yes |
| Number of Doors | Numerical | Yes |
| Vehicle Size | Categorical | Yes |
| Vehicle Style | Categorical | Yes |
| highway MPG | Numerical | Yes |
| city mpg | Numerical | Yes |
| Popularity | Numerical | Yes |

---

# 🧹 Data Preprocessing

Missing values were handled before training.

## Engine Fuel Type

Missing values were replaced with the mode.

```python
df["Engine Fuel Type"].fillna(
    df["Engine Fuel Type"].mode()[0],
    inplace=True
)
```

## Engine HP

Missing values were replaced with the mean.

```python
df["Engine HP"].fillna(
    df["Engine HP"].mean(),
    inplace=True
)
```

## Engine Cylinders

Missing values were replaced with the mode.

```python
df["Engine Cylinders"].fillna(
    df["Engine Cylinders"].mode()[0],
    inplace=True
)
```

## Number of Doors

Missing values were replaced with the mode.

```python
df["Number of Doors"].fillna(
    df["Number of Doors"].mode()[0],
    inplace=True
)
```

---

# 🔢 Feature Types

## Categorical Features

```text
Make
Model
Engine Fuel Type
Transmission Type
Driven_Wheels
Vehicle Size
Vehicle Style
```

## Numerical Features

```text
Year
Engine HP
Engine Cylinders
Number of Doors
highway MPG
city mpg
Popularity
```

---

# 🔤 Categorical Encoding

Machine learning algorithms require numerical representations of categorical data.

The project uses Scikit-learn's `OrdinalEncoder`:

```python
OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)
```

The categorical features are:

```text
Make
Model
Engine Fuel Type
Transmission Type
Driven_Wheels
Vehicle Size
Vehicle Style
```

The `handle_unknown="use_encoded_value"` and `unknown_value=-1` configuration allows the pipeline to represent an unseen categorical value without immediately failing during prediction.

---

# 🔢 Numerical Feature Processing

The numerical features are passed directly through the preprocessing stage.

The final model does **not** use `StandardScaler`.

The numerical features are:

```text
Year
Engine HP
Engine Cylinders
Number of Doors
highway MPG
city mpg
Popularity
```

---

# 🏗️ ColumnTransformer

A `ColumnTransformer` is used to process numerical and categorical columns differently.

```text
                    Input Data
                        │
             ┌──────────┴──────────┐
             │                     │
      Numerical Features    Categorical Features
             │                     │
        Passthrough            OrdinalEncoder
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
              Decision Tree Regressor
                        │
                        ▼
                  MSRP Prediction
```

The categorical column positions are:

```python
[0, 1, 3, 6, 7, 9, 10]
```

Corresponding to:

```text
Make
Model
Engine Fuel Type
Transmission Type
Driven_Wheels
Vehicle Size
Vehicle Style
```

The numerical column positions are:

```python
[2, 4, 5, 8, 11, 12, 13]
```

Corresponding to:

```text
Year
Engine HP
Engine Cylinders
Number of Doors
highway MPG
city mpg
Popularity
```

---

# 🌳 Decision Tree Regressor

The final model uses:

```python
DecisionTreeRegressor(random_state=42)
```

A Decision Tree Regressor learns relationships between the input features and the continuous target value.

The trained model is placed inside a complete Scikit-learn pipeline so that the same preprocessing used during training is automatically applied during prediction.

Conceptually:

```python
model = make_pipeline(
    transformer,
    DecisionTreeRegressor(random_state=42)
)
```

---

# 🔄 Machine Learning Pipeline

```text
Vehicle Dataset
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Feature / Target Separation
      ↓
Train-Test Split
      ↓
ColumnTransformer
      ↓
 ┌─────────────────────────┐
 │ Numerical → Passthrough │
 │ Categorical → Ordinal   │
 │ Encoder                 │
 └─────────────────────────┘
      ↓
Decision Tree Regressor
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Joblib Serialization
      ↓
Streamlit Deployment
```

---

# 📈 Model Evaluation

Since this is a regression problem, suitable evaluation metrics include:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Example:

```python
from sklearn.metrics import r2_score

y_train_pred = model.predict(X_train)

r2 = r2_score(
    y_train,
    y_train_pred
)

print("Training R² Score:", r2)
```

The exact evaluation values depend on the dataset split and model configuration.

A proper evaluation should consider both training and test performance rather than relying only on training R².

---

# 💾 Model Serialization

The trained model is saved using Joblib:

```python
import joblib

joblib.dump(
    model,
    "car_price_model.pkl"
)
```

This allows the trained model to be reused without retraining every time the Streamlit application starts.

---

# 📦 Saved Files

## `car_price_model.pkl`

Contains the trained machine learning pipeline, including:

- ColumnTransformer
- OrdinalEncoder
- Decision Tree Regressor

Loaded using:

```python
model = joblib.load("car_price_model.pkl")
```

## `car_categories.pkl`

Stores:

```text
Engine Fuel Type
Transmission Type
Driven_Wheels
Vehicle Size
Vehicle Style
```

Loaded using:

```python
categories = joblib.load("car_categories.pkl")
```

## `make_model_mapping.pkl`

Stores the Make → Model relationship.

Generated using:

```python
make_model_mapping = (
    df.groupby("Make")["Model"]
    .unique()
    .apply(list)
    .to_dict()
)
```

Saved using:

```python
joblib.dump(
    make_model_mapping,
    "make_model_mapping.pkl"
)
```

Loaded using:

```python
make_model_mapping = joblib.load(
    "make_model_mapping.pkl"
)
```

---

# 🖥️ Streamlit Application

The interface is organized into:

```text
🚘 Car Information
⚙️ Engine Information
🚙 Vehicle Information
⛽ Fuel Economy
📊 Other Information
🔮 Prediction
```

The application loads the trained model and supporting files when it starts.

---

# 🔗 Dynamic Make → Model Selection

The application uses a dependent dropdown system.

```python
make = st.selectbox(
    "Make",
    sorted(make_model_mapping.keys()),
    key="make_selectbox"
)

available_models = sorted(
    make_model_mapping[make]
)

model_name = st.selectbox(
    "Model",
    available_models,
    key="model_selectbox"
)
```

This means that selecting a manufacturer automatically changes the available models.

---

# 🔄 Application Workflow

```text
User opens application
        ↓
Select Make
        ↓
Available Models update
        ↓
Select Model
        ↓
Enter Year
        ↓
Enter Engine Information
        ↓
Enter Vehicle Information
        ↓
Enter Fuel Economy
        ↓
Enter Popularity
        ↓
Click "Predict Car Price"
        ↓
Input converted to DataFrame
        ↓
Saved ML Pipeline processes input
        ↓
Decision Tree predicts MSRP
        ↓
Estimated MSRP displayed
```

---

# 📂 Project Structure

```text
CarPricePrediction/
│
├── app.py
├── car_price_model.pkl
├── car_categories.pkl
├── make_model_mapping.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📄 File Description

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `car_price_model.pkl` | Trained ML pipeline |
| `car_categories.pkl` | Categorical values for UI |
| `make_model_mapping.pkl` | Make → Model mapping |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git exclusions |
| `README.md` | Project documentation |

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Decision Tree Regression
- ColumnTransformer
- OrdinalEncoder
- Joblib
- Streamlit
- Git
- GitHub
- Streamlit Community Cloud

---

# 📦 Requirements

`requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/CarPricePrediction.git
```

Move into the project:

```bash
cd CarPricePrediction
```

---

# 🐍 Virtual Environment

## Windows

```bash
python -m venv CarPricePredEnv
CarPricePredEnv\Scriptsctivate
```

## Linux / macOS

```bash
python3 -m venv CarPricePredEnv
source CarPricePredEnv/bin/activate
```

---

# 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

---

# ▶️ Run Locally

```bash
streamlit run app.py
```

If needed:

```bash
python -m streamlit run app.py
```

The local application normally opens at:

```text
http://localhost:8501
```

---

# 🔮 How to Use

1. Select a **Make**.
2. Select a **Model** from the dynamically updated list.
3. Enter the **Year**.
4. Select the **Engine Fuel Type**.
5. Enter **Engine HP**.
6. Enter **Engine Cylinders**.
7. Select **Transmission Type**.
8. Select **Driven Wheels**.
9. Select the **Number of Doors**.
10. Select **Vehicle Size**.
11. Select **Vehicle Style**.
12. Enter **Highway MPG**.
13. Enter **City MPG**.
14. Enter **Popularity**.
15. Click **Predict Car Price**.
16. View the estimated MSRP.

---

# 🧪 Example Prediction Workflow

Example input:

```text
Make:
Honda

Model:
Civic

Year:
2015

Engine Fuel Type:
Regular Unleaded

Engine HP:
140

Engine Cylinders:
4

Transmission Type:
Automatic

Driven Wheels:
Front Wheel Drive

Number of Doors:
4

Vehicle Size:
Compact

Vehicle Style:
Sedan

Highway MPG:
35

City MPG:
28

Popularity:
1000
```

Processing:

```text
User Input
    ↓
Pandas DataFrame
    ↓
ColumnTransformer
    ↓
Ordinal Encoding
    ↓
Decision Tree Regressor
    ↓
Estimated MSRP
```

---

# ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

Architecture:

```text
Local Development
       ↓
Git
       ↓
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Live Application
```

### Live Deployment

https://car-price-predictor-cpp.streamlit.app/

### Deployment Configuration

```text
Repository: CarPricePrediction
Branch: main
Main file: app.py
```

Streamlit installs the packages from `requirements.txt`.

The following files must be available in the repository:

```text
app.py
car_price_model.pkl
car_categories.pkl
make_model_mapping.pkl
requirements.txt
```

---

# 🔧 Git and GitHub

The project uses Git for version control.

Initialize:

```bash
git init
```

Stage files:

```bash
git add .
```

Commit:

```bash
git commit -m "Initial commit - car price prediction app"
```

Connect GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/CarPricePrediction.git
```

Rename the branch:

```bash
git branch -M main
```

Push:

```bash
git push -u origin main
```

---

# 📋 `.gitignore`

Recommended `.gitignore`:

```text
CarPricePredEnv/
__pycache__/
*.pyc
```

This prevents the virtual environment and Python cache files from being committed.

---

# ⚠️ Limitations

The application provides an **estimated MSRP**, not a guaranteed market price.

Actual vehicle prices may differ because factors such as the following are not included:

- Vehicle condition
- Mileage
- Location
- Dealer pricing
- Taxes
- Registration charges
- Discounts
- Optional features
- Market demand
- Vehicle availability
- Depreciation
- Current market conditions

Therefore, predictions should be treated as machine-learning estimates.

---

# 🔒 Data and Model Considerations

The model is trained on historical vehicle information.

Vehicle prices can change over time, so historical patterns may not perfectly represent current prices.

This project is primarily intended for:

- Educational purposes
- Machine learning demonstrations
- Portfolio development
- Regression practice
- Streamlit deployment practice

---

# 🔮 Future Improvements

Potential improvements include:

### 1. Additional Regression Models

- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- XGBoost
- CatBoost

### 2. Hyperparameter Tuning

Use:

- GridSearchCV
- RandomizedSearchCV

Possible parameters:

```text
max_depth
min_samples_split
min_samples_leaf
max_features
```

### 3. Cross-Validation

Use K-Fold Cross-Validation for more reliable model evaluation.

### 4. Feature Importance

Display the features that contribute most to predictions.

### 5. Better Evaluation

Display:

```text
R²
MAE
MSE
RMSE
```

for training and testing data.

### 6. Improved UI

Possible additions:

- Vehicle images
- Interactive charts
- Prediction history
- Improved styling
- Mobile optimization

### 7. Prediction Range

Display an estimated price range instead of only one value.

### 8. More Vehicle Features

Possible additions:

- Mileage
- Vehicle condition
- Safety features
- Interior features
- Exterior features
- Location
- Current market information

### 9. Model Retraining

Periodically retrain using newer vehicle data.

---

# 🐛 Troubleshooting

## FileNotFoundError

Ensure these files exist beside `app.py`:

```text
car_price_model.pkl
car_categories.pkl
make_model_mapping.pkl
```

## ModuleNotFoundError

Run:

```bash
pip install -r requirements.txt
```

## Streamlit Command Not Found

Use:

```bash
python -m streamlit run app.py
```

## Duplicate Streamlit Element ID

Give widgets unique keys:

```python
make = st.selectbox(
    "Make",
    sorted(make_model_mapping.keys()),
    key="make_selectbox"
)
```

```python
model_name = st.selectbox(
    "Model",
    available_models,
    key="model_selectbox"
)
```

Also ensure there is only one Model selectbox in `app.py`.

---

# 📚 Concepts Demonstrated

## Python

- Python programming
- Functions
- Data structures
- File handling
- Libraries

## Pandas

- Data loading
- Data inspection
- Missing-value handling
- Feature selection
- Data manipulation

## NumPy

- Numerical operations
- Array handling

## Scikit-learn

- Train-test splitting
- ColumnTransformer
- OrdinalEncoder
- Pipeline
- Decision Tree Regression
- Regression evaluation

## Joblib

- Model serialization
- Saving models
- Loading models

## Streamlit

- Web application development
- Input widgets
- Selectboxes
- Dynamic dropdowns
- Prediction interface
- Cloud deployment

## Git and GitHub

- Repository initialization
- Staging
- Commits
- Remote repositories
- GitHub hosting

---

# 🎓 Learning Outcomes

This project demonstrates the ability to:

1. Understand a real-world regression problem.
2. Inspect a machine learning dataset.
3. Identify numerical and categorical features.
4. Handle missing values.
5. Separate features and target variables.
6. Split data for machine learning.
7. Encode categorical variables.
8. Build a preprocessing pipeline.
9. Train a Decision Tree Regressor.
10. Evaluate a regression model.
11. Serialize a trained model.
12. Load a trained model for inference.
13. Build an interactive Streamlit application.
14. Create dependent dropdown menus.
15. Use Git for version control.
16. Host a project on GitHub.
17. Deploy a machine learning application to the cloud.

---

# 🧩 Application Architecture

```text
┌─────────────────────────────┐
│       Streamlit UI          │
│                             │
│  Make / Model / Year        │
│  Engine Information         │
│  Vehicle Information        │
│  Fuel Economy               │
│  Popularity                 │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Input DataFrame       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      ML Preprocessing       │
│                             │
│ Numerical → Passthrough     │
│ Categorical → Encoder       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Decision Tree Regressor  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Estimated MSRP        │
└─────────────────────────────┘
```

---

# 🔗 Important Links

## 🌐 Live Application

https://car-price-predictor-cpp.streamlit.app/

## 💻 GitHub Repository

Replace the placeholder with your actual repository:

```text
https://github.com/YOUR_USERNAME/CarPricePrediction
```

---

# 👨‍💻 Author

## Surya

Machine Learning and Python project demonstrating an end-to-end workflow from data preprocessing and model training to web application development and cloud deployment.

### Skills Demonstrated

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Regression
- Decision Trees
- Data Preprocessing
- Feature Encoding
- Streamlit
- Joblib
- Git
- GitHub
- Cloud Deployment

---

# 📜 License

This project is intended primarily for educational, demonstration, and portfolio purposes.

You may study, modify, and extend the project for learning and development.

---

# ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

# 🚗 Live Project

**Try the Car Price Predictor:**

https://car-price-predictor-cpp.streamlit.app/
