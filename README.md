# H1B Visa Approval Prediction

A machine learning project that analyzes historical H1B visa petition data to predict the approval status (Certified/Denied) for new applications using a Logistic Regression model, deployed via a user-friendly Streamlit web application.

## 🌟 Key Features

  * **Data Analysis:** Exploratory Data Analysis (EDA) on a large dataset of H1B visa petitions from 2011 to 2016.
  * **Model Training:** Implementation of a **Logistic Regression** classifier for a multi-class prediction task.
  * **Feature Engineering:** Categorization of the `SOC_NAME` (Occupation) feature into numerical labels (`SOC_N`).
  * **Interactive Web App:** A user interface built with **Streamlit** that allows users to input application details and get an instant prediction.
  * **Prediction Output:** Provides a clear prediction (**Certified/Denied**) along with the probability scores for each class.

-----

## 💻 Technology Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python |
| **ML Libraries** | `scikit-learn` (for model training/serialization) |
| **Data Processing** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Web Framework** | `Streamlit` |
| **Serialization** | `pickle` (for saving the model and encoders) |

-----

## 📊 Data & Analysis Highlights

The core of this project is based on a dataset containing approximately 3 million H1B petition records from 2011 to 2016.

### Data Insights from the Notebook:

  * **Case Status:** The vast majority of applications were **CERTIFIED** ($\approx 87\%$), with a smaller percentage being **DENIED** ($\approx 3\%$).
  * **Time Series Trend:** The number of H1B applications showed an **exponential increase** between 2011 and 2016.
  * **Employment Type:** Approximately **85%** of the jobs applied for were **FULL-TIME** positions.
  * **Top Applicants:** Companies like **INFOSYS** and **TATA** were among the top applicants, and the data showed significant application volume from India-based IT service providers.

### Key Features Used for Prediction:

The final model used the following features after data cleaning and engineering:

1.  **`FULL_TIME_POSITION`** (0 for Part-Time, 1 for Full-Time)
2.  **`PREVAILING_WAGE`** (Entered as a numerical value)
3.  **`YEAR`** (The year of application)
4.  **`SOC_N`** (A numerical encoding of the general occupation/job title)

-----

## 🧠 Machine Learning Model

### Algorithm:

The model is a **Logistic Regression** classifier.

### Preprocessing:

  * The raw `CASE_STATUS` (target variable) was **Label Encoded** into numerical classes (0: CERTIFIED, 1: CERTIFIED-WITHDRAWN, 2: DENIED, etc.).
  * The `FULL_TIME_POSITION` (`Y`/`N`) was mapped to **1/0**.
  * The generalized occupation group (`SOC_NAME1` $\rightarrow$ `SOC_N`) was **Label Encoded**.

### Model Performance:

The classification report showed a high **Accuracy of $\approx 87\%$** on the test set. However, since the `CERTIFIED` class (0) is overwhelmingly dominant, the performance for the minority classes (like `DENIED`) was low, which is a common issue in heavily imbalanced datasets.

-----

## 🚀 Installation & Running the App

To run the Streamlit web application locally, follow these steps:

### Prerequisites

You need **Python** installed (version 3.7+ recommended).

### 1\. Clone the Repository

```bash
git clone [YOUR-GITHUB-REPO-URL-HERE]
cd h1b-visa-prediction-ml
```

### 2\. Install Dependencies

It's recommended to use a virtual environment.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# .\venv\Scripts\activate # On Windows

# Install required packages
pip install streamlit pandas scikit-learn
```

### 3\. Run the Streamlit Application

Start the application from your terminal:

```bash
streamlit run app.py
```

Your web browser should automatically open to the application (usually at `http://localhost:8501`).

### 4\. Making a Prediction

The app provides input fields for:

  * **Job Title** (The app currently uses a static list but maps to the `SOC_N` feature).
  * **Full Time or Part Time?**
  * **SOC\_N Code** (The Standard Occupational Classification code).
  * **Prevailing Wage**
  * **Year**

After entering the values, click the **"🔍 Predict"** button to see the result (e.g., "Prediction: Denied," with probabilities).

-----

## 📁 Project Files

  * **`h-1b-visa-prediction-using-machine-learning.ipynb`**: The Jupyter Notebook containing the full data cleaning, EDA, feature engineering, model training, and evaluation process.
  * **`visa_model.pkl`**: The serialized Logistic Regression model object.
  * **`le_ftp.pkl`**: The serialized `LabelEncoder` object used to convert `FULL_TIME_POSITION` to a numerical value.
  * **`app.py`**: The Python script for the Streamlit web application interface.
Here are some images:
<img width="889" height="600" alt="Screenshot 2025-10-23 202639" src="https://github.com/user-attachments/assets/ae95af88-2d27-460d-8432-71e128d4d6a8" />
<img width="815" height="204" alt="Screenshot 2025-10-23 202728" src="https://github.com/user-attachments/assets/16a3fe88-9dea-462c-af64-5e7024b0b8bd" />


