# Breast Cancer Diagnosis Using Machine Learning

This project demonstrates the application of machine learning algorithms to diagnose breast cancer using the Wisconsin Diagnostic Breast Cancer dataset. The goal is to accurately classify tumors as malignant or benign based on various cellular features.

The Jupyter notebook (`jenis (1).py`) performs a complete data analysis and modeling pipeline. It includes:
- Data loading and preprocessing
- Exploratory data analysis and visualizations
- Feature scaling and train-test splitting
- Training and evaluation of four models: Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine
- Accuracy comparison and confusion matrix visualization
- Custom functions for real-time predictions using user input

The Flask web app (`app.py`) enables user interaction via a web interface. It uses a pre-trained Logistic Regression model and scaler (saved as `.pkl` files) to:
![image](https://github.com/user-attachments/assets/cd42e67b-6c3d-440e-8dcb-a55e8fec0353)
- Accept user input for all 30 features
- Preprocess and scale the input
- Predict whether the tumor is benign or malignant
- Display the result with a confidence level

This project is ideal for beginners exploring real-world applications of supervised learning in healthcare, particularly in cancer detection.

## Technologies Used
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn for ML models
- Flask for web deployment
- Joblib for model serialization

## How to Use
1. Clone the repository
2. Train the models using the notebook
3. Run the Flask app to interact with the prediction tool via browser
