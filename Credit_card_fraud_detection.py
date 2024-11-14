import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDetectionModel:
    def __init__(self):
        self.supervised_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        self.anomaly_model = IsolationForest(
            contamination=0.01,  # Expected fraction of outliers
            random_state=42
        )
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess the data including scaling and handling imbalanced classes
        """
        # Scale the features
        if training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        if training and y is not None:
            # Apply SMOTE for supervised learning
            X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)
            return X_resampled, y_resampled
        
        return X_scaled
    
    def train_supervised(self, X, y):
        """
        Train the supervised learning model
        """
        # Preprocess the data
        X_resampled, y_resampled = self.preprocess_data(X, y, training=True)
        
        # Train the model
        self.supervised_model.fit(X_resampled, y_resampled)
    
    def train_anomaly(self, X):
        """
        Train the anomaly detection model
        """
        # Preprocess the data
        X_scaled = self.preprocess_data(X, training=True)
        
        # Train the model
        self.anomaly_model.fit(X_scaled)
    
    def predict_supervised(self, X):
        """
        Make predictions using the supervised model
        """
        X_scaled = self.preprocess_data(X, training=False)
        return self.supervised_model.predict(X_scaled)
    
    def predict_anomaly(self, X):
        """
        Make predictions using the anomaly detection model
        """
        X_scaled = self.preprocess_data(X, training=False)
        # Convert IsolationForest predictions (-1 for outliers, 1 for inliers)
        # to 1 for fraudulent (outliers) and 0 for normal (inliers)
        predictions = self.anomaly_model.predict(X_scaled)
        return np.where(predictions == -1, 1, 0)
    
    def evaluate_model(self, X_test, y_test, model_type='supervised'):
        """
        Evaluate the model and print metrics
        """
        if model_type == 'supervised':
            y_pred = self.predict_supervised(X_test)
        else:
            y_pred = self.predict_anomaly(X_test)
        
        # Print classification report
        print(f"\n{model_type.capitalize()} Model Performance:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_type.capitalize()} Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

# Example usage
def main():
    # Load and prepare your data
    # This is an example - replace with your actual data loading logic
    # data = pd.read_csv('credit_card_transactions.csv')
    # X = data.drop('is_fraud', axis=1)
    # y = data['is_fraud']
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 10000
    n_features = 10
    
    # Create synthetic data with 1% fraud rate
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train the model
    model = FraudDetectionModel()
    
    # Train and evaluate supervised model
    model.train_supervised(X_train, y_train)
    model.evaluate_model(X_test, y_test, model_type='supervised')
    
    # Train and evaluate anomaly detection model
    model.train_anomaly(X_train)
    model.evaluate_model(X_test, y_test, model_type='anomaly')

if __name__ == "__main__":
    main()

#dataset; https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets