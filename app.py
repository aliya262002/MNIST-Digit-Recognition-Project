# MNIST Digit Recognition Project
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import gradio as gr
from PIL import Image

# 1. Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(np.uint8)

# 2. Split data (60k train, 10k test)
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=5000, random_state=42)

# 3. Scale data for SGD
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Train classifiers
print("\nTraining classifiers...")

# SGD Classifier
print("Training SGD Classifier...")
sgd_clf = SGDClassifier(loss='hinge', alpha=0.001, max_iter=1000, random_state=42)
sgd_clf.fit(X_train_scaled, y_train)

# Random Forest Classifier
print("Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_clf.fit(X_train, y_train)  # RF doesn't need scaled data

# 5. Evaluate models
def evaluate_model(model, X, y, model_name, scaled=False):
    if scaled:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    return y_pred

print("\nEvaluating SGD Classifier:")
y_test_pred_sgd = evaluate_model(sgd_clf, X_test, y_test, "SGD Classifier", scaled=True)

print("\nEvaluating Random Forest Classifier:")
y_test_pred_rf = evaluate_model(rf_clf, X_test, y_test, "Random Forest Classifier")

# 6. Visualize errors
print("\nVisualizing worst misclassifications...")
errors = np.where(y_test != y_test_pred_rf)[0]
error_samples = errors[:5]  # Show top 5 errors

plt.figure(figsize=(12, 5))
for i, idx in enumerate(error_samples):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[idx]}\nPred: {y_test_pred_rf[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 7. Gradio web app 
def recognize_digit(image):
    try:
        # Convert to grayscale and resize to 28x28
        img = Image.fromarray(image).convert('L').resize((28, 28))
        img_array = np.array(img).reshape(1, -1) / 255.0  # Normalize
        
        # Get predictions
        sgd_pred = str(sgd_clf.predict(scaler.transform(img_array))[0])
        rf_pred = str(rf_clf.predict(img_array)[0])
        rf_probs = {str(i): float(rf_clf.predict_proba(img_array)[0][i]) for i in range(10)}
        
        return sgd_pred, rf_pred, rf_probs
    except Exception as e:
        return "Error", "Error", {}

print("\nLaunching Gradio interface...")
interface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Image(), 
    outputs=[
        gr.Textbox(label="SGD Prediction"),
        gr.Textbox(label="Random Forest Prediction"), 
        gr.Label(label="Class Probabilities")
    ],
    title="MNIST Digit Classifier",
    description="Upload a 28x28 image of a digit (0-9)"
)

interface.launch()  