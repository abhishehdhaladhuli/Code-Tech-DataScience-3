# CodeTech-DataScience-Task-3 Develop a full data science project from data collection and preprocessing to model deployment 
COMPANY: CODETECH IT SOLUTIONS<br>
NAME: ABHISHEK DHALADHULI<br>
INTERN ID: CT6WVEQ<br>
DOMAIN: DATA SCIENCE<br>
DURATION:6 WEEKS<br>
MENTOR:NEELA SANTHOSH<br>
This project implements a **machine learning-based web application** that classifies Iris flower species using a **Random Forest Classifier**. The solution includes **data preprocessing, model training, evaluation, persistence, and deployment**, making it an end-to-end machine learning pipeline. The trained model is integrated into a **Flask web application**, which has been successfully **deployed on Render.com**, allowing users to input flower measurements and receive predictions in real time.

---

## **1. Importing Required Libraries**
The script begins by importing essential libraries:

- **`pandas`** – For handling tabular data.
- **`sklearn.datasets`** – To load the **Iris dataset**.
- **`sklearn.model_selection`** – For splitting the dataset into **training and testing sets**.
- **`sklearn.ensemble`** – To use **RandomForestClassifier**.
- **`sklearn.metrics`** – For model evaluation.
- **`joblib`** – For saving and loading the trained model.
- **`flask`** – To create and serve the web application.

---

## **2. Loading and Preparing the Iris Dataset**
```python
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
```
- The **Iris dataset** is a well-known dataset containing **150 samples** with **four numerical features**:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- A **Pandas DataFrame** is created, and the target labels **(0, 1, 2)** are mapped to their respective **species names**.

---

## **3. Splitting the Dataset**
```python
X = df[data.feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
- The dataset is divided into **training (70%)** and **testing (30%)** sets using `train_test_split()`.

---

## **4. Training the RandomForest Model**
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
- A **Random Forest Classifier** with **100 trees** is trained on the dataset.
- This algorithm is effective for **classification tasks**, handling **non-linearity and feature importance**.

---

## **5. Evaluating the Model**
```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
- The trained model is evaluated on the test data.
- The **accuracy score** and a **detailed classification report** are displayed.

---

## **6. Saving the Model**
```python
joblib.dump(model, 'iris_model.pkl')
```
- The trained model is saved using `joblib.dump()`, allowing reuse without retraining.

---

## **7. Setting Up the Flask Web Application**
```python
app = Flask(__name__)
```
- A **Flask application** is initialized to create a web interface.

---

## **8. Defining Routes**
### **Home Route**
```python
@app.route('/')
def home():
    return render_template('index.html')
```
- This route serves the **index.html** page, where users can input feature values.

### **Prediction Route**
```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [
        float(data['sepal_length']),
        float(data['sepal_width']),
        float(data['petal_length']),
        float(data['petal_width'])
    ]
    model = joblib.load('iris_model.pkl')
    prediction = model.predict([features])[0]
    species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return render_template('index.html', prediction=species[prediction])
```
- The **user enters flower measurements** through a web form.
- The **saved model** (`iris_model.pkl`) is loaded.
- A **prediction is made**, and the species is displayed on the web page.

---

## **9. Running the Flask App**
```python
if __name__ == '__main__':
    app.run(debug=True, port=10000)
```
- The application runs on **port 10000**.
- `debug=True` enables **automatic code reloading** for development.

---

## **10. Deployment on Render.com**
The Flask application has been **deployed on Render.com**, making it publicly accessible. The deployment steps include:

1. **Push the code to GitHub**.
2. **Connect the repository to Render**.
3. **Create a new Flask web service** on Render.
4. **Specify the startup command** (`gunicorn app:app`).
5. **Deploy the application** and make it live.

Once deployed, users can **access the web app from anywhere**, input flower measurements, and receive instant predictions.

---

## **How the Web App Works**
1. **Users visit the deployed URL on Render.com**.
2. The home page (`index.html`) loads with a **form** to enter feature values.
3. When the user submits the form, a **POST request** is sent to `/predict`.
4. The Flask server:
   - Loads the **trained model**.
   - Predicts the **Iris species**.
   - Displays the **prediction result on the webpage**.

---

## **Key Features**
✅ **End-to-End Machine Learning Pipeline** – Includes **data processing, model training, evaluation, and deployment**.  
✅ **Model Persistence** – Uses `joblib` to **save and reuse the trained model**.  
✅ **Flask-Based Web App** – Provides a **user-friendly interface** for input and prediction.  
✅ **Deployed on Render.com** – Hosted for **public access and real-time predictions**.  

---

## **Conclusion**
This project successfully builds and deploys an **Iris classification web app** using **Flask** and **Random Forest**. The application is **publicly accessible via Render.com**, making it a practical, real-world example of **machine learning model deployment**. 🚀
