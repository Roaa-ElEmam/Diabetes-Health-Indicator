# Diabetes Health Indicator

## Description
This project detects if the user has diabetes or not. The user inputs his information like BMI (Body Mass Index), Cholestrol Check, Physical Health, etc... in a form, and using our training model, we will be predicting if that user has Diabetes or not.

## Installation
You need to download the python libraries in the requirements.txt file, which are:
```bash
flask==2.0.1
pandas==1.3.3
scikit-learn==0.24.2
numpy==1.21.2
```

What each library is for:
* Flask: For building the web application (the backend server).
* pandas: For data manipulation and creating DataFrames.
* scikit-learn: For machine learning (training and using the RandomForestClassifier).
* numpy: For numerical operations (used by pandas and scikit-learn).

You can install all of them at once by running this line in your bash:
```bash
pip install -r requirements.txt
```
## Run the WebApplication
To run the the code, go to the terminal and write:
```bash
python train_model.py
python app.py
```

Then, open your browser and go to ```http://127.0.0.1:5000``` to check the UI.
