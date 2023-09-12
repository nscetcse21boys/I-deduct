from flask import Blueprint, render_template, request,Flask, redirect, url_for, render_template,request,session
from flask_login import login_required, current_user
import csv


views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    return render_template("home.html", user=current_user)

@views.route('/cal_h', methods=['GET', 'POST'])
def cal_h():
    if request.method == "POST":
        age=request.form["age"]
        sex=request.form["sex"]
        cp=request.form["cp"]
        trestbps=request.form["trestbps"]
        chol=request.form["chol"]
        fbs=request.form["fbs"]
        restecg=request.form["restecg"]
        thalach=request.form["thalach"]
        oldpeak=request.form["oldpeak"]
        import numpy as np
        features = np.array([[age,	sex, cp, trestbps, chol , fbs, restecg, thalach, oldpeak]])
        import pandas as pd
        dataset=pd.read_csv('I-deduct-main\website\dataset.csv')
        dataset=dataset.drop(['exang'],axis=1)
        dataset=dataset.drop(['slope'],axis=1)
        dataset=dataset.drop(['ca'],axis=1)
        dataset=dataset.drop(['thal'],axis=1)
        X=dataset.iloc[:,:-1].values
        y=dataset.iloc[:, -1].values
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=1729)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn import metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        prediction = classifier.predict(features)
        print("Prediction: {}".format(prediction))
        return redirect(url_for("views.prediction",predictions= prediction))
    else:
        return render_template("heart.html",user=current_user)
@views.route("/<predictions>")
def prediction(predictions):
    return f"<h1>{predictions}</h1>"

@views.route('/cal_d', methods=['GET', 'POST'])
def cal_d():
    if request.method == "POST":
        Pregnancies=request.form["Pregnancies"]
        Glucose=request.form["Glucose"]
        Insulin=request.form["Insulin"]
        BMI=request.form["BMI"]
        Age=request.form["Age"]
        BloodPressure=request.form["BloodPressure"]
        SkinThickness=request.form["SkinThickness"]
        import numpy as np
        import pandas as pd
        dataset=pd.read_csv('I-deduct-main\website\di.csv')
        dataset=dataset.drop(['DiabetesPedigreeFunction'],axis=1)
        X=dataset.iloc[:,:-1].values
        y=dataset.iloc[:, -1].values
        from sklearn.model_selection import train_test_split
        X_train, X_test , y_train , y_test  = train_test_split(X, y ,test_size = 0.3, random_state  = 41 )
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=1529)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn import metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        import numpy as np
        features = np.array([[Pregnancies,	Glucose,	BloodPressure,	SkinThickness	,Insulin,	BMI,	Age]])
        prediction1 = classifier.predict(features)
        print("Prediction: {}".format(prediction1))
        return redirect(url_for("views.prediction1",predictions1= prediction1))
    else:
        return render_template("diabetes1.html",user=current_user)
@views.route("/<predictions1>")
def prediction1(predictions1):
    return f"<h1>{predictions1}</h1>"




@views.route('/cal_s', methods=['GET', 'POST'])
def cal_s():
    if request.method == "POST":
        gender=request.form["gender"]
        age=request.form["age"]
        hypertension=request.form["hypertension"]
        ever_married=request.form["ever_married"]
        work_type=request.form["work_type"]
        Residence_type=request.form["Residence_type"]
        avg_glucose_level=request.form["avg_glucose_level"]
        bmi=request.form["bmi"]
        smoking_status=request.form["smoking_status"]
        import pandas as pd
        import numpy as np
        df=pd.read_csv('I-deduct-main\website\healthcare-dataset-stroke-data.csv')
        df.dropna(axis=0, inplace=True)
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        label_encoder = LabelEncoder()
        df['gender'] = label_encoder.fit_transform(df['gender'])
        df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
        df['work_type'] = label_encoder.fit_transform(df['work_type'])
        df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
        df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])
        df=df.drop(['id'],axis=1)
        df=df.drop(['heart_disease'],axis=1)
        X=df.iloc[:,:-1].values
        y=df.iloc[:, -1].values
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=1329)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn import metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        import numpy as np
        features = np.array([[gender,age,	hypertension,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]])
        prediction2 = classifier.predict(features)
        print("Prediction: {}".format(prediction2))
        return redirect(url_for("views.prediction2",predictions2= prediction2))
    else:
        return render_template("stroke.html",user=current_user)
@views.route("/<predictions2>")
def prediction2(predictions2):
    return f"<h1>{predictions2}</h1>"


