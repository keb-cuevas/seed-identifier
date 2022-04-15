from numpy import float16
from flask import Flask, render_template
from flask import request
import pandas as pd

path = 'seeds_dataset.txt'

df = pd.read_csv(path, sep= '\t', header= None,
                names=['area','perimeter','compactness','lengthOfKernel',
                       'widthOfKernel','asymmetryCoefficient',
                      'lengthOfKernelGroove','seedType'])

df = df.dropna()
df.info()

X= df.drop('seedType', axis = 1)
y= df['seedType']

from sklearn.model_selection import KFold, cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.30, random_state=42)

#do not show warnings
import warnings
warnings.filterwarnings("ignore")

#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

#create an array of models
models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))

#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits=2)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result)

    xgb_model = xgb.XGBClassifier().fit(X_train, y_train)

import joblib

joblib.dump(xgb_model, "xgb.pkl") #export ML model to pkl file

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])

def text():
  if request.method == 'POST':
    
    xgb = joblib.load("xgb.pkl")
    # Get values through input bars
    area = request.form.get("area")
    perimeter = request.form.get("perimeter")
    compactness = request.form.get("compactness")
    lengthOfKernel = request.form.get("lengthOfKernel")
    widthOfKernel = request.form.get("widthOfKernel")
    asymmetryCoefficient = request.form.get("asymmetryCoefficient")
    lengthOfKernelGroove = request.form.get("lengthOfKernelGroove")

    # Put inputs to dataframe
    X = pd.DataFrame([[area, perimeter, compactness, lengthOfKernel, 
                       widthOfKernel, asymmetryCoefficient, lengthOfKernelGroove]], 
                     columns = ["area", "perimeter", "compactness", "lengthOfKernel", 
                                "widthOfKernel", "asymmetryCoefficient", "lengthOfKernelGroove"])
    X = X.astype(float16)

    # Get prediction
    predict = xgb.predict(X)[0]

    if predict == 1.0:
      prediction = "Kama"
    
    elif predict == 2.0:
      prediction = "Rosa"

    elif predict == 3.0:
      prediction = "Canadian"

    else:
      prediction = "Error"

  else:
    prediction = 'Unknown'

  return render_template('text.html', output = prediction)

if __name__ == "__main__":
	app.run(debug=True)
