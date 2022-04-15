from numpy import float16
from flask import Flask, render_template
from flask import request

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

app.run()