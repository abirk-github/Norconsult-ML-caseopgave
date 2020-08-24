from flask import Flask, request
import pandas as pd
import pickle
app = Flask(__name__)
@app.route("/api/predict", methods=["POST"])
def hello():
   csv_file = request.files['data']
   filename = csv_file.filename
   csv_file.save(filename)
   df = pd.read_csv(filename)
   # Transform the saledate datetime feature to integer timestamp
   df["saledate"] = pd.to_datetime(df["saledate"]).astype(int)/10**9

   # Transform the categorical variables to numerical representation
   for col_name in df.columns:
        if(df[col_name].dtype == 'object'):
            df[col_name]= df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes
   dataset = df.loc[:, df.columns != "SalesId"]
   loaded_model = pickle.load(open("model.sav", 'rb'))
   result = loaded_model.predict(dataset)
   print(result)

   
   print(df.head()) 
   # predict prices with your model
   # find a way to return the results. Got different ways to do it. For instance returning 
   # salesId - salesPrice
   # for all the gravemaskienr
   return "Hello World"
if __name__ == "_main_":
    app.run()

    
# curl -i -X POST -F data=@TrainAndValid.csv http://localhost:5000/api/predict