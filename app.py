import numpy as np 
import pandas as pd
import flask
import pickle
from flask import Flask, render_template, request
import json
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from get_scaler import X_scaler

app = Flask(__name__)

@app.route('/')
def index():
 return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
#  print('before:',to_predict_list)
#  to_predict = np.array(to_predict_list).reshape(1,53)
#  print('after:',to_predict)
#  to_predict = to_predict_list.reshape(1,53)
 print("This is the prediction numpy array:",to_predict_list)
 loaded_model = pickle.load(open('model.pkl','rb'))
 to_predict = np.asarray(to_predict_list, dtype=np.float32)
 print('to_predict:',to_predict)
 result = loaded_model.predict(to_predict)
 print('result:',result)
 return ((result[0] - 25)[0], (result[0] + 25)[0])

@app.route('/predict', methods = ['POST'])
def result():

 if request.method == 'POST':

    to_predict_list = request.form.to_dict()
    print('to_predict_list:', request.form)
    to_predict_list=list(to_predict_list.values())
    print('to_predict_list:', to_predict_list)
    to_predict_list = list(map(float, to_predict_list))
    print('to_predict_list:', to_predict_list)

    data_json = json.load(open('data.json'))
   #  print(data_json)
    
    dict_list = list()

    counter = 1

    for i in data_json:
      if i['listing_id'] == to_predict_list[0]:
         i['month'] = to_predict_list[1]
         dict_list.append(i)
      else:
         i['month'] = counter
         dict_list.append(i)
         if counter < 12:
            counter += 1

   #  print(dict_list)

    df = pd.DataFrame(dict_list)
    
      
    
   #  print('df:',df)
    
    df['bedrooms']=df['bedrooms'].apply(np.sqrt)

    df['bathrooms'] = df['bathrooms'].apply(lambda x: pow(x,1/3))

    df['accommodates']=df['accommodates'].apply(np.log10)

    enc = OneHotEncoder(sparse=False)
    encode_df = pd.DataFrame(enc.fit_transform(df[['city','zipcode','month']]))
    encode_df.columns = enc.get_feature_names(['city','zipcode','month'])
    print('encode_df:',encode_df)
    df = df.merge(encode_df, left_index=True, right_index=True)
    df = df[df['listing_id'] == to_predict_list[0]]
    df.drop(columns=['city','zipcode','month','state','listing_id','price'], inplace=True)
    df.rename(columns={"month_1.0":'month_1',"month_2.0":'month_2',"month_3.0":'month_3',"month_4.0":'month_4', "month_5.0":'month_5',"month_6.0":'month_6',"month_7.0":'month_7',"month_8.0":'month_8',"month_9.0":'month_9',"month_10.0":'month_10',"month_11.0":'month_11',"month_12.0":'month_12'}, inplace=True)
   #  print('df:', df)
   #  print('df shape:', df.shape)

   #  df.drop(columns='city_Seattle ')

    print('Here are the remaining columns:',df.columns)
    val = X_scaler.transform(df)
    
    print(val)
    
    result = ValuePredictor(val)

    prediction = str(result)
 return render_template('predict.html',prediction=prediction)

if __name__ == '__main__':
 app.run(debug=True)
