import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import boto3
from dotenv import load_dotenv
import joblib
import os
import util
load_dotenv()

PRODUCTS,EVENT_MATRIX_IDX,NO_OF_EVENTS,INPUT_PRODUCTS=None,None,None,None
def main():
    print("1.Loading data")
    data = pd.read_csv("src/model/dependent_data.csv")
    global PRODUCTS
    PRODUCTS=data['product'].unique()
    processed_data=data.groupby(['cus_id']).apply(transform_column).reset_index()

    print("2.Clearing dirty data")
    processed_data.dropna(subset=['Product A'],inplace=True)
    processed_data.dropna(how='all',subset = PRODUCTS[PRODUCTS !='Product A'],inplace=True)

    print("3.Spliting train data")
    y=processed_data["Product A"]
    x=processed_data.drop(["Product A"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=7)


    print("4.Preparing training data")
    y=processed_data["Product A"]
    x=processed_data.drop(["Product A"], axis=1)

    print("5.Preparing event matrix")
    input_products = sorted(PRODUCTS[PRODUCTS !='Product A'])
    event_matrix=generate_event_matrix_index(input_products,processed_data)
    x_train_temp=x_train.apply(lambda x : pd.Series(util.encode_events(event_matrix,x[input_products].dropna().items())) ,axis=1)
    x_train=x_train_temp.join(x_train[['cus_type','cus_point']])

    print("6.Preprocessing categorial data")
    cus_type_category=processed_data['cus_type'].unique();
    cus_type = CategoricalDtype(categories=cus_type_category, ordered=True)
    x_train['cus_type']=x_train['cus_type'].astype(cus_type)
    x_train=pd.get_dummies(x_train,prefix='cus')

    print("7.Preprocessing continuous data")
    cus_point_scaler = MinMaxScaler()
    x_train["cus_point"]=cus_point_scaler.fit_transform(x_train[["cus_point"]])
    
    print("9.Training the model")
    lr = LogisticRegression(max_iter=1000,solver='lbfgs',multi_class='auto')
    model_lr = lr.fit(x_train,y_train)

    print("10.Dumping the moodel to s3 bucket")
    dump_data={ "model":model_lr, 
                "event_matrix": event_matrix,
                "customer_type": cus_type_category,
                "cus_point_scaler":cus_point_scaler }

    util.upload_model_to_s3(dump_data)

def transform_column(x):
    columns = {}
    columns['cus_type']=x['cus_type'].iloc[0]
    columns['cus_point']=x['cus_point'].iloc[0]
    present = dict((row['product'],row['brand']) for i,row in x.iterrows())
    for row in PRODUCTS:
        columns[row]=present[row] if row in present else None
    return pd.Series(columns)

def generate_event_matrix_index(input_products,processed_data):
    event_matrix_index={}
    i=0
    for product in input_products:
        prodct_brands=sorted(processed_data[product].dropna().unique())
        for brand in prodct_brands:
            event_matrix_index[product+"_"+brand]=i
            i+=1
    return event_matrix_index


main()