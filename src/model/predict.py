import json
import numpy as np
from src.model import util

def handler(event, context):
    dump=util.getModel()
    # Preprocess
    data=pre_process(event,dump)

    # Predict the value and its probability
    vendor,probability=predict(dump["model"],data)
    
    return json.dumps({ "vendor": vendor,
                        "confidence": probability})

def pre_process(data,dump):
    # Encode event matrix
    value=util.encode_events(dump["event_matrix"],data["purchase_events"].items())

    #Preprocess categorial and continuos data
    value=np.append(value,one_hot_encoder(dump["customer_type"],data["cus_type"]))
    return [np.append(value,dump["cus_point_scaler"].transform([[data["cus_point"]]]))]

def one_hot_encoder(categories,data):
    oh_array=np.zeros(len(categories),dtype=int)
    try:
        oh_array[categories.index(data)]=1
    except:
        pass
    return oh_array

def predict(model,data):
    brand=model.predict(data)[0]
    probability=sorted(model.predict_proba(data)[0], reverse=True)[0]
    probability=round(float(probability)*100,2)
    return brand,probability