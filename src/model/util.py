import numpy as np
import boto3
import os
import joblib

def getModel():
    bucket=os.getenv("BUCKET_NAME")
    S3 = boto3.client('s3')
    model_file=os.getenv("MODEL_FILE_NAME")
    local_file='/tmp/'+model_file
    response = S3.download_file(bucket,model_file,local_file)
    return joblib.load(local_file)

def upload_model_to_s3(value):
    s3 = boto3.client('s3')
    bucket=os.getenv("BUCKET_NAME") 
    file_name=os.getenv("MODEL_FILE_NAME")
    joblib.dump(value,file_name)
    s3.upload_file(file_name, bucket,file_name)
    print("Uploaed sucessfully modal "+file_name+" to "+bucket)

def encode_events(COLUMNS_NAMES,data):
    value=np.zeros(len(COLUMNS_NAMES),dtype=int)
    for service_id, vendor_id in data:
        key=str(service_id)+"_"+str(vendor_id)
        if key in COLUMNS_NAMES:
            value[COLUMNS_NAMES[key]]=1
    return value