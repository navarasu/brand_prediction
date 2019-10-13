from src.model import predict
from dotenv import load_dotenv

from time import time
load_dotenv()

start_time = time()
event = {"cus_type":"Type 1","cus_point":2517,"purchase_events":{"Product D":"Brand 59","Product N":"Brand 2"}}
# result=predict_vendor.handler(event, {})
result=predict.handler(event, {})
end_time = time()
print(end_time - start_time)
print(result)