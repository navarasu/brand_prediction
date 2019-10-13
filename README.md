# Prediction Service

Machine Learning model training and prediction api for brand prediction.

## Train Model

### Set Up Project

```sh
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train and Deploy models to S3

```sh
python src/model/train.py
```

## Deploy Predicton Api

### Set Up Serverless

```sh
npm install -g serverless
npm i -D serverless-dotenv-plugin
sls plugin install -n serverless-python-requirements
```

### Deploy prediction Api to AWS lambda

```sh
sls deploy
```