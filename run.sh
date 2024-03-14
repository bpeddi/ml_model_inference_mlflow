python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
python mlmodel.py
#mlflow models build-docker -m models:/Diabetes_model/11 -n diabetes_model:latest --enable-mlserve
mlflow models generate-dockerfile -m "models:/Diabetes_model/12"
