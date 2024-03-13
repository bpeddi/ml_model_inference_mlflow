## Preferred IDE: Pycharm

### Install MlFlow on local:

1. Install required dependencies on local:

```commandline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


2. Start Mlflow server Loacally:

```
mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db    
```


### Run Mlflow ML Job 

3. Run following python command to train and register  your model into MLFlow

```
python mlmodel.py    
```

4. Last Line on above python job will print MLFlow Model ID , Use this ID to build Docker File 

```
mlflow models build-docker -m models:/Diabetes_model/11 -n diabetes_model:latest --enable-mlserve
```

5. Now you check in the code into codecommit repo to start codepipelin

- git remote add codecommit://mymlflow-test-repo
- git add . 
- git commit -m "new Docker Image to run MLFlow"
- git push 

6. Once Pipeline is succesfull, you will be able to Infer model on end point as follows 

```
curl http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{"dataframe_split": {
"columns": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
"data": [[5,225,72,19,25,17.8,0.587,51]]}}'
```

### (Below commands are some docker helpful commands for local testing)

### Running the container 

1. Important - Make sure you have installed Docker on your PC:
- Linux: Docker
- Windows/Mac: Docker Desktop

2. Start Docker:
- Linux (Home Directory):
  ```
  sudo systemctl start docker
  ```
- Windows: You can start Docker engine from Docker Desktop.

3. Build Docker image from the project directory:

```commandline
sudo docker build -t Image_name:tag .
```



4. witch to Home Directory:

```
cd ~
```
List the built Docker images
```
$ sudo docker images
```

5. Start a container:
```commandline
sudo docker run -p 80:80 Image_ID
```

6. This will display the URL to access the Streamlit app (http://0.0.0.0:80). Note that this URL may not work on Windows. For Windows, go to http://localhost/.

7. In a different terminal window, you can check the running containers with:
```
sudo docker ps
```

8. Stop the container:
 - Use `ctrl + c` or stop it from Docker Desktop.

9. Check all containers:
 ```
 sudo docker ps -a
 ```

10. Delete the container if you are not going to run this again:
 ```
 sudo docker container prune
 ```

### Pushing the docker image to Docker Hub

11. Sign up on Docker Hub.

12. Create a repository on Docker Hub.

13. Log in to Docker Hub from the terminal. You can log in with your password or access token.
```
sudo docker login
```

14. Tag your local Docker image to the Docker Hub repository:
 ```
 sudo docker tag Image_ID username/repo-name:tag
 ```

15. Push the local Docker image to the Docker Hub repository:
 ```
 sudo docker push username/repo-name:tag
 ```

(If you want to delete the image, you can delete the repository in Docker Hub and force delete it locally.)

16. Command to force delete an image (but don't do this yet):
 ```
 $ sudo docker rmi -f IMAGE_ID
 ```
