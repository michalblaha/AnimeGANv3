FROM python:3.7
# Use Bash
SHELL ["/bin/bash", "-c"]


RUN apt-get update; apt-get install git-lfs nano -y ; apt-get install ffmpeg libsm6 libxext6  -y; apt-get install -y python3-opencv


RUN pip install opencv-python

RUN pip install --upgrade pip
RUN pip install Flask


RUN mkdir /app
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt  

CMD ["python", "app.py", "-vvv"]

# python deploy/test_by_onnx.py -i inputs/imgs/ -o output/results -m deploy/AnimeGANv3_Hayao_36.onnx  
