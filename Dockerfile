# base image
FROM ubuntu
# update the respository and install python3.8 and pip
RUN apt update && apt install -y python3.8 python3-pip
# set up the working directory
WORKDIR /app
# copy necessary sources to the working directory
COPY requirements.txt /app
COPY ResNet50.py /app
COPY model-weights /app/model-weights
COPY model.h5 /app
COPY app.py /app
# install all the required modules
RUN pip3 install -r requirements.txt
# set up the port
EXPOSE 5000
# run the app
CMD ["python3", "app.py"]
