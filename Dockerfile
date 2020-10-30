# base image
FROM ubuntu
# update the respository and install python3.8 and pip
RUN apt update && apt install -y python3.8 python3-pip
# set up the working directory
WORKDIR /app
# copy evertything to the working directory
COPY . /app
# install all the required modules
RUN pip3 install -r requirements.txt
# run the app
CMD ["python3", "app.py"]
