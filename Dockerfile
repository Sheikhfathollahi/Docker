FROM python:3.8
ENV PYTHONUNBUFFERED=1
RUN mkdir /deepfake
WORKDIR /deepfake
COPY ./ /deepfake/
RUN apt-get install libopencv-dev python3-opencv -y
RUN pip --timeout=1000 install --no-cache-dir --upgrade -r /deepfake/requirements.txt
CMD python main.py

