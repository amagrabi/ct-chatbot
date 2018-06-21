FROM python:3.6.5
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN ["chmod", "+x", "run_will.py"]
CMD ./run_will.py