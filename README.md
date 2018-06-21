## Setup

* Get the data:

`mkdir data`

`gsutil -m rsync -r gs://ctp-playground-ml-datasets/hipchat/data data`

* Get the models:

`mkdir models`

`gsutil -m rsync -r gs://ctp-playground-ml-datasets/hipchat/models models`

* Install redis (used by Will):

`brew install redis`

* Create environment (e.g. with [Anaconda](https://anaconda.org/anaconda/python)):

`conda create -n ct-chatbot python==3.6`

`conda activate ct-chatbot`

`pip install -r requirements.txt`

* Set up credentials in secrets.py

## Running the chatbot

* Start redis server:

`redis-server`

or

`docker-compose up -d`

* Run bot:

`python run_will.py`

