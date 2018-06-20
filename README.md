## Setup

* Get the data:

`mkdir data`

`gsutil -m rsync -r gs://ctp-playground-ml-datasets/hipchat/data data`

* Install redis (used by Will):

`brew install redis`

* Create environment:

`conda create -y -n ct-chatbot python==3.6`

`conda activate ct-chatbot`

`pip install --user -r requirements.txt`

* Set up credentials in secrets.py

## Running the chatbot

* Start redis server:

`redis-server`

* Run bot:

`python run_will.py`

