## Setup

* Get the data:

`gsutil -m rsync -r gs://ctp-playground-ml-datasets/hipchat/data data`

* Create environment:

`conda create -y -n ct-chatbot python==3.6`

`conda activate ct-chatbot`

`pip install --user -r requirements.txt`

* Set up credentials in secrets.py

* ...
