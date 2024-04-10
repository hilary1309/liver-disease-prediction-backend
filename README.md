# FastApi

FastApi endpoint models

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the necessary dependencies.

create a mongodb collection and an image folder.. add them to the main.py file

## Usage

```bash
pip install -r requirements.txt
```

## Run command

```bash
uvicorn main:app --reload  
```

## Endpoints

```bash
# get all endpoints 
http://127.0.0.1:8000/docs

# get all predictions
http://127.0.0.1:8000/get_predictions

# post detection or prediction
http://127.0.0.1:8000/detection

# delete prediction or detection
http://127.0.0.1:8000/delete_predictions/id/

```
