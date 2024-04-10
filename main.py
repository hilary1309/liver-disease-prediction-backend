import io
#from msilib.schema import File
import os
import traceback
import uuid
from typing import Optional
from typing import List, Tuple
import cv2
import os
from numpy import ndarray


import numpy as np
from bson import ObjectId

from fastapi import FastAPI, File, UploadFile, Request, Path, HTTPException
from typing import Annotated

from fastapi import FastAPI, Form

from PIL import Image
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime

from starlette.middleware.cors import CORSMiddleware
from starlette.applications import Starlette
from starlette.middleware import Middleware

from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles


import uvicorn

import cloudinary
import cloudinary.uploader
from fastapi.responses import JSONResponse
from cloudinary.uploader import upload




app = FastAPI()

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best.onnx')


class Detection:
    def __init__(self,
                 model_path: str,
                 classes: List[str]
                 ):
        self.model_path = model_path
        self.classes = classes
        self.model = self.__load_model()

    def __load_model(self) -> cv2.dnn_Net:
        net = cv2.dnn.readNet(self.model_path)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def __extract_ouput(self,
                        preds: ndarray,
                        image_shape: Tuple[int, int],
                        input_shape: Tuple[int, int],
                        score: float = 0.1,
                        nms: float = 0.0,
                        confidence: float = 0.0
                        ) -> dict[str, list]:
        class_ids, confs, boxes = list(), list(), list()

        image_height, image_width = image_shape
        input_height, input_width = input_shape
        x_factor = image_width / input_width
        y_factor = image_height / input_height

        rows = preds[0].shape[0]
        for i in range(rows):
            row = preds[0][i]
            conf = row[4]

            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            # print(classes_score[class_id])
            if (classes_score[class_id] > score):
                confs.append(conf)
                label = self.classes[int(class_id)]
                class_ids.append(label)

                # extract boxes
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = list(), list(), list()
        indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms)
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i] * 100)
            r_boxes.append(boxes[i].tolist())

        return {
            'boxes': r_boxes,
            'confidences': r_confs,
            'classes': r_class_ids
        }

    def __call__(self,
                 image: ndarray,
                 width: int = 640,
                 height: int = 640,
                 score: float = 0.1,
                 nms: float = 0.0,
                 confidence: float = 0.0
                 ) -> dict[list, list, list]:

        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (width, height),
            swapRB=True, crop=False
        )
        self.model.setInput(blob)
        preds = self.model.forward()
        preds = preds.transpose((0, 2, 1))

        # extract output
        results = self.__extract_ouput(
            preds=preds,
            image_shape=image.shape[:2],
            input_shape=(height, width),
            score=score,
            nms=nms,
            confidence=confidence
        )
        return results


# Define classes
CLASESS_YOLO = ['UnHealthy', 'Healthy']

detection = Detection(
    model_path=model_path,
    classes=CLASESS_YOLO
)

# Local db
client = MongoClient('mongodb://localhost:27017/')
db = client['testing_db']
collection = db['testing_collecttion']


# Cloud db with Atlas
#client = MongoClient('mongodb+srv://aghasilihilary:Hilary139@liverdisease.e0rfdfc.mongodb.net/')
#db = client['Fatty_liver_db']
#collection = db['LiverDisease']


# upload_image = 'C:\\Users\\Admin\\PycharmProjects\\Docker application\\Docker\\images'
upload_image = '/Users/hilaryaghasili/Desktop/Final implement backend/images'
app.mount("/images", StaticFiles(directory=upload_image), name="images")


# Set up Cloudinary configuration (replace 'cloud_name', 'api_key', 'api_secret' with your Cloudinary credentials)
cloudinary.config(
cloud_name="dhqgwswuo",
api_key= "784332335378179",
api_secret= "uFeza5HQR1x013jLYQUE-zaFIs4",
)


# mongo_host = "mongo"  # This corresponds to the service name in your Docker Compose
# mongo_port = 27017
# db_name = "Maize_disease_Classification"
# collection_name = "predictions"
#
# client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
# db = client[db_name]
# collection = db[collection_name]


origins = [
    'http://localhost:4200',
    #'https://detection-system-frontend-9xapjuzvm-mchim91.vercel.app',
    'http://127.0.0.1:8000',
    'http://127.0.0.1:36132'
    'http://0.0.0.0:8000'
]
app.add_middleware(
    CORSMiddleware,
    #allow_origins=origins,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)




@app.get("/get_predictions")
def get_predictions():
    predictions = list(collection.find({}))
    return {"predictions": predictions}



def upload_to_cloudinary(file_contents: bytes) -> str:
    try:
        # Upload image to Cloudinary
        upload_result = cloudinary.uploader.upload(file_contents)
        # Return secure URL
        return upload_result['secure_url']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading the image: {str(e)}")


@app.post("/upload")
async def upload_image_file(file: UploadFile = File(...)):
    try:
        # Read the file contents
        contents = await file.read()
        # Upload file contents to Cloudinary using synchronous function call
        upload_result = upload_to_cloudinary(contents)
        # Return the upload result as JSON response
        return JSONResponse(content={"url": upload_result['secure_url']})

    except Exception as e:
        # Handle any exceptions
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")






@app.post('/detection')
async def post_detection(file: UploadFile, request: Request):
    try:
        # Generate random id
        prediction_id = str(ObjectId())

        # Read image bytes from UploadFile
        image_bytes = await file.read()

        # Process the image as you were doing
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)
        image = image[:, :, ::-1].copy()
        results = detection(image)

        # Find the index with the highest confidence
        max_confidence_index = results["confidences"].index(max(results["confidences"]))

        # Get the corresponding class with the same index
        highest_confidence_class = results["classes"][max_confidence_index]

        # Get the highest confidence score
        highest_confidence_score = max(results["confidences"])

        # Convert the highest confidence score into the float range of 0 to 1
        highest_confidence_score_normalized = round(highest_confidence_score, 2)  # Round to 2 decimal places

        # Provides a list of preventions for each model being identified
        prevention_messages = {
            #'Blight': "Blight detected. Prune infected leaves and use fungicide to control spread.",
            #'Rust': "Rust identified. Use fungicide and remove infected leaves.",
            'UnHealthy': "Fatty Liver Disease spot found. Take Medication, Seek medical care.",
            'Healthy': "Liver looks healthy. Maintain current care."
        }

        suggestion = prevention_messages.get(highest_confidence_class, "")

        # Upload the image to Cloudinary and retrieve the image URL
        image_url = upload_to_cloudinary(image_bytes)

        # Construct prediction data
        prediction_data = {
            "_id": prediction_id,
            "timestamp": datetime.now(),
            "prediction": results,
            "highest_confidence_class": highest_confidence_class,
            "highest_confidence_score": highest_confidence_score_normalized,
            "suggestions": suggestion,
            "image_url": image_url  # Store the Cloudinary image URL
        }

        # Store prediction data in the database
        collection.insert_one(prediction_data)

        # Remove 'boxes' key from results before returning
        del results['boxes']

        # Return response
        return {
            "_id": prediction_id,
            "highest_confidence_class": highest_confidence_class,
            "highest_confidence_score": highest_confidence_score_normalized,
            "suggestions": suggestion,
            "Image URL": image_url  # Return the Cloudinary image URL
        }

    except Exception as e:
        return {"error": f"An error occurred while processing the image: {str(e)}"}



@app.delete('/delete_prediction/{prediction_id}')
async def delete_prediction(prediction_id: str):
    try:
        # Log the received prediction_id
        print(f"Received DELETE request for prediction_id: {prediction_id}")

        prediction = collection.find_one_and_delete({"_id": prediction_id})
        print(f"Retrieved Prediction: {prediction}")

        # object_id = ObjectId(prediction_id)
        # print(f"Parsed ObjectId: {object_id}")
        #
        # # Find the prediction by its id
        # prediction = collection.find_one_and_delete({"_id": object_id})
        # print(f"Retrieved Prediction: {prediction}")

        if prediction is not None:
            # Delete the associated image file
            image_url = prediction.get("image_url", "")
            if image_url:
                image_path = os.path.join(upload_image, image_url.split("/")[-1])
                print(f"Image Path: {image_path}")  # Debugging: Log the image path
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Image file deleted: {image_path}")  # Debugging: Log image deletion

            return {"success": True, "message": "Prediction deleted successfully"}
        else:
            return {"success": False, "message": "Prediction not found"}
    except Exception as e:
        return {"error": f"An error occurred while deleting the prediction: {str(e)}"}


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
