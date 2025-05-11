"""This file contains the NN-based distance predictor.

Here, you will design the NN module for distance prediction.
"""

from mp2_distance_predictor.inference_distance import infer_dist
from mp2_distance_predictor.detect import detect_cars

from pathlib import Path
from keras.models import load_model
from keras.models import model_from_json
import numpy as np

# NOTE: Very important that the class name remains the same
class Predictor:
    def __init__(self):
        self.detect_model = None
        self.distance_model = None
        self.previous_distances = []  
        self.max_history = 5  

    def initialize(self):
        self.detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
        self.distance_model = self.load_inference_model()

    def load_inference_model(self):
        MODEL = 'distance_model'
        WEIGHTS = 'distance_model'

        with open(f'/home1/jainak/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{MODEL}.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights(f"/home1/jainak/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{WEIGHTS}.h5")
        print("Loaded model from disk")

        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        return loaded_model

    def predict(self, obs) -> float:
        
        image_path = 'camera_images/vision_input.png'

        car_bounding_box = detect_cars(self.detect_model, image_path)  
        if car_bounding_box is not None:
            print(f"Car detected with bounding box: {car_bounding_box}")

            dist_test = obs.distance_to_lead
            dist = infer_dist(self.distance_model, car_bounding_box, [[dist_test]])
        else:
            print("No car detected. Using fallback distance.")
            dist = self._fallback_distance()

        self._update_distance_history(dist)

        print("Estimated distance:", dist)
        return dist

    def _update_distance_history(self, dist: float):
        self.previous_distances.append(dist)
        if len(self.previous_distances) > self.max_history:
            self.previous_distances.pop(0)

    def _fallback_distance(self) -> float:
        if self.previous_distances:
            return np.mean(self.previous_distances)
        return 30.0  