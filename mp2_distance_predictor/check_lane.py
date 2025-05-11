import os
import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from mp2_distance_predictor.detect import detect_cars

class LeadCarLanePredictor:
    def __init__(self):
        self.bb_detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
        self.carlane_model = None

    def create_dataset(self):
        data = []

        lane_in_images = 'lane_data/ado_in_lane/'
        lane_out_images = 'lane_data/ado_not_in_lane/'

        for img_name in os.listdir(lane_in_images):
            img_path = os.path.join(lane_in_images, img_name)
            img = cv2.imread(img_path)
            bbox = self.get_bounding_boxes(img_path)  
            
            for box in bbox:  
                xmin, ymin, xmax, ymax = box
                data.append([img_path, 1, xmin, ymin, xmax, ymax])

        for img_name in os.listdir(lane_out_images):
            img_path = os.path.join(lane_out_images, img_name)
            img = cv2.imread(img_path)
            bbox = self.get_bounding_boxes(img_path)  
            
            for box in bbox:  
                xmin, ymin, xmax, ymax = box
                data.append([img_path, 0, xmin, ymin, xmax, ymax])

        df = pd.DataFrame(data, columns=['image_path', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        train_df.to_csv("data/lane_train.csv", index=False)
        test_df.to_csv("data/lane_test.csv", index=False)

        print("Dataset created and saved to CSV files.")

    def get_bounding_boxes(self, img):
        bounding_boxes = detect_cars(self.bb_detect_model, img)
        
        if bounding_boxes is None:
            return []  
        else:
            return bounding_boxes
    
    def check_lane_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=4, activation='relu'))  
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        train_data = pd.read_csv('data/lane_train.csv')
        X_train = train_data[['xmin', 'ymin', 'xmax', 'ymax']].values
        y_train = train_data['label'].values
        
        model.fit(X_train, y_train, epochs=20, batch_size=32)

        model_json = model.to_json()
        os.makedirs('/home1/jainak/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights', exist_ok=True)
        with open("/home1/jainak/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/car_lane_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("/home1/jainak/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/car_lane_model.h5")
        print("Saved car_lane model to disk")
        self.carlane_model = model

    def check_lane_model_test(self):
        with open("/home1/jainak/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/car_lane_model.json", "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)

        model.load_weights("/home1/jainak/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/car_lane_model.h5")

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        test_data = pd.read_csv('data/lane_test.csv')
        X_test = test_data[['xmin', 'ymin', 'xmax', 'ymax']].values
        y_test = test_data['label'].values
        
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        
        return accuracy > 0.95  

    def predict(self, img):
        boxes = self.get_bounding_boxes(img)
        
        if not boxes:
            return False
        
        model = load_model(f"/home1/jainak/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/car_lane_model.h5")
        
        predictions = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            
            prediction = model.predict(np.array([[xmin, ymin, xmax, ymax]]))
            predictions.append(prediction)
        
        return any(pred > 0.5 for pred in predictions)


def main():
    leadcarlanepredictor = LeadCarLanePredictor()
    
    # Create the dataset from the images in lane_data
    leadcarlanepredictor.create_dataset()

    # Train the model to predict if the car is in the same lane or not
    leadcarlanepredictor.check_lane_model()

    # Test the model using data/lane_test.csv
    if leadcarlanepredictor.check_lane_model_test():
        print("The model is accurate enough to be used in the simulation.")
    else:
        print("The model needs improvement.")

if __name__ == '__main__':
    main()
