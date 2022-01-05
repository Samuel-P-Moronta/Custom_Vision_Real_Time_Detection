# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import cv2

MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'


class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()
        
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


def main():
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    
    # Load model 
    od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)
    
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        predictions = od_model.predict_image(Image.fromarray(frame))

        # Iterate over predictions
        for index in range(len(predictions)):
            # Getting probabilities
            result_proba = (predictions[index]['probability'] * 100)
            # Probability over 60%
            if(result_proba > 60):
                print(" Fruit:", predictions[index]['tagName'],end='')
                print(" Quantity:",predictions[index]['cant_boxes'],end='')
                print(" Probability: ",result_proba)

            elif(result_proba < 50):
                # If we geg < 60 then Unknown object
                print("Unknown")
        cv2.imshow('Object detector', frame)

        keyCode = cv2.waitKey(30) & 0xFF

        if keyCode == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()