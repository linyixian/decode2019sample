import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import cv2
#import json

MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'

class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow
    """
    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
            
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:,:,(2,1,0)] # RGB -> BGR

        with tf.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis,...]})
            return outputs[0]


def main(image_filename):
    # Load a TensorFlow model
    graph_def = tf.GraphDef()
    #with tf.gfile.FastGFile(MODEL_FILENAME, 'rb') as f:
    with tf.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFObjectDetection(graph_def, labels)

    image = Image.open(image_filename)
    predictions = od_model.predict_image(image)

    ##
    pilImg=np.asarray(image)
    cvImage=cv2.cvtColor(pilImg,cv2.COLOR_RGB2BGR)
    screen=cvImage
    ih,iw,ic=screen.shape
    
    for p in predictions:
        if p["probability"]>0.1:
            
            probab=int(p["probability"]*100)
            w=int(p["boundingBox"]["width"]*iw)
            h=int(p["boundingBox"]["height"]*ih)
            l=int(p["boundingBox"]["left"]*iw)
            t=int(p["boundingBox"]["top"]*ih)

            cv2.rectangle(cvImage,(l,t),(l+w,t+h),(0,0,255),2)
            cv2.putText(cvImage,str(probab)+"%",(l+5,t-5),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)
            cv2.putText(cvImage,p["tagName"],(l+60,t-5),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)
            

    cv2.imshow('Dog',cvImage)

    cv2.waitKey()

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} image_filename'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
