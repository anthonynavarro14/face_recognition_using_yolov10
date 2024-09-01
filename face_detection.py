import os
from facenet_pytorch import MTCNN
import torch
from PIL import Image, ImageDraw

class FaceDetector:
    '''
    Exmaple usage:
    from face_detection import FaceDetector

    model = FaceDetector()
    path = ""
    bbox, img_with_boxes = model.detect_faces(image_path=path, draw=True)
    img_with_boxes.show()

    For folder processing use:
    model = FaceDetector()
    folder = PATH_HERE
    result_dict = model.process_folder(folder_path=folder)

    You can access the bounding boxes calling "bbox" on single image mode. 
    If you are trying to process a whole folder of images, this will return a 
    dictionary of filenames as keys and corresponding list of bounding boxes as values for each image.

    Credit: Daniil Valiano
    '''
    
    def __init__(self, device=None, thresholds=[0.95, 0.95, 0.95]):
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device, thresholds=thresholds)
        print(f'Running on device: {self.device}')

    def detect_faces(self, image_path, draw=False):
        image = Image.open(image_path)
        boxes, _ = self.mtcnn.detect(image)

        if draw and boxes is not None:
            frame_draw = image.copy()
            draw = ImageDraw.Draw(frame_draw)
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
            return boxes, frame_draw
        return boxes, image
    
    def process_folder(self, folder_path):
        result = {}
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                boxes = self.detect_faces(image_path)
                if boxes is not None:
                    result[filename] = boxes
        return result
