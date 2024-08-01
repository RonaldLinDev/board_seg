## TO DO CONVERT TO PREDICTOR PLEASE

from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import os
from PIL import Image, UnidentifiedImageError, ImageEnhance
# from skimage import exposure
# import numpy as np

class predictor:
    def __init__(self) -> None:
        self.source = "test_images/"
        self.model = FastSAM("models/FastSAM-s.pt")  # or FastSAM-x.pt
        # self.PROMPT = "segment the entire PCB board from the dark background from edge to edge. Do not leave any board out of the box"
        self.PROMPT = "segment board from background, corner to corner, ignoring visual artifacts"
        self.result_dir = "results/testhalflight/"
        self.filetype = '.png'

    def predict(self, src: str, *args) -> None:
        bw_image = Image.open(src) #bit map format
        exposed_image = ImageEnhance.Brightness(bw_image).enhance(0.8)
        everything_results = self.model(bw_image, device="mps", retina_masks=True, imgsz=640, conf=0.4, iou=0.9) #best 480 rn 
        prompt_process = FastSAMPrompt(src, everything_results, device="mps")
        results = prompt_process.text_prompt(text=self.PROMPT)
        results[0].path = src
        prompt_process.plot(annotations=results, output = self.result_dir)
        print(results[0].boxes)
        return "bounding box"

    def crop_dir(self, dir: str):
        for image in os.listdir(dir):
            if image.endswith(self.filetype):
                self.predict(os.path.join(dir, image))
            
    def save(self, results):
        prev = os.getcwd()
        os.chdir(self.result_dir)
        for result in results:
            result.save()
        os.chdir(prev)
    
        
            
fastSAM = predictor()
fastSAM.crop_dir("test_images/") 