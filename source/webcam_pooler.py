import cv2
import threading
from PIL import Image

class WebcamPooler(threading.Thread):
    def __init__(self, on_output):
        threading.Thread.__init__(self)
        self.stopped = False
        self.on_output = on_output

    def stop(self):
        self.stopped = True
    
    def run(self):
        while not self.stopped:
            cap = cv2.VideoCapture(0)
            _, cv2_im = cap.read()
            cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            try:
                self.on_output(pil_im)
            except:
                pass



