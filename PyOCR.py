import os 
from PIL import Image
import pytesseract
import Algorithmia

class PyOCR:
    def __init__(self, image, preprocess=True):
        if(preprocess):
            os.system('python3 ImagePreprocess.py ' + image + ' proc_' + image)
            image = 'proc_' + image 
        self.img_name = image
        self.text = ''
    def toText(self, method='tesseract'):
        if method == 'tesseract':
            self.text = pytesseract.image_to_string(Image.open(self.img_name))
        elif method == 'algorithmia_smartOCR':
            client = Algorithmia.client('simwMUzTtxRs30OR13kELwpYPnM1')
            algo = client.algo('ocr/SmartOCR/0.2.6')
            self.text = algo.pipe(self.img_name).result
        return self.text

#s = PyOCR("https://raw.githubusercontent.com/schollz/python-ocr/master/test.jpg", False).toText("algorithmia_smartOCR")
#s = PyOCR("test.jpg", False).toText("tesseract")