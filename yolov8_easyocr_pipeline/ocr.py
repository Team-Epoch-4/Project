import easyocr
from typing import List, Tuple
import numpy as np

class PillOCR:
    def __init__(self, languages=['ko', 'en']):
        self.reader = easyocr.Reader(languages, gpu=False)

    def read_text(self, image: np.ndarray) -> List[dict]:
        results = self.reader.readtext(image)
        ocr_results = []
        for (bbox, text, conf) in results:
            ocr_results.append({
                'bbox': bbox,
                'text': text,
                'confidence': conf
            })
        return ocr_results
