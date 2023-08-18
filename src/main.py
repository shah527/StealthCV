import re
import spacy
import en_core_web_sm
from PIL import Image, ImageDraw
import pytesseract

class Anonymizer:
    EMAIL_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    WEBSITE_REGEX = r'(https?://www\.linkedin\.com/in/[A-Za-z0-9._-]+|https?://\S+|www\.\S+\.[com|net|edu]{2,3}|[A-Za-z0-9.-]+\.[com|net|edu]{3})'
    PHONE_REGEX = r'\b\d{10}\b|\b\d{3}-\d{3}-\d{4}\b|\b\d{3}\s\d{3}\s\d{4}\b|\(\d{3}\)\s\d{3}-\d{4}\b'
    EDU_REGEX = r'\b(?:\w+\s+){0,4}(?:University|College)\b'

    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def find_info_regex(self, text):
        personal_info = set()
        personal_info.update(re.findall(self.EMAIL_REGEX, text))
        personal_info.update(re.findall(self.WEBSITE_REGEX, text))
        personal_info.update(re.findall(self.PHONE_REGEX, text))
        personal_info.update(re.findall(self.EDU_REGEX, text))
        
        return personal_info
    def find_info_ner(self, text, personal_info):
        nlp = en_core_web_sm.load()
        text = nlp(text)

        for X in text.ents:
            print("Text:", X.text, "\tLabel:", X.label_)
            if X.label_ in ["ORG", "PERSON"]:
                personal_info.add(X.text)

    def anonymize_resume(self, image_path):
        # Load the image
        image = Image.open(image_path)
        
        # ocr
        text = pytesseract.image_to_string(image)
        
        # find info
        personal_info = self.find_info_regex(text)
        self.find_info_ner(text, personal_info)

        # Store coordinates
        coordinates_to_box = []
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        with open('personal_info.txt', 'w') as file:
            for word, x, y, w, h in zip(data['text'], data['left'], data['top'], data['width'], data['height']):
                if any(info in word for info in personal_info):
                    coordinates_to_box.append((x, y, x + w, y + h))
                    file.write(f"Blacking out: {word}\n")

        print("Personal information written to personal_info.txt")

        # hide info
        draw = ImageDraw.Draw(image)
        for coordinates in coordinates_to_box:
            draw.rectangle(coordinates, fill="black")

        # Save or display the anonymized image
        anonymized_image_path = 'anonymized_' + image_path
        image.save(anonymized_image_path)

        return anonymized_image_path

# Example usage
anonymizer = Anonymizer()
image_path = 'resume.bmp'
anonymized_image_path = anonymizer.anonymize_resume(image_path)
print(f"Anonymized resume saved at {anonymized_image_path}")


# need to fine tune
# need to use gpt api for greater accuracy
# need to finish webapp