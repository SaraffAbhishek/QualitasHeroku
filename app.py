import requests
import os
import json
from groq import Groq
from dotenv import load_dotenv
import time
import ast
from queue import Queue
import tempfile
import os
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import logging
from PIL import Image
import cv2
# Load environment variables
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import ast
from werkzeug.utils import secure_filename
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Function to extract text from the uploaded PDF
class LightweightOCRProcessor:
    def __init__(self, lang='eng'):
        self.logger = logging.getLogger(__name__)
        self.lang = lang
        self.tesseract_configs = {
            'standard': '--psm 6 --oem 3',
            'sparse': '--psm 11 --oem 3',
            'table': '--psm 4 --oem 3'
        }
        self.result_queue = Queue()

    def preprocess_image(self, image, doc_type='standard'):
        """
        Lightweight image preprocessing optimized for different document types
        Args:
            image: PIL Image
            doc_type: Type of document (standard, scan, table)
        Returns:
            PIL Image: Preprocessed image
        """
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if doc_type == 'scan':
            # Optimize for scanned documents
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Adaptive thresholding for better handling of shadows
            thresh = cv2.adaptiveThreshold(
                denoised, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif doc_type == 'table':
            # Optimize for tables
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Remove noise while preserving lines
            kernel = np.ones((2,2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
        else:  # standard
            # Basic preprocessing for typical documents
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        return Image.fromarray(thresh)

    def detect_document_type(self, image):
        """
        Detect the type of document from the image
        Args:
            image: PIL Image
        Returns:
            str: Document type (standard, scan, or table)
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        if cv2.countNonZero(horizontal_lines) > 100 and cv2.countNonZero(vertical_lines) > 100:
            return 'table'
        
        # Check if it's a scan by analyzing image quality
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 100:  # Low variance indicates potential scan
            return 'scan'
            
        return 'standard'

    def ocr_with_fallback(self, image, doc_type='standard'):
        """
        Perform OCR with fallback options
        Args:
            image: PIL Image
            doc_type: Type of document
        Returns:
            tuple: (text, confidence)
        """
        # Try primary OCR
        try:
            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.tesseract_configs[doc_type]
            )
            
            # Get confidence scores
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self.tesseract_configs[doc_type],
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [float(x) for x in data['conf'] if x != '-1']
            confidence = np.mean(confidences) if confidences else 0
            
            if confidence < 50:  # Low confidence, try different preprocessing
                enhanced_image = self.enhance_low_confidence_image(image)
                new_text = pytesseract.image_to_string(
                    enhanced_image,
                    lang=self.lang,
                    config=self.tesseract_configs[doc_type]
                )
                
                # Use the better result
                if len(new_text.strip()) > len(text.strip()):
                    text = new_text
                
            return text.strip(), confidence
            
        except Exception as e:
            self.logger.error(f"OCR error: {str(e)}")
            return "", 0

    def enhance_low_confidence_image(self, image):
        """
        Enhance image when OCR confidence is low
        Args:
            image: PIL Image
        Returns:
            PIL Image: Enhanced image
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)
        
        # Additional denoising
        enhanced = cv2.fastNlMeansDenoising(enhanced)
        
        return Image.fromarray(enhanced)

    def process_page(self, image):
        """
        Process a single page
        Args:
            image: PIL Image
        Returns:
            dict: Extracted text and metadata
        """
        # Detect document type
        doc_type = self.detect_document_type(image)
        
        # Preprocess image
        preprocessed = self.preprocess_image(image, doc_type)
        
        # Perform OCR
        text, confidence = self.ocr_with_fallback(preprocessed, doc_type)
        
        return {
            'text': text,
            'confidence': confidence,
            'doc_type': doc_type
        }

    def process_pdf(self, pdf_path, dpi=200):
        """
        Process PDF document
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for PDF conversion (lower for faster processing)
        Returns:
            list: List of dictionaries containing extracted text and metadata
        """
        try:
            # Convert PDF to images with moderate DPI for balance of speed and quality
            images = convert_from_path(pdf_path, dpi=dpi)
            
            results = []
            for i, image in enumerate(images):
                self.logger.info(f"Processing page {i+1}/{len(images)}")
                result = self.process_page(image)
                result['page_number'] = i + 1
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise

    def clean_text(self, text):
        """
        Clean extracted text
        Args:
            text: Raw extracted text
        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I').replace('1', 'I', 1)
        
        # Fix common OCR errors
        text = text.replace('0', 'O', 1)
        
        return text

# Function to classify the extracted text using the LLM
def classification_LLM(text):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful classification assistant. You understand engineering concepts. You will be given some text which mostly describes a problem. You have to classify the problem according to a list of choices. More than one choice can also be applicable. Return as a array of applicable CHOICES only. Only return the choices that you are very sure about\n\n#CHOICES\n\n2D Measurement: Diameter, thickness, etc.\n\nAnomaly Detection: Scratches, dents, corrosion\n\nPrint Defect: Smudging, misalignment\n\nCounting: Individual components, features\n\n3D Measurement: Volume, surface area\n\nPresence/Absence: Missing components, color deviations\n\nOCR: Optical Character Recognition, Font types and sizes to be recognized, Reading speed and accuracy requirements\n\nCode Reading: Types of codes to read (QR, Barcode)\n\nMismatch Detection: Specific features to compare for mismatches, Component shapes, color mismatches\n\nClassification: Categories of classes to be identified, Features defining each class\n\nAssembly Verification: Checklist of components or features to verify, Sequence of assembly to be followed\n\nColor Verification: Color standards or samples to match\n"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.21,
        max_tokens=2048,
        top_p=1,
        stream=True,
        stop=None,
    )

    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""
    return answer

def obsjsoncreate(json_template,text,ogtext):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You will be given a text snippet. You will also be given a JSON where some of the fields match with the bullet points in the text. I want you return a JSON where only the fields and subproperties mentioned in the text are present.ENSURE THE JSON IS VALID AND PROPERLY FORMATTED. DONT OUTPUT ANYTHING OTHER THAN THE JSON\n"
            },
            {
                "role": "user",
                "content": "JSON:"+str(json_template)+"\nText:"+text
            }
        ],
        temperature=0.21,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    cutjson=""
    for chunk in completion:
        cutjson += chunk.choices[0].delta.content or ""
    
    completion2 = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. Your task is to populate a JSON structure based on information provided in a PDF document and subsequent user responses. Follow these guidelines carefully:\n\n1. JSON Structure:\n You will be given a JSON template with properties and their descriptions.\nYour goal is to fill the \"User Answer\" subproperty for each field based on the information provided.\n\n2. Information Sources:\nPrimary source: Details extracted from the PDF document.\n\n3. Filling the \"User Answer\":\nIf a clear, unambiguous answer is found, fill it in the \"User Answer\" field.\nIf no information is available or the answer is unclear, mark the field as 'TBD.\n\n Mark a field as 'CONFLICT' in the following scenarios:\na: Multiple occurrences of the same field in the PDF with different answers.\nb: Multiple, inconsistent answers provided by the user for the same field.\n\n5. Accuracy and Relevance:\nEnsure that the answers are relevant to the field descriptions.\nDo not infer or assume information until explicitly stated.\n\n6. Output Format:\nProvide only the valid, properly formatted JSON as output.\nGive the JSON output with the filled fields only.\nEnsure proper nesting, quotation marks, and commas in the JSON structure.\n\n7. Also:\nPay attention to units of measurement and formats specified in the field descriptions.\nIf a field requires a specific format (e.g., date, number range), ensure the answer adheres to it.\n\nRemember, your role is to accurately capture and classify the information provided, highlighting any inconsistencies or conflicts. Do not output anything other than the requested JSON structure. Your goal is to provide a clear, accurate, and properly formatted JSON output that reflects the information given, including any ambiguities or conflicts encountered.Give the JSON output with the filled fields only. ENSURE THE JSON IS VALID AND PROPERLY FORMATTED. DO NOT OUTPUT ANYTHING OTHER THAN THE JSON."
            },
            {
                "role": "user",
                "content": "JSON: "+cutjson+"\n Text: "+ogtext
            }
        ],
        temperature=0.23,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    answer = ""
    for chunk in completion2:
        answer += chunk.choices[0].delta.content or ""
    return answer

def bizobjjsoncreate(json_template,text):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion2 = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful classification assistant. You understand engineering concepts. You will be given a JSON where there are properties and their descriptions. You need to fill up the JSON subproperty \"User Answer\" from the details given in the text. If no information is available or the answer is unclear or you are not sure, mark the field as 'TBD' (To Be Determined) and mark a field as 'CONFLICT' in the following scenario:\nMultiple occurrences of the same field in the text with different answers.\n For Example if the Budget Constraints are mentioned twice or more in the PDF Input with contrasting values then that field should be marked as CONFLICT\n\n Give the JSON output with the filled fields only. Make sure that you consider all categories which are BIZ_OBJ, PROD_VARIANT_INFO, MATERIAL_HANDLING, SOFTWARE, CUSTOMER_DEPENDENCY and ACCEPTANCE. ENSURE THE JSON IS VALID AND PROPERLY FORMATTED. DO NOT OUTPUT ANYTHING OTHER THAN THE JSON. THE OUTPUT JSON SHOULD BE SAME AS THE GIVEN JSON JUST WITH FILLED USER ANSWERS FROM THE TEXT OR MARKED AS 'TBD' OR 'CONFLICT'."
            },
            {
                "role": "user",
                "content": "JSON: "+str(json_template)+"\n Text: "+text
            }
        ],
        temperature=0.21,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    answer = ""
    for chunk in completion2:
        answer += chunk.choices[0].delta.content or ""
    return answer

def question_create(json_template):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a JSON where some subproperties labelled \"User Answer\" are marked as \"TBD\". I want you to create questions that you as an assistant would ask the user in order to fill up the \"User Answer\" field. Create questions to fill these fields, considering the following:\n\n1. For \"TBD\" fields, ask for the missing information.\n2. Ensure questions are relevant to the field descriptions.\n3. Pay attention to required formats or units of measurement.\n4. Avoid asking questions and information about the fields which are not marked as 'TBD' in the JSON.\n\nReturn all the questions for the user in an array. Make sure that you consider all categories which are BIZ_OBJ, PROD_VARIANT_INFO, MATERIAL_HANDLING, SOFTWARE, CUSTOMER_DEPENDENCY and ACCEPTANCE. DO NOT MISS OUT ON ANY FIELD WITH \"User Answer\" SUB PROPERTY AS \"TBD\".DO NOT OUTPUT ANYTHING OTHER THAN THE QUESTION ARRAY." 
            },
            {
                "role": "user",
                "content": str(json_template)
            }
        ],
        temperature=0.21,
        max_tokens=2048,
        top_p=1,
        stream=True,
        stop=None,
    )

    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""

    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an experienced writer tasked with refining a set of questions. Follow these guidelines:\n\n1. Ignore all questions which requires uploading of images.\n2. Merge two or more questions asking about different aspects of the same topic.\n3. Maintain a professional yet slightly funny tone.\n4. Ensure questions are clear and concise.\n5. AVOID REDUNDANCY and limit the output to a maximum of 15 questions. Make sure that all the questions are asked within these 15 questions. All questions should be included in the least number of questions possible with the maximum being 15 questions.\n6. Format the questions to elicit precise answers that can be stored in a JSON structure.\n\nRETURN AN ARRAY OF THE REFINED QUESTIONS ONLY WITH NO OTHER FIELDS EXCEPT THE QUESTION ITSELF. MAKE SURE THE QUESTION DO NOT EXCEED 15 QUESTIONS. DO NOT RETURN ANYTHING ELSE."
            },
            {
                "role": "user",
                "content": answer
            }
        ],
        temperature=0.23,
        max_tokens=2240,
        top_p=1,
        stream=True,
        stop=None,
    )
    final=""
    for chunk in completion:
        final+=chunk.choices[0].delta.content or ""

    return final

def answer_refill(questions,answers,obs_json_template,bizobj_json_template):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You will be given two arrays: `questions` and `answers`. Your task is to create a list of question-answer pairs in the format 'Question: [question] Answer: [answer]' for each question and its corresponding answer. For example: Questions = ['What is the material of the observed object?', 'What are the dimensions of the object?'],Answers = ['The object appears to be made of stainless steel', '10 cm x 5 cm x 2 cm']['Question: What is the material of the observed object? Answer: stainless steel', 'Question: What are the dimensions of the object? Answer: 10 cm x 5 cm x 2 cm']. RETURN ONLY THE FINAL ARRAY OF QUESTION-ANSWER PAIRS."
            },
            {
                "role": "user",
                "content": "Question="+str(questions)+"\nAnswer="+str(answers)
            }
        ],
        temperature=0.5,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    qapair = ""
    for chunk in completion:
        qapair += chunk.choices[0].delta.content or ""
    # print(qapair)
    # print(obs_json_template+bizobj_json_template)
    # print("Question Answer:"+str(qapair)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template))
    completion2 = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a question-answer pair array and a two JSON templates. Follow these guidelines:\n\n1. Fill the \"User Answer\" subproperties in the JSONs based on the question-answer pairs. There might be a possiblity that the answer or a part of the answer is not relevant to the question but gives information about some other field in the JSON so classify the answers considering all the fields present in the JSON templates not just the fields relevant to the question.\n\n2. For fields still marked as \"TBD\" after filling, keep them as \"TBD\".\n\n3. If multiple answers conflict for the same field or there is an answer for an already filled field except \"TBD\", mark its  \"User Answer\" subproperty as \"CONFLICT\". Suppose in two answers the user gives the problem statement or budget constraint or any other field information twice with contrasting data then mark it as 'CONFLICT'. This situation can also arise when the \"User Answer\" subproperty is already filled in the given JSON input and the answer given by the userr to any of the questions provides conflicting information about the same field then that field \"User Answer\" subproperty should also be marked as 'CONFLICT'\n\n4. Ensure answers are relevant to field descriptions and adhere to specified formats or units.\n\n5. Do not infer or assume information until not explicitly stated.\n\n6. After filling, merge the two JSONs into a single JSON structure and make sure that there is NO RACE CONDITION while merging.\n\n7. Make sure to Return the complete, filled, and merged JSON without missing any field.\n\n8. Ensure the final JSON is valid and properly formatted without any error. Make sure that the final JSON returned has all fields and subproperties as given in the input just with the User Answers filled accordingly. DO NOT OUTPUT ANYTHING OTHER THAN THE FINAL MERGED JSON."
            },
            {
                "role": "user",
                "content": "Question Answer:"+str(qapair)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template)
            }
        ],
        temperature=1,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    filled_json=""
    for chunk in completion2:
        filled_json+=chunk.choices[0].delta.content or ""
    # print(filled_json)
    return filled_json

def text_refill(text,obs_json_template,bizobj_json_template):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion= client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a text and two JSON templates. Follow these guidelines:\n\n1. Fill the \"User Answer\" subproperties in the JSONs based on the text. Fill the subproperty on the basis of what is written in the text and classify it into the various fields of the JSON.\n\n2. For fields still marked as \"TBD\" after filling, keep them as \"TBD\".\n\n3. If multiple answers conflict for the same field or there is an answer for an already filled field except \"TBD\", mark its  \"User Answer\" subproperty as \"CONFLICT\". Suppose in two answers the user gives the problem statement or budget constraint or any other field information twice with contrasting data then mark it as 'CONFLICT'. This situation can also arise when the \"User Answer\" subproperty is already filled in the given JSON input and the answer given by the user to any of the questions provides conflicting information about the same field then that field \"User Answer\" subproperty should also be marked as 'CONFLICT'\n\n4. Ensure answers are relevant to field descriptions and adhere to specified formats or units.\n\n5. Do not infer or assume information until not explicitly stated.\n\n6. After filling, merge the two JSONs into a single JSON structure and make sure that there is NO RACE CONDITION while merging. Make sure you return the full JSON, without missing any field. \n\n7. Return the complete, filled, and merged JSON.\n\n8. Ensure the final JSON is valid and properly formatted. Make sure that the final JSON returned has all fields and subproperties as given in the input just with the User Answers filled accordingly. DO NOT OUTPUT ANYTHING OTHER THAN THE FINAL MERGED JSON."
            },
            {
                "role": "user",
                "content": "Text:"+str(text)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template)
            }
        ],
        temperature=0.53,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    filled_json=""
    for chunk in completion:
        filled_json+=chunk.choices[0].delta.content or ""
    # print(filled_json)
    return filled_json

import logging
logging.basicConfig(level=logging.ERROR)



def question_create_conflict(json_template):
    try:
        # First completion to generate initial questions
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        initial_completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a JSON where some subproperties labelled \"User Answer\" are marked as \"CONFLICT\" or \"TBD\". I want you to create questions that you as an assistant would ask the user in order to fill up the User Answer field. Create questions to fill these fields, considering the following:\n\n1. For 'CONFLICT' or 'TBD' fields, ask for the correct and precise information.\n2. Ensure questions are relevant to the field descriptions.\n3. Pay attention to required formats or units of measurement.\n4. Avoid asking about information already present in the JSON.\n\nReturn all the questions for the user in an array. DO NOT OUTPUT ANYTHING OTHER THAN THE QUESTION ARRAY. RETURN ONLY QUESTIONS DO NOT WRITE ANYTHING LIKE python or any other specifiers."
                },
                {
                    "role": "user",
                    "content": str(json_template)
                }
            ],
            temperature=0.21,
            max_tokens=2048,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Safely handle first streaming response
        answer = ""
        for chunk in initial_completion:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content

        # Second completion to refine questions
        refining_completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an experienced writer tasked with refining a set of questions. Follow these guidelines:\n\n1. Ignore any questions which requires uploading images.\n2. Merge questions asking about different aspects of the same fields.\n3. Maintain a professional yet slightly humorous tone.\n4. Ensure questions are clear and concise.\n5. AVOID REDUNDANCY and limit the output to a maximum of 15 questions. Make sure that all the questions are asked within these 15 questions. All questions should be included in the least number of questions possible with the maximum being 15 questions. Do not ask any extra questions.\n6. Format the questions to elicit precise answers that can be used in a JSON structure.\n\nRETURN AN ARRAY OF THE REFINED QUESTIONS ONLY WITH NO OTHER FIELDS EXCEPT THE QUESTION ITSELF. MAKE SURE THE QUESTION DO NOT EXCEED 15 QUESTIONS. DO NOT RETURN ANYTHING ELSE."
                },
                {
                    "role": "user",
                    "content": answer
                }
            ],
            temperature=0.23,
            max_tokens=2240,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Safely handle second streaming response
        final = ""
        for chunk in refining_completion:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                final += chunk.choices[0].delta.content

        return final

    except Exception as e:
        logging.error(f"Error in question_create_conflict: {str(e)}")
        return f"Error generating questions: {str(e)}"

def answer_refill_conflict(questions,answers,obs_json_template,bizobj_json_template):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You will be given two arrays: questions and answers. Create a question-answer pair array for each question asked for conflict fields. For example:\n\n#INPUT\nQuestions=['What is the material of the observed object?', 'What are the dimensions of the object?']\nAnswers=['The object appears to be made of stainless steel', '10 cm x 5 cm x 2 cm']\n\n#OUTPUT\n['Question: What is the material of the observed object? Answer: stainless steel','Question: What are the dimensions of the object? Answer: 10 cm x 5 cm x 2 cm']. RETURN ONLY THE FINAL ARRAY OF QUESTION-ANSWER PAIRS."
            },
            {
                "role": "user",
                "content": "Question="+str(questions)+"\nAnswer="+str(answers)
            }
        ],
        temperature=0.5,
        max_tokens=4048,
        top_p=1,
        stream=True,
        stop=None,
    )

    qapair = ""
    for chunk in completion:
        qapair += chunk.choices[0].delta.content or ""
    # print(qapair)
    # print(obs_json_template+bizobj_json_template)
    # print("Question Answer:"+str(qapair)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template))
    completion2 = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sophisticated classification assistant with expertise in engineering concepts. You will be given a question-answer pair array and a two JSON templates. Follow these guidelines:\n\n1. Fill the \"User Answer\" subproperties in the JSONs marked as \"CONFLICT\" or \"TBD\" based on the question-answer pairs.\n\n2. Ensure answers are relevant to field descriptions and adhere to specified formats or units.\n\n3. Do not infer or assume information until not explicitly stated.\n\n4. After filling, merge the two JSONs into a single JSON structure and make sure that there is NO RACE CONDITION while merging.Make sure you return the full JSON, without missing any field. \n\n5. Return the complete, filled, and merged JSON.\n\n6. Ensure the final JSON is valid and properly formatted. Make sure that the final JSON returned has all fields and subproperties as given in the input just with the User Answers filled accordingly. DO NOT OUTPUT ANYTHING OTHER THAN THE FINAL MERGED JSON."
            },
            {
                "role": "user",
                "content": "Question Answer:"+str(qapair)+"\nJSON:\n"+str(obs_json_template+bizobj_json_template)
            }
        ],
        temperature=1,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    filled_json=""
    for chunk in completion2:
        filled_json+=chunk.choices[0].delta.content or ""
    # print(filled_json)
    return filled_json


def executive_summary(json_template):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    # Placeholder for writing the summary status

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a professional copyrighter. You will be given a JSON, I want you to create a complete executive summary with headers and subheaders. It should be a structured document. \"User Answer\" are what are the answers you have to focus on. Dont skip any of the Fields in both JSONs. Use the Description to frame the User answer. DONT OUTPUT ANYTHING OTHER THAN THE SUMMARY."
            },
            {
                "role": "user",
                "content": str(json_template)
            }
        ],
        temperature=0.53,
        max_tokens=5610,
        top_p=1,
        stream=True,
        stop=None,
    )
    final_summ=""
    for chunk in completion:
        final_summ+=chunk.choices[0].delta.content or ""

    return final_summ

def airtable_write(json_template):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Groq inference
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": 
                    """You are an AI assistant specializing in JSON data restructuring. Your task is to convert unstructured JSON into a structured JSON array that can be easily transformed into a CSV format. Each object in the output array must have exactly four fields: "Category", "Sub-category", "Description", and "User Answer". Map the input fields as follows: use "Category" or "Observation type" for the "Category" field; "Field Name" or "Sub-Parameters" for the "Sub-category" field; "Description" or "Example" for the "Description" field; and fields explicitly marked as "User Answer" for the "User Answer" field. If multiple instances of the same field type exist, combine them logically or use the most relevant one. If a field doesn't clearly map to one of these four, use your best judgment to place it appropriately. Include all relevant information from the input, ensuring no data is lost. Your output must be a valid, properly formatted JSON array with no missing braces, quotes, or commas. Do not include any explanations, comments, or text outside the JSON structure. If the input is ambiguous or challenging to structure, still attempt to produce a valid JSON output with the available information. Your response should consist exclusively of the restructured JSON array, nothing else, as this output will be directly passed to an Airtable write function.\n """
            },
            {
                "role": "user",
                "content": json_template
            }
        ],
        temperature=0.27,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    content = ""
    for chunk in completion:
        content += chunk.choices[0].delta.content or ""
    
    print("Raw content from Groq:")
    print(content[:1000] + "..." if len(content) > 1000 else content)  # Print first 1000 chars for very long content
    
    # Write raw content to a file for potential debugging
    with open("raw_groq_output.json", "w") as file:
        file.write(content)
    
    # Parse the JSON content
    records = parse_json_content(content)
    
    if not records:
        print("Failed to parse JSON content. Exiting.")
        return

    API_KEY = os.getenv("AIRTABLE_KEY")
    BASE_ID = "appGIi65aZ2YxQrmH"
    TABLE_ID = "Table1"
    url = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}'

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    # Process records and send to Airtable
    process_records(records, url, headers)

    print("Airtable write operation completed.")

def parse_json_content(content):
    try:
        # First, try to parse as a JSON array
        records = json.loads(content)
        if isinstance(records, list):
            return records
        elif isinstance(records, dict):
            return [records]
    except json.JSONDecodeError:
        print("Failed to parse as a complete JSON. Attempting to parse line by line.")
        
        # If that fails, try to parse line by line
        records = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
                except json.JSONDecodeError:
                    print(f"Failed to parse line: {line}")
        
        if records:
            return records
    
    print("Failed to parse JSON content.")
    return None

def process_records(records, url, headers, batch_size=10):
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        airtable_data = {
            "records": [
                {
                    "fields": {
                        "Category": item.get("Category", ""),
                        "Sub-category": item.get("Sub-category", ""),
                        "Description": item.get("Description", ""),
                        "User Answer": item.get("User Answer", "")
                    }
                } for item in batch
            ]
        }
        
        send_to_airtable(url, headers, airtable_data)

def send_to_airtable(url, headers, data):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            print(f"Batch of {len(data['records'])} records added successfully!")
            return
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed. Error: {e}")
            if hasattr(e, 'response'):
                print(f"Response content: {e.response.text}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Failed to send batch to Airtable.")

def check_for_conflicts(completed_json: str) -> bool:
    # Check if 'CONFLICT' is present in the string (case-insensitive)
    if 'CONFLICT' in completed_json.upper():
        return True
    elif 'TBD' in completed_json.upper():
        return True
    else:
        return False

def extract_text_from_pdf(file_path):
    """
    Extract text from PDF file using OCR
    Args:
        file_path: Path to the PDF file
    Returns:
        str: Extracted text from PDF
    """
    try:
        # First try using PyPDF2 as it doesn't require external dependencies
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n\n"
            
            # If we got meaningful text, return it
            if len(full_text.strip()) > 50:  # Arbitrary threshold to check if we got real text
                return full_text.strip()
                
            # If PyPDF2 didn't get good text, fall back to OCR if poppler is available
            try:
                import pkg_resources
                pkg_resources.get_distribution('pdf2image')
                
                # Process PDF using OCR
                results = state.ocr_processor.process_pdf(file_path)
                
                # Combine text from all pages
                ocr_text = ""
                for result in results:
                    if result['confidence'] > 50:
                        cleaned_text = state.ocr_processor.clean_text(result['text'])
                        ocr_text += cleaned_text + "\n\n"
                
                return ocr_text.strip() if ocr_text.strip() else full_text.strip()
                
            except (ImportError, pkg_resources.DistributionNotFound):
                app.logger.warning("pdf2image/poppler not found. Using basic text extraction.")
                return full_text.strip()
                
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global state management (replace Streamlit session state)
class AppState:
    def __init__(self):
        self.file_processed = False
        self.current_question_index = 0
        self.questions = []
        self.conflict_resolution_mode = False
        self.obs = None
        self.bizobj = None
        self.text = None
        self.classification_result = None
        self.questionnaire_complete = False
        self.ocr_processor = LightweightOCRProcessor()

state = AppState()

@app.route('/api/process_additional_text', methods=['POST'])
def process_additional_text_endpoint():
    try:
        data = request.get_json()
        text_content = data.get('text')
        
        if not text_content:
            return jsonify({'error': 'No text content provided'}), 400

        # Call text_refill function
        updated_json = text_refill(text_content, state.obs, state.bizobj)
        
        # Update state
        state.obs = updated_json
        state.bizobj = updated_json

        if check_for_conflicts(updated_json):
            # Conflicts found, generate conflict questions
            conflict_questions = question_create_conflict(updated_json)
            state.questions = ast.literal_eval(conflict_questions)
            state.current_question_index = 0
            state.conflict_resolution_mode = True
            
            return jsonify({
                'status': 'conflicts_detected',
                'questions': state.questions
            })
        else:
            # No conflicts, proceed to finalization
            result = finalize_questionnaire_processing(updated_json)
            return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_document', methods=['POST'])
def process_document_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_path)

        # Read PDF file directly from memory
        try:
            extracted_text = extract_text_from_pdf(temp_path)  # Assume extract_text_from_pdf can handle file streams
            
            if not extracted_text:
                return jsonify({'error': 'Could not extract text from PDF'}), 400

            state.text = extracted_text
            # Use modified functions that handle chunking
            state.classification_result = classification_LLM(state.text)

            obs_json_path = os.path.abspath("./observationsJSON.json")
            bizobj_json_path = os.path.abspath("./bizobj.json")           
            # Process observations JSON
            with open(obs_json_path, 'r') as f:
                obs_json_template = json.load(f)
            final_obs_json = obsjsoncreate(obs_json_template, 
                                           state.classification_result, 
                                           state.text)
            state.obs = final_obs_json

            # Process business objects JSON
            with open(bizobj_json_path, 'r') as f:
                bizobj_json_template = json.load(f)
            final_bizobj_json = bizobjjsoncreate(bizobj_json_template, state.text)
            state.bizobj = final_bizobj_json

            # Generate questions
            questionobs = question_create(final_obs_json)
            questionbizobj = question_create(final_bizobj_json)
            
            obs_questions = ast.literal_eval(questionobs) if questionobs else []
            bizobj_questions = ast.literal_eval(questionbizobj) if questionbizobj else []
            state.questions = bizobj_questions + obs_questions
            
            state.file_processed = True
            
            return jsonify({
                'status': 'success',
                'questions': state.questions,
                'current_question_index': state.current_question_index
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            os.rmdir(temp_dir)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_add_document', methods=['POST'])
def process_add_document_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_path)

        try:
            # Extract text using PDF processing
            extracted_text = extract_text_from_pdf(temp_path)
            
            if not extracted_text:
                return jsonify({'error': 'Could not extract text from PDF'}), 400

            state.text = extracted_text

            
            state.file_processed = True

                # Call text_refill function
            updated_json = text_refill(extracted_text, state.obs, state.bizobj)
                
                # Update state
            state.obs = updated_json
            state.bizobj = updated_json

            if check_for_conflicts(updated_json):
                # Conflicts found, generate conflict questions
                conflict_questions = question_create_conflict(updated_json)
                state.questions = ast.literal_eval(conflict_questions)
                state.current_question_index = 0
                state.conflict_resolution_mode = True
                
                return jsonify({
                    'status': 'conflicts_detected',
                    'questions': state.questions
                })
            else:
                # No conflicts, proceed to finalization
                result = finalize_questionnaire_processing(updated_json)
                return jsonify(result)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            os.rmdir(temp_dir)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/show_question', methods=['POST'])
def show_question_endpoint():
    try:
        data = request.get_json()
        answer = data.get('answer')  # Previous answer if any

        if state.current_question_index >= len(state.questions):
            # All questions answered, process answers
            answers = data.get('all_answers', [])
            
            if not state.conflict_resolution_mode:
                completed_json = answer_refill(state.questions, answers, state.obs, state.bizobj)

                if check_for_conflicts(completed_json):
                    # Handle conflicts
                    conflict_questions = question_create_conflict(completed_json)
                    state.questions = ast.literal_eval(conflict_questions)
                    state.current_question_index = 0
                    state.conflict_resolution_mode = True
                    
                    return jsonify({
                        'status': 'conflicts_detected',
                        'questions': state.questions,
                        'current_question': state.questions[0]
                    })
                else:
                    # Finalize questionnaire
                    result = finalize_questionnaire_processing(completed_json)
                    return jsonify(result)
            else:
                # Process conflict resolution
                conflict_answers = answers[-len(state.questions):]
                completed_json = answer_refill_conflict(state.questions, conflict_answers, 
                                                      state.obs, state.bizobj)
                
                if check_for_conflicts(completed_json):
                    # Still have conflicts
                    conflict_questions = question_create_conflict(completed_json)
                    state.questions = ast.literal_eval(conflict_questions)
                    state.current_question_index = 0
                    
                    return jsonify({
                        'status': 'conflicts_continue',
                        'questions': state.questions,
                        'current_question': state.questions[0]
                    })
                else:
                    # Finalize questionnaire
                    result = finalize_questionnaire_processing(completed_json)
                    return jsonify(result)

        # Return next question
        current_question = state.questions[state.current_question_index]
        state.current_question_index += 1
        
        return jsonify({
            'status': 'question',
            'question': current_question,
            'question_number': state.current_question_index,
            'total_questions': len(state.questions)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def finalize_questionnaire_processing(completed_json):
    """Helper function for finalize_questionnaire logic"""
    try:
        time.sleep(60)
        # Write to Airtable
        airtable_write(completed_json)
        
        # Generate executive summary
        exec_summ = executive_summary(completed_json)
        
        # Reset state
        state.questionnaire_complete = True
        state.conflict_resolution_mode = False
        
        return {
            'status': 'complete',
            'executive_summary': exec_summ,
            'completed_json': completed_json
        }
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/get_json_state', methods=['GET'])
def get_json_state():
    try:
        return jsonify({
            'obs': state.obs,
            'bizobj': state.bizobj
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)