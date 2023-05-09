import re
import PyPDF2
from io import BytesIO

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', ' ', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

def extract_text_from_pdf(pdf_file):
    pdfReader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdfReader.pages)
    text = ''
    for page in range(num_pages):
        pdfPage = pdfReader.pages[page]
        text += pdfPage.extract_text()
    return text

def preprocess_resume_text(text):
    cleaned_text = cleanResume(text)
    return [cleaned_text]