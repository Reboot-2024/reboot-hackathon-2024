import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
import pypdfium2
import pytesseract
import docx2txt
import io
from dotenv import load_dotenv
from PIL import Image
import re

# Configure LLM with Gemini Key
load_dotenv()
genai.configure(api_key='AIzaSyBuinM21k9mshd-YQy7y_eHinclhDry-PY')

# Set the Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path as necessary

# User Define Functions
def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')  # Changed to gemini-pro
    response = model.generate_content(input)
    return response.text

def format_str(stng):
    if isinstance(stng, str):
        stng = ' '.join(stng.split())
        return str(stng)
    return np.nan

def format_email(stng):
    if isinstance(stng, str):
        stng = ''.join(stng.split())
        return str(stng)
    return np.nan

def format_phone(stng):
    if isinstance(stng, str):
        flag = False
        if '+' in stng:
            flag = True
        numbers = re.findall(r'\d+', stng)
        numbers = ''.join(numbers)
        if flag:
            numbers = "+"+numbers
        return str(numbers)
    return np.nan

def format_experience(stng):
    if isinstance(stng, str):
        match = re.search(r'\d+(\.\d+)?', stng)
        if match:
            return str(match.group())
    return np.nan


def pdf_bitmap_to_pil_image(pdf_bitmap):
    # Ensure the bitmap is in the right format
    width, height = pdf_bitmap.width, pdf_bitmap.height
    buffer = pdf_bitmap.to_numpy()  # Convert PdfBitmap to NumPy array

    # Check if the buffer has only one channel (grayscale)
    if len(buffer.shape) == 2:  # Grayscale image
        buffer = np.stack((buffer,) * 3, axis=-1)  # Convert to RGB

    try:
        pil_image = Image.fromarray(buffer)
    except Exception as e:
        print(f"Error creating PIL image: {e}")
        return None  # Handle as appropriate
    return pil_image

def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name
    extension = filename.split('.')[-1].lower()
    text = ""

    if extension == 'pdf':
        # Read PDF and extract text using PDFium and pytesseract
        pdf_reader = pypdfium2.PdfDocument(uploaded_file)
        for page_number in range(len(pdf_reader)):
            page = pdf_reader.get_page(page_number)
            pdf_bitmap = page.render(scale=4.0)  # Increase the scale for better resolution
            pil_image = pdf_bitmap_to_pil_image(pdf_bitmap)
            if pil_image is not None:  # Check if image conversion was successful
                text += pytesseract.image_to_string(pil_image)

    elif extension == 'docx':
        # Extract text using docx2txt
        text = docx2txt.process(io.BytesIO(uploaded_file.read()))

    else:
        print(f"Unsupported file format: {extension}")

    return text

# Prompt to match resumes to JD
input_prompt_2 = """
You are an HR executive of a company and your role is to match the candidates resumes to the Job Description provided.
You need to look at both resume and job description, use techniques like TF-IDF or word embeddings to give a standard score so that it will be constant for a resume and job description.
Recent role should be the last working role in the experience section.
Total Experience should be the sum of all the periods working for a company in experience section.
In experience section there might be more than one company so consider all the individual companies as work experience and sum all of them and make sure to specify exact number.
Highest Qualification should be the latest education in the resume.
Skills must be extracted from the skills section. Do not keep it empty. Srictly list only the important skills within 20 words.
Summary should be brief in 30 words
If you do not find any information return as 'Information Not Found'. Do check the information again if it is available before judging as 'Information not found' 
Consider the inputs here:
Job Description: {JD}
Resume:{text}

The output should be in this format and mention skills in a single straight line:
Candidate Name:
Matching Percentage:
Matching Skills:
Contact Number:
Email id:
Highest Qualification:
Recent Role:
Total years of experience:
Summary of the resume:

"""
# Set app title
st.set_page_config(page_title="Resume Screener", layout="wide")

# Create containers for horizontal alignments
container1 = st.container()
container2 = st.container()

# Create columns for vertical alignments
col3, col4 = st.columns([1.5, 4])
image = col3.image("images.png", width=200)
col4.write('<h1 style="color: green;margin-top:5px;">Reboot @ Lloyds Technology Center</h1>', unsafe_allow_html=True)

st.write('<h1 style="color: black; width: 10000px;padding:0px;">Resume Scanner</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([2.8, 1])

container3 = st.container()

# This container has Job Description and Upload Resumes styling
with container1: 
    # Style for Job Description text area
    st.write('<style>.st-bp { font-size: 14px; padding: 0px; gap:0px;} /* Adjust styles as needed */</style>', unsafe_allow_html=True)
    job_description = col1.text_area("Enter Your Job Description", key="job_description", height=300)

    uploaded_files = col2.file_uploader("Upload Resumes", type=['pdf', 'docx'], accept_multiple_files=True)
    upload_clicked = col2.button("Upload")
    
    if uploaded_files and upload_clicked:
        progress_bar = col2.progress(0)
        for perc_completed in range(100):
            time.sleep(0.01)
            progress_bar.progress(perc_completed + 1)

        col2.success('Resumes have been uploaded')

# This container starts the process once the Analyze is clicked
with container2: 
    analyze_clicked = col1.button("Analyze")
    if analyze_clicked:
        start_time = time.perf_counter()

        JD_text = job_description

        # Main loop to get the data from LLM
        data = []
        progress_text = col1.empty()
        progress_bar = col1.progress(0)
        loop_counter = 0

        def update_progress(loop_counter, total_resumes):
            return f"Processed {loop_counter} out of {total_resumes} resumes"

        # Passing each resume through the loop
        for CV_file in uploaded_files:
            CV_text = extract_text_from_file(CV_file)
            formatted_prompt = input_prompt_2.format(text=CV_text, JD=JD_text)
            response = get_gemini_response(formatted_prompt)  
            
            response_lines = response.split('\n')
            info = {
                "File Name": CV_file.name,
                "Candidate Name": "Information Not Found",
                "Matching Percentage": "Information Not Found",
                "Matching Skills": "Information Not Found",
                "Contact Number": "Information Not Found",
                "Email id": "Information Not Found",
                "Highest Qualification": "Information Not Found",
                "Recent Role": "Information Not Found",
                "Total years of experience": "Information Not Found",
                "Summary of the resume": "Information Not Found",
            }
            for line in response_lines:
                if "Candidate Name:" in line:
                    info["Candidate Name"] = format_str(line.split(":", 1)[1].strip()).title()
                elif "Matching Percentage:" in line:
                    info["Matching Percentage"] = line.split(":", 1)[1].strip()
                elif "Matching Skills:" in line:
                    info["Matching Skills"] = line.split(":", 1)[1].strip()
                elif "Contact Number:" in line:
                    info["Contact Number"] = format_phone(line.split(":", 1)[1].strip())
                elif "Email id:" in line:
                    info["Email id"] = format_email(line.split(":", 1)[1].strip()).lower()
                elif "Highest Qualification:" in line:
                    info["Highest Qualification"] = format_str(line.split(":", 1)[1].strip()).title()
                elif "Recent Role:" in line:
                    info["Recent Role"] = format_str(line.split(":", 1)[1].strip()).title()
                elif "Total years of experience:" in line:
                    info["Total years of experience"] = format_experience(line.split(":", 1)[1].strip())
                elif "Summary of the resume:" in line:
                    info["Summary of the resume"] = line.split(":", 1)[1].strip()

            data.append(info)
            loop_counter += 1
            progress_bar.progress(int(loop_counter / len(uploaded_files) * 100))
            progress_text.text(update_progress(loop_counter, len(uploaded_files)))

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        col1.success(f"Hola! Finished the analysis in {elapsed_time:.2f} seconds")

        # Transforming the data into a dataframe
        df = pd.DataFrame(data)
        styles = """
                    td {
                        max-width: 0px;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                        color: black;
                        background-color: beige;
                        border-radius: 5px;
                    }

                    th {
                        background-color: lightseagreen;
                        font-weight: bold;
                    }

                    table {
                        width: 40%;
                        border-radius: 5px;
                    }
                    """
        # This container is for the table
        with container3: 
            st.markdown(f'<style>{styles}</style>', unsafe_allow_html=True)
            st.table(df)
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            st.download_button("Download CSV", df.to_csv(), mime='text/csv', file_name=(f"matching_score_{timestamp}.csv"))

# Make sure to install required packages
# pip install streamlit
