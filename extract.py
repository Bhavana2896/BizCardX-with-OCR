import streamlit as st
from streamlit_option_menu import option_menu
import easyocr
import pandas as pd
from PIL import Image
import re
import io
# from io import BytesIO
from fpdf import FPDF
import numpy as np
import psycopg2

# Initialize EasyOCR reader
#The output will be in a list format, each item represents a bounding box,
# the text detected and confident level, respectively.

reader = easyocr.Reader(['en'])
df_data = {
            'Field': [],
            'Value': []
        }
extracted_data = {
        'Name': None,
        'Designation': None,
        'Company Name': None,
        'Primary Number': None,
        'Secondary Number': None,
        'Email': None,
        'Website': None,
        'Address': None
    }
# Establish a connection to your PostgreSQL database
conn = psycopg2.connect(
    host='localhost',
    user='bhavana',
    password='Girish47$',
    port=5432,
    database='Biz'
)
# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Define the SQL CREATE TABLE statement
# Choose TEXT to accommodate longer text
create_table_sql = """
CREATE TABLE IF NOT EXISTS business_card (
    name TEXT,
    designation TEXT,
    company_name TEXT,
    primary_number TEXT,
    secondary_number TEXT,
    email TEXT,
    website TEXT,
    address TEXT,
    image BYTEA
);
"""
# Execute the SQL command to create the table
cursor.execute(create_table_sql)
# Commit the changes
conn.commit()
def fetch_data():
    cursor.execute("SELECT * FROM business_card")
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(data, columns=columns)
    return df


def extract_information(result,result_1):
    # Extract phone numbers with various formats
    phone_numbers = []
    invalid_numbers = []
    emails = []
    websites = []

    # Compile the address regex pattern
    # address_regex = re.compile(r'[\w\s\.\,]*[\w\s,]+')
    name_regex = re.compile(r'[A-Za-z]{2,25}( [A-Za-z]{2,25})?')
    # company_name_regex = r'(?:[A-Z][a-z0-9\s]+\s?(?:Inc\.|Agency|Co\.|Co\.|Electricals|Restaurant|AIRLINES|INSURANCE|MEDICAL|digitals))|(?:[A-Z][a-z\s]*[A-Z]\s?(?:Inc\.|Agency|Co\.|Co\.|Electricals|Restaurant|AIRLINES|INSURANCE|MEDICAL|digitals))'

    # # Find the text with the maximum sum of y-values
    # max_sum_y = 0
    # bottom_text = None
    #
    # for bounding_box, text, confidence in result:
    #     sum_y = sum(point[1] for point in bounding_box)
    #     if sum_y > max_sum_y:
    #         max_sum_y = sum_y
    #         bottom_text = text

    # Find the longest text
    # longest_text = max(result, key=lambda x: len(x[1]))[1]
    # Process the EasyOCR results
        # if text == longest_text and text == bottom_text and address_regex.match(text):
        #     extracted_data['Address'] = text
        # elif text == bottom_text and address_regex.match(text):
        #     extracted_data['Address'] = bottom_text
        # elif address_regex.match(text):
        #     extracted_data['Address'] = text

    text_name = result[0][1]
    text_designation = result[1][1]
    keywords = ['Inc.', 'Agency', '& Co.', '& co', 'Electricals', 'Restaurant', 'AIRLINES', 'INSURANCE', 'MEDICAL', 'digitals', 'Corporation', 'Corp.']
    tlds = ['com', 'org', 'net', 'edu', 'gov', 'info', 'biz', 'io', 'in']
    if re.match(name_regex, text_name):
        extracted_data['Name'] = text_name
    if re.match('\w+', text_designation):
        extracted_data['Designation'] = text_designation
        # if re.findall(r'[A-Z][a-z]+(?: [A-Z]\.?| [A-Z][a-z]+(?: [A-Z]\.?)?)? [A-Z][a-z]+', text):

    for bounding_box, text, confidence in result:
        if re.findall(r'^\w+[\.+\"\-_]?[\w+]?@\w+[\.+\"\-_\w]?.[a-zA-Z]{2,}?', text):
            emails.append(text)
            extracted_data['Email'] = text
    for bounding_box, text, confidence in result:
        stripped_text = text.strip()  # Remove leading and trailing spaces
        if (re.findall(r'^(?:[wW]+\s*\.)*\s*[\w-]+\.[a-zA-Z]{2,}', stripped_text) or any(stripped_text.lower().endswith(tld.lower()) for tld in tlds)) and not extracted_data['Email']:
            websites.append(text)
            extracted_data['Website'] = stripped_text
            # Exit the loop after finding the first valid website
            break
        # elif 'Designation' not in extracted_data:
        #     extracted_data['Designation'] = text
    # for bounding_box, text, confidence in result:
    #     if len(text) > 5 and re.match(company_name_regex, text):
    #         extracted_data['Company Name'] = text

    for bounding_box, text, confidence in result:
        if re.findall(r'^\+?\d*\s?\(?\d{3}\)?[\-\.\s]?\d{3}[\-\.\s]?\d{4}?', text):
            phone_numbers.append(text)
        elif re.findall(r'\+?\d+-\d+-\d+',text):
            invalid_numbers.append(text)

        # elif re.findall(r'^\w+[\.+\"\-_]?[\w+]?@\w+[\.+\"\-_\w]?.[a-zA-Z]{2,}?', text):
        #     extracted_data['Email'] = text
        # elif re.findall(r'^(www\.)?[\w-]+\.[a-zA-Z]{2,}?', text):
        #     extracted_data['Website'] = text
        # Assign phone numbers to extracted_data
    if len(phone_numbers) >= 2:
        extracted_data['Primary Number'] = ''.join(phone_numbers[0])
        extracted_data['Secondary Number'] = ''.join(phone_numbers[1])
    elif len(phone_numbers) == 1:
        extracted_data['Primary Number'] = ''.join(phone_numbers[0])
        extracted_data['Secondary Number'] = 'Not found'
    elif len(invalid_numbers) >= 2:
        extracted_data['Primary Number'] = 'Invalid number found'
        extracted_data['Secondary Number'] = 'Invalid number found'
    elif len(invalid_numbers) == 1:
        extracted_data['Primary Number'] = 'Invalid number found'
        extracted_data['Secondary Number'] = 'Not found'
    else:
        extracted_data['Primary Number'] = 'Not found'
        extracted_data['Secondary Number'] = 'Not found'
    extracted_data['Company Name'] = 'None'
    for bounding_box, text in result_1:
        # text = text.replace('\n', ' ').strip()
        if len(text) >= 15 and text not in extracted_data.values() and ',' in text:
            if any(invalid_number in text for invalid_number in invalid_numbers):
                for invalid_number in invalid_numbers:
                    # Omitting invalid_numbers from text
                    text = text.replace(invalid_number, '')
            for phone_number in phone_numbers:
                # Omitting invalid_numbers from text
                text = text.replace(phone_number, '')
            for website in websites:
                # Omitting invalid_numbers from text
                text = text.replace(website, '')
            for email in emails:
                # Omitting invalid_numbers from text
                text = text.replace(email, '')
            extracted_data['Address'] = text
    for bounding_box, text in result_1:
        if any(keyword.lower() in text.lower() for keyword in keywords) and len(text)<=25:
            extracted_data['Company Name'] = text
            # Exit the loop once a keyword is found
            break


        # # Extract name based on variations
        # if re.findall(r'\b[A-Z][a-z]+(?: [A-Z]\.?| [A-Z][a-z]+(?: [A-Z]\.?)?)? [A-Z][a-z]+\b', text):
        #     extracted_data['Name'] = text
        # # Extract designation using EasyOCR's built-in text recognition
        # elif 'Designation' not in extracted_data:
        #     extracted_data['Designation'] = text
        # # Extract company name without explicit "Company"
        # elif len(text) > 5:  # Assuming company name is generally longer than 5 characters
        #     extracted_data['Company Name'] = text
        # # Extract phone numbers with various formats
        # elif re.findall(r'^\+?\d*\s?\(?\d{3}\)?[\-\.\s]?\d{3}[\-\.\s]?\d{4}?', text):
        #     if 'Primary Number' not in extracted_data:
        #          extracted_data['Primary Number'] = text
        #     elif 'Secondary Number' not in extracted_data:
        #         extracted_data['Secondary Number'] = text
        # # Extract email based on pattern
        # elif re.findall(r'^\w+[\.+\"\-_]?[\w+]?@\w+[\.+\"\-_\w]?.[a-zA-Z]{2,}?', text):
        #     extracted_data['Email'] = text
        # # Extract website with variations
        # elif re.findall(r'^(www\.)?[\w-]+\.[a-zA-Z]{2,}?', text):
        #     extracted_data['Website'] = text
        # Everything else as address
        # else:
        #     extracted_data['Address'] = text

    return extracted_data

#-------------------  Push Data  --------------------#
def insert_data(extracted_data,image_bytes):
    try:
        # Insert data into the table
        insert_query = """
        INSERT INTO business_card (name, company_name, primary_number, secondary_number, email, website, address, image)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        # # Encode the image as bytes
        # image_bytes = io.BytesIO(file_bytes).read()

        # Define the data to insert
        data_to_insert = (
            extracted_data['Name'],
            extracted_data['Designation'],
            extracted_data['Company Name'],
            extracted_data['Primary Number'],
            extracted_data['Secondary Number'],
            extracted_data['Email'],
            extracted_data['Website'],
            extracted_data['Address'],
            image_bytes
        )
        # Execute the INSERT statement
        cursor.execute(insert_query, data_to_insert)
        # Commit the transaction
        conn.commit()

    except Exception as error:
        print("Error:", error)

def extract_and_upload():
    uploaded_file = st.file_uploader("Upload a business card image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Convert uploaded file to bytes
        file_bytes = uploaded_file.read()
        # Open image using PIL
        image = Image.open(io.BytesIO(file_bytes))
        # Convert image to numpy array
        image_array = np.array(image)
        # Initialize extracted_data with a default value
        extracted_data = {}
        image_col, data_col = st.columns(2)
        # Display the image using Streamlit
        with image_col:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Extract & Upload Information'):
            result = reader.readtext(image_array)
            result_1 = reader.readtext(image_array, paragraph=True)
            extracted_data = extract_information(result, result_1)

            # Insert data into the database
            insert_data(extracted_data, file_bytes)

            # df_data = {'Field': [], 'Value': []}
            #
            # for key, value in extracted_data.items():
            #     if key == 'Primary Number' or key == 'Secondary Number':
            #         for idx, number in enumerate(value):
            #             df_data['Field'].append(f"{key} {idx + 1}")
            #             df_data['Value'].append(number)
            #     else:
            #         df_data['Field'].append(key)
            #         df_data['Value'].append(value)
            #
            # df = pd.DataFrame(df_data)
            # st.table(data=df.T)
        with data_col:
            st.write('')
            st.write('')
            for k, v in extracted_data.items():
                st.markdown(f"**{k}:** {v}")

def update_data(selected_row, new_values):
    try:
        # Execute an SQL UPDATE statement
        update_query = """
        UPDATE business_card
        SET name = %s, designation = %s, company_name = %s, primary_number = %s, secondary_number = %s,
        email = %s, website = %s, address = %s
        WHERE name = %s
        """
        data_to_update = (
            new_values['Name'],
            new_values['Designation'],
            new_values['Company Name'],
            new_values['Primary Number'],
            new_values['Secondary Number'],
            new_values['Email'],
            new_values['Website'],
            new_values['Address'],
            selected_row[0]# Assuming 'name' is the unique identifier
            )
        cursor.execute(update_query, data_to_update)
        conn.commit()
        st.success('Data updated successfully!')
    except Exception as error:
        st.error(f'Error: {error}')

# Function to delete data from the database
def delete_data(selected_row):
    try:
        # Execute an SQL DELETE statement
        delete_query = "DELETE FROM business_card WHERE name = %s"
        cursor.execute(delete_query, (selected_row[0],))
        conn.commit()
        st.warning('Data deleted successfully!')
    except Exception as error:
        st.error(f'Error: {error}')

# def modify_data():
#     st.write('Select a row to update or delete:')
#     selected_row_name = st.selectbox("Select a row:", [row[0] for row in data])
#
#     if selected_row_name:
#         # Fetch the details of the selected row
#         selected_row = [row for row in data if row[0] == selected_row_name][0]
#
#         # Create input fields for new values
#         new_values = {}
#         for i, column in enumerate(extracted_data.keys()):
#             new_values[column] = st.text_input(f'{column}:', selected_row[i])
#
#         # Add an update button
#         if st.button('Update'):
#             update_data(selected_row_name, new_values)
#
#         # Add a delete button
#         if st.button('Delete'):
#             delete_data(selected_row)
#
#         # Display selected row details
#         st.write('Selected Row Details:')
#         st.write({k: v for k, v in zip(extracted_data.keys(), selected_row)})

# def download_data():

# --------------------------- STREAMLIT APP --------------------------#
# Set the layout to wide
st.set_page_config(layout="wide")
# Center-align the title
# col1, col2, col3 = st.columns([2,8,1])
# with col2:
#     st.title(":red[BizCardX: Business Card Extractor with OCR]")
# Define the title with HTML tags for rainbow colors
rainbow_title = """
<div style="background-image: linear-gradient(to right, violet, indigo, blue, green, yellow, orange, red); 
            -webkit-background-clip: text;
            color: transparent;
            font-size: 48px; text-align: center;">
    BizCardX: Business Card Extractor with OCR
</div>
"""
# Display the rainbow title
st.markdown(rainbow_title, unsafe_allow_html=True)
st.write('')
st.write('')
# Modify option menu
selected = option_menu(None, ["Home","Extract & upload",  "Modify", "Download"],
    icons=['house', "database-fill-add", "gear", 'download'],
    menu_icon="cast", default_index=0, orientation="horizontal")
# Display selected page based on the menu selection
if selected == 'Home':
    # Instructions
    st.write("")
    st.write("")
    st.markdown("""
        :orange[**Welcome to the Business Card Extraction tool!**] This tool allows you to upload a business card image 
        and extract relevant information using Optical Character Recognition (OCR).

        OCR is a technology that extracts text from images, making it useful for tasks like business card 
        scanning, document digitization, and more. EasyOCR is a powerful OCR library in Python that simplifies 
        the process of extracting text from images.
        """)
    st.write("")
    # Developer Contact Details
    st.markdown(":green[**ABOUT & CONTACT DETAILS**]")
    # Brief About Me
    st.markdown("I am  Bhavana.B, a passionate data enthusiast.I love exploring the insights hidden within data. "
                "With a knack for uncovering patterns and trends, I thrive on turning raw information into meaningful insights."
                " Let's dive into the world of business card OCR!")
    # Developer Info Columns
    st.markdown("[Follow on LinkedIn](https://linkedin.com/in/bbhavana)")
    st.markdown("[Follow on Github](https://github.com/Bhavana2896)")
elif selected == 'Extract & upload':
    # Display content for Extract & upload page
    extract_and_upload()
elif selected == 'Modify':
    st.title("Modify Business Card Data")

    # Fetch data from the database
    data = fetch_data()

    if data is not None:
        # Exclude 'image' from the options
        valid_columns = data.columns[data.columns != 'image']

        # Select a row to update or delete
        st.write('Select a row to update or delete:')

        # Display select boxes in the same row
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_col_name = st.selectbox("Select a column name:", valid_columns)
        # Get distinct values of the selected column
        distinct_values = data[selected_col_name].unique()
        with col2:
            # Select a value from the distinct values
            selected_value = st.selectbox(f"Select a value for '{selected_col_name}':", distinct_values)
        # Filter the data to get the rows with the selected value
        filtered_data = data[data[selected_col_name] == selected_value]
        # Get the names of the rows for the selected value
        row_names = filtered_data['name'].tolist()
        with col3:
            # Select a row from the filtered data
            selected_row_name = st.selectbox("Select a row:", row_names)

        # If a row is selected, proceed with update
        if selected_row_name:
            # Fetch the details of the selected row
            selected_row = filtered_data[filtered_data['name'] == selected_row_name].iloc[0]
            # Create input fields for new values
            new_values = {}
            # Display selected row details
            st.write('Selected Row Details:')
            st.write({k: v for k, v in zip(extracted_data.keys(), selected_row)})
            for column in valid_columns:
                new_values[column] = st.text_input(f'{column}:', selected_row[column])

            col4, col5, col6 = st.columns([10, 2, 2])
            # Add an update button
            with col5:
                if st.button('Update'):
                    update_data(selected_row, new_values)
            # Add a delete button
            with col6:
                if st.button('Delete'):
                    delete_data(selected_row)

        # elif selected_col_name:
        #     st.error("please enter a row")

elif selected == 'Download':
    st.title("Download Data")

    # Fetch data from the database
    df = fetch_data()
    df_without_image = df.iloc[:, :-1]  # Exclude the last column (image)
    # # Provide options for downloading
    # download_format = st.selectbox("Select download format:", ["CSV", "PDF"])

    # Download as CSV
    csv_data = df_without_image.to_csv(index=True).encode('utf-8')
    st.download_button(label="🔻 Download data as CSV 🔻", data=csv_data,
                       file_name='business_card_info.csv', mime='text/csv')
#------------------------------update section -------------------------#
       # st.title("Download Data")
       #
       #  # Fetch data from the database
       #  df = fetch_data()
       #
       #  # Provide options for downloading
       #  download_format = st.selectbox("Select download format:", ["CSV", "PDF"])
       #
       #  if download_format == "CSV":
       #      # Download as CSV
       #      csv_data = df.to_csv(index=False)
       #      st.download_button(label="🔻 Download data as CSV 🔻", data=csv_data,
       #                         file_name='business_card_info.csv', mime='text/csv')
       #  else:
       #      # Download as PDF
       #      pdf_data = df.to_pdf()
       #      st.download_button(label="🔻 Download data as PDF 🔻", data=pdf_data,
       #                         file_name='business_card_info.pdf', mime='application/pdf')
