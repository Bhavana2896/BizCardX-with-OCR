#BizCardX: Business Card Extractor with OCR

BizCardX is a Python application that uses OCR (Optical Character Recognition) to extract information from business cards. It can extract details like name, company name, contact numbers, email, website, and address from uploaded images of business cards.

Features
Extracts business card information from images.
Supports multiple phone numbers and email addresses.
Saves extracted data to a PostgreSQL database.

Usage
Launch the Streamlit app:streamlit run app.py
1. The app opens in your default web browser.
2. Select the "Extract & Upload" option to upload a business card image. Click "Extract Information" to process the image and display extracted data.
3. Select the "Modify" option to update or delete existing data. Choose a row from the dropdown menu, make the necessary changes, and click "Update" or "Delete".
4. Select the "Download" option to save the data as a CSV or PDF file.

Function Descriptions
fetch_data()
Fetches data from the PostgreSQL database and returns it as a Pandas DataFrame.

extract_information(result, result_1)
Extracts information from the OCR results and returns a dictionary containing business card details.

extract_and_upload()
Provides a file upload widget for users to upload business card images. Upon upload, it processes the image and displays extracted information.

insert_data(extracted_data)
Inserts the extracted data into the PostgreSQL database.

update_data(selected_row_name, new_values)
Updates existing data in the database.

delete_data(selected_row)
Deletes a row from the database.

Contributing
Fork the project.
Create your feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Feel free to adjust any details as per your actual project and add any additional sections you think would be helpful for users and contributors.







