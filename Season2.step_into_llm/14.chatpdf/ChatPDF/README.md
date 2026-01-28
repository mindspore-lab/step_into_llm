# ChatPDF

The ChatPDF(PDF Chatbot) is an application that allows users to upload PDF files and interact with pdf using a chatbot. Users can ask questions or provide input, and the chatbot will generate responses based on the provided information.

## Technologies Used

- MindSpore >= 2.6.0
- MindNLP >= 0.4.0
- ms2vec
- msimilarities
- Python 3.11

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/babahaochi/step_into_llm.git
   cd step_into_llm/Season2.step_into_llm/14.chatpdf/ChatPDF
   ```

2. Create a virtual environment and install the required dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Migration Notes (MindNLP 0.4.0)

This project has been updated to support MindSpore 2.6.0 and MindNLP 0.4.0. Key changes include:
- Migrating tensor operations to use `mindnlp.core.ops`.
- Updating imports to align with the new MindNLP directory structure.
- Ensuring compatibility with the latest `PeftModel` and `AutoModel` APIs.
## Usage

1. Run the application:

   ```bash
   # complex version
   python simple_ui.py
   # complex version
   python complex_ui.py
   ```

2. Access the application in your web browser as specified in the console.

3. To preview a PDF file, click the "Upload PDF" button and select the PDF file from your local machine. The application will display a preview of the PDF file.

5. Use the chatbox to ask questions or have a conversation with the chatbot. The chatbot will generate responses based on the input.
