import fitz
from PIL import Image
import gradio as gr
from chatpdf import ChatPDF

# 使用延迟加载模式，避免在导入时加载模型
_model = None

def _get_model():
    """延迟加载模型"""
    global _model
    if _model is None:
        _model = ChatPDF()
    return _model

# Function to add text to the chat history
def add_text(history, text):
    """
    Adds the user's input text to the chat history.

    Args:
        history (list): List of tuples representing the chat history.
        text (str): The user's input text.

    Returns:
        list: Updated chat history with the new user input.
    """
    if not text:
        raise gr.Error('Enter text')
    history.append((text, ''))
    return history


def predict_stream(message, history):
    model = _get_model()
    history_format = []
    for human, assistant in history:
        history_format.append([human, assistant])
    model.history = history_format
    for chunk in model.predict_stream(message):
        yield chunk

# Function to generate a response based on the chat history and query
def generate_response(history, query, btn):
    """
    Generates a response based on the chat history and user's query.

    Args:
        history (list): List of tuples representing the chat history.
        query (str): The user's query.
        btn (FileStorage): The uploaded PDF file.

    Returns:
        tuple: Updated chat history with the generated response and the next page number.
    """
    if not btn:
        raise gr.Error(message='Upload a PDF')

    model = _get_model()
    history_format = []
    for human, assistant in history:
        history_format.append([human, assistant])
    model.history = history_format
    for chunk in model.predict_stream(query):
        history[-1][-1] = chunk
        yield history, " "

# Function to render a specific page of a PDF file as an image
def render_file(file):
    """
    Renders a specific page of a PDF file as an image.

    Args:
        file (FileStorage): The PDF file.

    Returns:
        PIL.Image.Image: The rendered page as an image.
    """
    # global n
    model = _get_model()
    model.reset_corpus(file)
    doc = fitz.open(file.name)
    page = doc[0]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

def clear_chatbot():
    return []
