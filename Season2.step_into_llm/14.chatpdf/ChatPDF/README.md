# ChatPDF

The ChatPDF (PDF Chatbot) is an application that allows users to upload PDF files and interact with PDF using a chatbot. Users can ask questions or provide input, and the chatbot will generate responses based on the provided information.

## Technologies Used

- **MindSpore**: Deep learning framework (>= 2.6.0)
- **MindNLP**: Natural Language Processing toolkit (>= 0.4.1)
- **ms2vec**: Sentence embedding library
- **msimilarities**: Similarity search library
- **Gradio**: Web UI framework

## Version Requirements

- **Python**: >= 3.9 (Recommended: 3.11)
- **MindSpore**: >= 2.6.0
- **MindNLP**: >= 0.4.1 (Tested with 0.4.1)

**Note**: This code has been migrated to be compatible with MindSpore 2.6.0+ and MindNLP 0.4.0+. If you need to use older versions, please check the git history for the previous version.

## Installation

### Option 1: Quick Installation (Recommended)

Use the provided installation script:

```bash
# Windows
install.bat

# Linux/Mac
bash install.sh
```

### Option 2: Manual Installation

1. Clone the repository:

```bash
git clone https://github.com/mindspore-courses/step_into_llm.git
cd step_into_llm/Season2.step_into_llm/14.chatpdf
```

2. Create a Python 3.11 virtual environment:

```bash
# Windows
python -m venv chatpdf_env
chatpdf_env\Scripts\activate

# Linux/Mac
python3.11 -m venv chatpdf_env
source chatpdf_env/bin/activate
```

3. Install dependencies with locked versions:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install MindSpore 2.6.0 (CPU version)
pip install mindspore==2.6.0 -f https://www.mindspore.cn/install

# Install MindNLP 0.4.1
pip install mindnlp==0.4.1

# Install other dependencies
pip install -r ChatPDF/requirements_locked.txt
```

### Option 3: Using requirements.txt

For development or testing, you can use the standard requirements file:

```bash
pip install -r ChatPDF/requirements.txt
```

**Warning**: Using the standard requirements.txt may install the latest versions, which might not be compatible. For production use, please use `requirements_locked.txt`.

## Migration Notes (From MindNLP 0.3.x to 0.4.x)

This code has been migrated according to [PR #1519](https://github.com/mindspore-lab/mindnlp/pull/1519/files). Key changes:

### 1. Model Training Mode API
**Before (MindNLP < 0.4.0):**
```python
model.set_train(False)
```

**After (MindNLP >= 0.4.0):**
```python
model.set_train(mode=False)
```

### 2. Functional API Changes
**Before:**
```python
ops.log_softmax(logits, axis=-1)
ops.gather_elements(dim=-1, index=labels)
ops.cat(tensors, axis=0)
```

**After:**
```python
F.log_softmax(logits, dim=-1)  # Using functional module
ops.gather(tensors, dim=-1, index=labels)
ops.cat(tensors, dim=0)
```

### 3. Optimizer Import Changes
**Before:**
```python
import mindspore.experimental.optim as optim
```

**After:**
```python
from mindnlp.core import optim
```

## Usage

1. Run the application:

```bash
# Simple version
python simple_ui.py

# Complex version with PDF preview
python complex_ui.py
```

2. Access the application in your web browser at `http://localhost:8082`

3. Upload a PDF file using the "Upload PDF" button

4. Ask questions in the chatbox

## API Usage Example

```python
import sys
sys.path.insert(0, 'ChatPDF')
from chatpdf import ChatPDF

# Initialize ChatPDF with default settings
chatpdf = ChatPDF(
    generate_model_name_or_path="01ai/Yi-6B-Chat",
    corpus_files="sample.pdf",
    chunk_size=250
)

# Ask a question
response, references = chatpdf.predict("What is the main topic of this paper?")
print(response)
```

## Directory Structure

```
14.chatpdf/
├── ChatPDF/
│   ├── chatpdf.py           # Main ChatPDF class
│   ├── logic.py             # UI interaction logic
│   ├── simple_ui.py         # Simple Gradio UI
│   ├── complex_ui.py        # Advanced UI with PDF preview
│   ├── requirements.txt     # Dependency specifications
│   ├── requirements_locked.txt  # Version-locked dependencies
│   ├── sample.pdf           # Example PDF file
│   └── README.md            # This file
├── install.bat              # Windows installation script
├── install.sh               # Linux/Mac installation script
└── test_full_functionality.py  # Test suite
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'winfcntlock'**
   - This is an optional dependency of msimilarities for file locking on Windows
   - The application will work normally without it
   - Solution: Ignore this warning

2. **Memory Issues with Large Models**
   - Reduce `chunk_size` parameter (e.g., 100-150)
   - Use a smaller model (e.g., use a 6B model instead of 34B)
   - Ensure sufficient RAM available

3. **Model Download Issues**
   - Check network connection
   - Try using a different mirror (ModelScope/HuggingFace)
   - Configure proxy if needed

### Performance Tips

- Use GPU acceleration if available (install CUDA version of MindSpore)
- Reduce `similarity_top_k` and `rerank_top_k` for faster responses
- Use batch processing for multiple queries

## License

MIT License - See LICENSE file for details.

## References

- [MindSpore](https://www.mindspore.cn/)
- [MindNLP](https://github.com/mindspore-lab/mindnlp)
- [Step into LLM Course](https://github.com/mindspore-courses/step_into_llm)
- [PR #1519 - GPT Summarization Fix](https://github.com/mindspore-lab/mindnlp/pull/1519)
