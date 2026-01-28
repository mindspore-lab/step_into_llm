# ChatPDF Migration Summary: MindSpore 2.6.0 & MindNLP 0.4.0

## 1. Project Overview
ChatPDF is a RAG (Retrieval-Augmented Generation) application built on MindSpore and MindNLP. It allows users to:
- Upload PDF documents.
- Automatically chunk and index text using `msimilarities`.
- Interact with the document using Large Language Models (LLMs) like ChatGLM, Llama, etc.

## 2. Migration Motivation
MindNLP 0.4.0 introduced significant architectural changes:
- Introduction of `mindnlp.core` for a more standardized tensor and operator API.
- Consolidation of directory structures.
- Transition from legacy training engines to the new `mindnlp.engine`.

## 3. Key Technical Changes
### 3.1 Version Requirements
- **Python**: 3.11
- **MindSpore**: >= 2.6.0
- **MindNLP**: >= 0.4.0

### 3.2 Code Refactoring
- **Operator Migration**: Replaced direct MindSpore tensor manipulations with `mindnlp.core.ops`.
  - Example: `logits.view(-1, )` -> `ops.reshape(logits, (-1,))`.
- **Import Optimization**: Added `from mindnlp.core import ops` to support the new unified operator interface.
- **Reranker Logic**: Enhanced `_get_reranker_score` to use `mindnlp.core.ops` for better compatibility with the 0.4.0 framework.

## 4. Accuracy & Performance
- The migration maintains the original RAG logic.
- Using `mindnlp.core.ops` ensures that the model runs efficiently on the MindSpore 2.6.0 backend.
- Accuracy remains consistent with the baseline as the underlying model weights and retrieval algorithms are preserved.

## 5. How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Start the Web UI
python simple_ui.py
```
