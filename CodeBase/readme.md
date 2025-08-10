# ADGM-Compliant Corporate Agent with Document Intelligence

## Overview

This project is a Flask-based Compliant Corporate Agent with Document Intelligence that provides intelligent document processing and compliance validation for legal workflows.

### Key Capabilities
- Upload multiple `.docx` legal documents through a user-friendly web interface
- Automatically identify document types by comparing content with predefined requirements
- Leverage RAG (Retrieval-Augmented Generation) pipeline with comprehensive knowledge base
- Detect compliance issues and missing elements in uploaded legal documents
- Deliver structured results through an intuitive professional dashboard

---

## Features

### Core Functionality
- **Multiple Document Upload** (`.docx` format support)
- **Automated Document Type Classification** using AI
- **Intelligent Content Parsing & Text Extraction**
- **RAG Integration** for document validation & contextual analysis
- **Compliance Red-Flag Detection** system

### User Interface
- **Modern Professional Dashboard**
- **Table-Based Results Presentation**
- **Enhanced Readability Interface**
- **Comprehensive Error Handling**

---

## Technology Stack

### Backend
- **Flask** - Python web framework
- **python-docx** - Document parsing library
- **Groq API** - LLM integration (`gpt-oss-20b` model) 

### Frontend
- **HTML5/CSS3** - Modern web standards
- **Jinja2** - Template engine
- **Responsive Design** - Professional theme UI

### Data Processing
- **JSON** - Configuration and knowledge base management
- **RAG Pipeline** - Retrieval-Augmented Generation
- **UUID** - Secure file handling
- **Werkzeug** - File upload security

---

## AI & Machine Learning

### Large Language Model
- **Provider**: Groq API
- **Model**: `gpt-oss-20b`
- **Purpose**: Document type identification, requirement comparison, compliance validation

### RAG Implementation
- **Knowledge Base**: `knowledgebase_docs.json`
- **Embedding-Based Retrieval** for contextual enhancement(`all-MiniLM-L6-v2`Model)
- **Intelligent Information Retrieval** before LLM querying

---

## Project Structure

```
legal-doc-classifier/
│
├── app.py                      # Main Flask application
├── requirement.json            # Document requirements for checklist verification
├── knowledgebase_docs.json     # RAG knowledge base
├── templates/
│   ├── index.html              # Document upload interface
│   └── results.html            # Results dashboard
├── static/
│   ├── style.css               # Main stylesheet
│   └── result.css              # Results page stylesheet
├── uploads/                    # Temporary document storage
├── requirement.txt             # Python dependencies
└── README.md                   # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Groq API access

### 1. Clone Repository
```bash
git clone <repository-url>
cd legal-doc-classifier
```

### 2. Install Dependencies
```bash
pip install flask python-docx groq
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Launch Application
```bash
python app.py
```

### 5. Access Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## How It Works

### Document Processing Workflow
1. **Document Upload** - Users upload `.docx` files via web interface
2. **Content Extraction** - `python-docx` parses document content
3. **Type Classification** - Content compared against `requirement.json`
4. **RAG Enhancement** - Knowledge base provides contextual information
5. **AI Analysis** - Groq LLM validates and flags compliance issues
6. **Results Display** - Structured output in professional dashboard

---
## Watch Demo Video

[![Video Demo](https://img.shields.io/badge/📹_Video_Demo-Watch_Here-red?style=for-the-badge)](https://drive.google.com/file/d/1TyhQ_YQgfnFXtL-sBn6UVdyt5VkCa8Qq/view)

---

### Configuration Files
- **`requirement.json`** - Defines required document types and descriptions
- **`knowledgebase_docs.json`** - Stores legal knowledge for RAG pipeline

---

## Author

**Shweta Nagapure**  
SVKM Institute of Technology | Dhule, India

Specialized in data-driven decision systems, AI-assisted tools, and Flask web applications.

### Contact Information
- **Email**: shwetanagapure@gmail.com
- **LinkedIn**: www.linkedin.com/in/shweta-nagapure

---

## Documentation

For comprehensive project documentation, visit:
**[Project Documentation](https://docs.google.com/document/d/1YnwdUrAfk2Vwh6Isafu5kc3te4MyWmOV4-2t74EFRe8/edit?usp=sharing)**

---
**Important Note**

```This project was developed as a demonstration task. The configuration files (requirement.json and knowledgebase_docs.json) contain limited information for proof-of-concept purposes. For production deployment and optimal results, these files should be expanded with comprehensive legal document requirements and extensive knowledge base content specific to your organization's compliance needs.```


---
*Professional legal document intelligence solution for corporate compliance*






