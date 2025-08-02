# Advanced RAG System for Insurance Policy Analysis

A sophisticated Retrieval-Augmented Generation (RAG) system designed to analyze insurance policies and provide accurate coverage decisions based on document content.

## Features

- **Multi-Model Embedding System**: Uses multiple sentence transformer models for robust document retrieval
- **Large PDF Support**: Handles PDFs of any size (49+ pages) with efficient memory management
- **Dynamic Content Analysis**: Extracts coverage rules and conditions from actual document content
- **Conversational Responses**: Provides natural language answers to insurance queries
- **Comprehensive Query Parsing**: Extracts age, gender, procedure, location, and policy information
- **Batch Processing**: Can process multiple queries efficiently

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced functionality, install additional packages:

```bash
# For better PDF processing
pip install PyMuPDF

# For document processing
pip install python-docx

# For NLP capabilities
pip install spacy
python -m spacy download en_core_web_sm

# For FAISS vector search
pip install faiss-cpu
```

## Usage

### Basic Usage

```bash
# Test with sample documents
python rag.py

# Test with your own PDF document
python rag.py /path/to/your/insurance_policy.pdf
```

### Interactive Mode

The system provides an interactive interface where you can:

1. Load your insurance policy document
2. Ask questions about coverage
3. Get conversational responses with confidence scores

Example queries:
- "46M, knee surgery, Pune, 3-month policy"
- "25F, dental treatment, Mumbai, 2-year policy"
- "60 year old woman, heart surgery, Delhi, 6 month old policy"

### Programmatic Usage

```python
from rag import AdvancedRAGSystem

# Initialize the system
rag_system = AdvancedRAGSystem()

# Load documents
rag_system.load_documents(['policy_document.pdf'])

# Process a query
result = rag_system.process_query("46M, knee surgery, Pune, 3-month policy")
print(result['decision'])  # 'approved', 'rejected', or 'requires_review'
```

## System Architecture

### Components

1. **AdvancedQueryParser**: Extracts structured information from natural language queries
2. **HybridEmbeddingRetriever**: Uses multiple embedding models for document retrieval
3. **LLMDecisionEngine**: Analyzes document content and makes coverage decisions
4. **AdvancedRAGSystem**: Orchestrates all components

### Document Processing

- **PDF Support**: Full PDF processing with page-by-page analysis
- **Text Chunking**: Intelligent document splitting for better retrieval
- **Memory Optimization**: Batch processing for large documents
- **Progress Tracking**: Real-time progress indicators for large files

### Decision Making

The system uses a multi-step approach:

1. **Query Analysis**: Extracts age, gender, procedure, location, policy duration
2. **Document Retrieval**: Finds relevant content using semantic search
3. **Content Analysis**: Identifies coverage patterns and conditions
4. **Decision Logic**: Applies rules based on document content
5. **Confidence Scoring**: Provides confidence levels for decisions

## Configuration

### Embedding Models

The system uses multiple sentence transformer models:

- `all-MiniLM-L6-v2`: Fast and efficient
- `all-mpnet-base-v2`: High quality (optional)
- `multi-qa-MiniLM-L6-cos-v1`: Question-answering optimized (optional)

### Memory Management

For large documents, the system:
- Processes documents in batches
- Uses CPU-based processing to avoid GPU memory issues
- Implements progress tracking for transparency

## File Structure

```
BAJAJ/
├── rag.py                 # Main RAG system implementation
├── requirements.txt       # Python dependencies
├── README.md            # This file
└── setup.py             # Package setup (optional)
```

## Testing

### Sample Documents

The system includes sample insurance policy documents for testing:

- General policy terms and conditions
- Dental coverage policy
- Cardiac care coverage

### Debug Mode

Enable debug mode to see detailed processing information:

```python
result = rag_system.process_query(query, debug=True)
```

This will show:
- Retrieved document chunks
- Coverage indicators found
- Decision reasoning process
- Confidence calculations

## Performance

### Large Document Handling

- **Memory Efficient**: Processes documents in chunks to avoid memory issues
- **Scalable**: Handles PDFs with 100+ pages
- **Fast**: Uses optimized embedding models and FAISS indexing

### Accuracy Improvements

- **Content-Aware**: Makes decisions based on actual document content
- **Pattern Recognition**: Identifies coverage patterns and exclusions
- **Context Analysis**: Considers policy age, waiting periods, and conditions

## Troubleshooting

### Common Issues

1. **Memory Errors**: Use CPU processing and smaller batch sizes
2. **Missing Dependencies**: Install required packages from requirements.txt
3. **PDF Extraction Issues**: Ensure PyMuPDF is installed for PDF processing
4. **Low Accuracy**: Check if document contains relevant coverage information

### Debug Information

The system provides comprehensive debug output to help identify issues:

- Document loading progress
- Embedding creation status
- Retrieved content preview
- Decision reasoning chain

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review debug output
3. Open an issue on GitHub

## Roadmap

- [ ] Integration with external LLM APIs
- [ ] Support for more document formats
- [ ] Enhanced decision logic
- [ ] Web interface
- [ ] API endpoints 