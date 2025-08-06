# Enhanced RAG System for Insurance Policy Processing

This is an improved version of the RAG (Retrieval-Augmented Generation) system specifically designed for processing insurance policy queries and making coverage decisions.

## Key Improvements

### 1. Enhanced Query Parsing
- **Better abbreviated query handling**: Now correctly parses queries like "46M, knee surgery, Pune, 3-month policy"
- **Indian city recognition**: Enhanced location extraction for Indian cities
- **Improved medical terminology**: Better recognition of medical procedures and conditions
- **Flexible age/gender parsing**: Handles various formats (46M, 46-year-old male, etc.)

### 2. Sophisticated Decision Logic
- **Policy rule engine**: Implements specific insurance policy rules for different procedures
- **Age-based restrictions**: Checks age limits for different procedure types
- **Waiting period validation**: Validates policy age against required waiting periods
- **Location coverage**: Verifies if treatment location is in network
- **Procedure-specific rules**: Different rules for knee, cardiac, dental, and general surgery

### 3. Enhanced Document Processing
- **Improved sample documents**: More comprehensive policy documents with detailed coverage rules
- **Better semantic understanding**: Enhanced retrieval with hybrid search (semantic + keyword)
- **Clause mapping**: Maps decisions to specific policy clauses
- **Structured response format**: Detailed JSON responses with policy compliance mapping

### 4. Better Response Format
- **Conversational responses**: Natural language responses like "Yes, knee surgery is covered under the policy"
- **Structured JSON**: Detailed response with decision, amount, justification, and clause mapping
- **Policy compliance tracking**: Shows compliance status for age, policy age, location, and procedure
- **Confidence scoring**: Indicates confidence level and recommends manual review when needed

## Usage Examples

### Sample Query
```
"46M, knee surgery, Pune, 3-month policy"
```

### Expected Response
```json
{
  "decision": "rejected",
  "amount": null,
  "confidence": 0.85,
  "justification": "Policy is 3 months old, but 6-month waiting period applies for knee_surgery",
  "query_analysis": {
    "age": 46,
    "gender": "male",
    "procedure": "knee surgery",
    "location": "Pune",
    "policy_age_months": 3
  },
  "policy_mapping": {
    "age_compliance": {"status": "compliant", "reason": "Age 46 within limits 18-65"},
    "policy_age_compliance": {"status": "non_compliant", "reason": "Policy age 3 months, waiting period 6 months"},
    "location_coverage": {"status": "covered", "reason": "Location Pune is in network"},
    "procedure_coverage": {"status": "covered", "reason": "Procedure knee surgery found in policy documents"}
  }
}
```

### Conversational Response
```
"No, knee surgery is not covered under the policy. Policy is only 3 months old and may have waiting period restrictions."
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

3. Run the setup script:
```bash
python setup.py
```

## Testing

Run the test script to see the system in action:

```bash
python test_rag.py
```

This will test various insurance queries and demonstrate the improved functionality.

## Key Features

### Query Processing
- Parses natural language queries into structured format
- Handles abbreviated formats (46M, 3-month policy)
- Extracts age, gender, procedure, location, and policy duration
- Recognizes Indian cities and medical terminology

### Decision Making
- **Approved**: Procedure is covered under policy
- **Rejected**: Procedure is explicitly excluded or doesn't meet requirements
- **Requires Review**: Insufficient information for automatic decision

### Policy Rules
- **Age Limits**: Different age restrictions for different procedures
- **Waiting Periods**: Policy age requirements (3-12 months depending on procedure)
- **Location Coverage**: Network hospital verification
- **Procedure Coverage**: Specific procedure coverage validation

### Response Format
- **Decision**: approved/rejected/requires_review
- **Amount**: Coverage amount if applicable
- **Confidence**: Confidence score (0-1)
- **Justification**: Human-readable explanation
- **Policy Mapping**: Detailed compliance status
- **Relevant Clauses**: Source clauses used for decision

## Supported Query Formats

1. **Full format**: "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
2. **Abbreviated format**: "46M, knee surgery, Pune, 3-month policy"
3. **Mixed format**: "35F, cardiac surgery, Mumbai, 18-month policy"

## Supported Procedures

- **Knee Surgery**: Age 18-65, 6-month waiting period, $50,000 coverage
- **Cardiac Surgery**: Age 18-70, 12-month waiting period, $100,000 coverage
- **Dental Surgery**: Age 18-75, 3-month waiting period, $3,000 coverage
- **General Surgery**: Age 18-80, 6-month waiting period, $25,000 coverage

## Network Coverage

Supported cities: Pune, Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata, Ahmedabad, Nagpur, Indore, Bhopal, Lucknow

## Error Handling

- Graceful handling of missing dependencies
- Fallback mechanisms for parsing failures
- Clear error messages and debugging information
- Confidence scoring for uncertain decisions

## Future Enhancements

1. **LLM Integration**: Connect to actual LLM for more sophisticated reasoning
2. **Document Upload**: Support for uploading custom policy documents
3. **Batch Processing**: Process multiple queries efficiently
4. **Audit Trail**: Track decision history and reasoning
5. **API Interface**: REST API for integration with other systems 