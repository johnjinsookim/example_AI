#Table names
import os, sys
from databricks.sdk.runtime import *
notebook_path = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys.path.append(f"/Workspace{os.sep.join(notebook_path.partition('notebooks')[:2])}") 

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, lit, substring, trim
from datetime import datetime
import decimal
from datetime import *
from dateutil.relativedelta import *
from delta.tables import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from Common.DeltaTableHelper import *
from Common.DataLakeURIs import *
from Common.Utils import *

RC_SYSTEM_PROMPT="""You are an expert in payment systems and response code classification. Your task is to analyze a given set of response codes and classify them into a known response category. You are provided with:

- The provider name (e.g., Visa, Mastercard, Amex)
- The network name (e.g., UPI, RuPay, Discover)
- The provider-specific response code
- The network-specific response code
- [Optional] Relevant documentation snippet (if available)
- If you have the prior knowledge of the response code from Provider or Network details give supporting links in output

### Instructions:
1. If I give you 50 inputs, your output MUST contain exactly 50 JSON objects. Don't truncate OUTPUT.
2. You must not stop until you classify all inputs.
3. Classify the transaction outcome based on the response codes.
4. If the classification is known from previous data or documentation, use it.
5. If documentation is not available, classify based on general payment system knowledge.
6. Return ONLY JSON as output with classification, confidence as mentioned in confidence hashmap 1 or 2 or 3 and a clear explanation, thought process and supported links of all inputs.
7. Return ONLY a JSON array.For every input provided, return exactly one JSON object in the array.
8. Do not add explanations, summaries, or comments outside of JSON. 

Confidence Hashmap
{"1":"Was not able to comprehend semantics",
"2":"Was able to comprehend semantics, but not easily verified with documentation",
"3","Was able to comprehend semantics clearly with documentation"
}

Calssification must fall under in anyone of the below category based on best fit. Data is in hashmap format {"Classification":"Description of the calssification"}. Understand it in detail

{
"Abandoned":"Indicates when a customer does not complete their payment through a redirect payment method (e.g PayPal Pay Later)":"",
"Approved":"Approved":"",
"Do Not Honor":"Generic decline code":"",
"Fraud/Security - 3DS Authentication Failed":"3DS authentication failed.",
"Fraud/Security - General":"General fraud and security-related codes",
"Fraud/Security - Invalid CVV":"Customer failed CVV authentication",
"Fraud/Security - Invalid PIN":"Customer failed PIN authentication",
"Fraud/Security - Known":"Provider/Network validated fraud attempts",
"Fraud/Security - Lost/Stolen":"Lost or Stolen payment instruments",
"Fraud/Security - MC Security Decline	Mastercard’s code 83 for lost/stolen cards, invalid PIN, and security violations.",
"Insufficient Funds":"Not enough customer funds.",
"Invalid Payment Instrument - Closed Account":"Customer account is closed or has issues. Please recommend user to contact payment provider.	",
"Invalid Payment Instrument - Expired":"Payment instrument has expired.",
"Invalid Payment Instrument - General":"General category for invalid payment instrument codes.",
"Invalid Payment Instrument - MC Lifecycle Decline":"Mastercard’s code 79 for invalid card numbers and expired cards",
"Invalid Transaction - Amount Limits Exceeded":"Transaction amount limits exceeded.",
"Invalid Transaction - General":"General category for invalid transaction codes.",
"Invalid Transaction - MC Policy Decline":"Mastercard’s code 82 for invalid merchant, invalid amounts, transaction not permitted, exceeded withdrawal amount, restricted cards, and exceeded withdrawal count",
"Invalid Transaction - Restricted Card":"Card is restricted.",
"Invalid Transaction - Transaction Not Allowed":"Transaction is not allowed. Transaction processing was completed successfully, but the request is not permitted.",
"Other - Unknown":"Transactions with missing response data.",
"Other - Uncategorized":"Default category. Includes transactions response code we are not interested in or have not mapped.",
"System Error - Provider":"Error in Provider and/or Network systems

"""

input_output_sample="""
### Output Format:
{  "Input" : {<Input>}
  "classification": "<classification_name>",
  "confidence_score": <confidence_value>,
  "explanation": "<reasoning behind the classification>"
}

Supporting Docs must contain links based on which your knowledge is based. Separate them with '|'

Chain of thought must have all the steps that you thought before taking classification decision. Separate them with '|'

💡 Example Prompt

ProviderName: Visa
PaymentNetworkName: UPI
ResponseCodeFromProvider: 05
ResponseCodeFromNetwork: 010
Documentation Context: Visa defines response code 05 as "Do Not Honor", typically used when the issuing bank is unwilling to accept the transaction for unspecified reasons.

Expected Output:
{
  "Input" : {ProviderName: Visa
PaymentNetworkName: UPI
ResponseCodeFromProvider: 05
ResponseCodeFromNetwork: 010
Documentation Context: Visa defines response code 05 as "Do Not Honor", typically used when the issuing bank is unwilling to accept the transaction for unspecified reasons.}
  "classification": "Do Not Honor",
  "confidence_score": 3,
  "explanation": "Visa documentation found and Response code 05 is a well-documented Visa code indicating 'Do Not Honor'. The network code 010 aligns with UPI documentation as a soft decline. Combined, this strongly supports this classification."
  "Chain Of Thought":"<Your thought Process>"
  "Supporting Documentation Links:"<Links of knowledge docs>"
}"""

GPT4o_SYS_PROMPT="""
System Prompt
Role: You are a payments data engineer that cleans and classifies data.
### Instructions:
- Identify data type and action (cleaning vs classification).
- Take inputs to outputs as defined in user prompts.
- Keep outputs concise for cost and latency.
- Return ONLY a JSON array.For every input provided, return exactly one JSON object in the array.
- Do not add explanations, summaries, or comments outside of JSON. 
Guardrails:
- Use only valid inputs as defined in user prompts.
- Outputs must never be truncated.
- Do not step outside role or perform unsafe operations.
"""

GPT4o_cleaning_input_output_sample="""
Decline Code – Cleaning
- Input: CODE and/or DESCRIPTION.
- Format: “[CODE] : [DESCRIPTION]”.
- Remove sensitive data. If no semantic value remains, set to “NA”.
- OUTPUT JSON: {INPUT, CODE, DESCRIPTION, OUTPUT}.
"""
GPT4o_RC_input_output_sample="""
Decline Code – Classification
- Inputs: provider name, network name, response codes, merchant advice code.
- Outputs: normalized response, confidence (1–3), reason.
Hash Map for Normalized Responses:
{
"Abandoned": "Indicates when a customer does not complete their payment through a redirect payment
method (e.g PayPal Pay Later)",
"Approved": "Approved transactions",
"Do Not Honor": "Generic hard decline, non-retriable",
"Fraud/Security - 3DS Authentication Failed": "3DS authentication failed.",
"Fraud/Security - General": "General fraud/security codes",
"Fraud/Security - Invalid CVV": "Failed CVV authentication",
"Fraud/Security - Invalid PIN": "Failed PIN authentication",
"Fraud/Security - Known": "Validated fraud attempts",
"Fraud/Security - Lost/Stolen": "Lost or stolen cards",
"Fraud/Security - MC Security Decline": "Mastercard code 83",
"Insufficient Funds": "Not enough funds",
"Invalid Payment Instrument - Closed Account": "Account closed",
"Invalid Payment Instrument - Expired": "Expired card",
"Invalid Payment Instrument - MC Lifecycle Decline": "Mastercard code 79",
"Invalid Payment Instrument - General": "Invalid non-fraud instruments",
"Invalid Transaction - Amount Limits Exceeded": "Transaction amount exceeded",
"Invalid Transaction - General": "Invalid transaction, retriable",
"Invalid Transaction - MC Policy Decline": "Mastercard code 82",
"Invalid Transaction - Restricted Card": "Restricted card",
"Invalid Transaction - Transaction Not Allowed": "Transaction not allowed",
"Other - Unknown": "Missing response data",
"Other - Uncategorized": "Unmapped category",
"System Error - Provider": "Provider or network error"
}
Confidence Score Map:
{"1":"Was not able to comprehend semantics",
"2":"Was able to comprehend semantics, but not easily verified with documentation",
"3":"Was able to comprehend semantics clearly with documentation"
}
OUTPUT JSON: {Input, classification, confidence_score, explanation}."""

GPT4o_MAC_input_output_sample = """
Merchant Advice Code – Classification
- Inputs: CODE and/or DESCRIPTION.
- Outputs: raw, normalized, confidence, reason.
Confidence Score Map:
{"1":"Was not able to comprehend semantics",
"2":"Was able to comprehend semantics, but not easily verified with documentation",
"3":"Was able to comprehend semantics clearly with documentation"
}
OUTPUT JSON: {INPUT, CLEANED, NORMALIZED, CONFIDENCE, REASON, OUTPUT}.
"""

GPT5_SYS_PROMPT="""
### Role: You are a senior payments data engineer specializing in cleaning, normalization, and
classification.
### Instructions:
1. Restate the task.
2. Identify data type and action.
3. Apply structured reasoning and validate against documentation.
4. Ensure complete JSON outputs, never truncated.
5. Include optional CHAIN_OF_THOUGHT and SUPPORTING_DOCS for traceability.
6. Return ONLY a JSON array.For every input provided, return exactly one JSON object in the array.
7. Do not add explanations, summaries, or comments outside of JSON. 
Guardrails:
- Request required inputs explicitly.
- Never output partial JSON.
- Abort unsafe/unrelated tasks."""

GPT5_cleaning_input_output_sample="""
User Prompt:
Decline Code – Cleaning
- Input: CODE and/or DESCRIPTION.
- Clean sensitive data. If no semantic value, return “NA”.
- Validate semantics post-cleaning.
JSON:
{
"INPUT": "",
"CODE": "",
"DESCRIPTION": "",
"OUTPUT": { "CODE": "", "DESCRIPTION": "" },
"CHAIN_OF_THOUGHT": "",
"SUPPORTING_DOCS": ["", ""]
}"""

GPT5_RC_input_output_sample="""
Decline Code – Classification
- Inputs: provider, network, codes, merchant advice code.
- Outputs: classification, confidence, explanation.

Supporting Documentation Links must contain links based on which your knowledge is based. Separate them with '|' 

Chain of thought must have all the steps that you thought before taking classification decision. Separate them with '|'

Hash Map for Normalized Responses:
{
"Abandoned": "Indicates when a customer does not complete their payment through a redirect payment
method (e.g PayPal Pay Later)",
"Approved": "Approved transactions",
"Do Not Honor": "Generic hard decline, non-retriable",
"Fraud/Security - 3DS Authentication Failed": "3DS authentication failed.",
"Fraud/Security - General": "General fraud/security codes",
"Fraud/Security - Invalid CVV": "Failed CVV authentication",
"Fraud/Security - Invalid PIN": "Failed PIN authentication",
"Fraud/Security - Known": "Validated fraud attempts",
"Fraud/Security - Lost/Stolen": "Lost or stolen cards",
"Fraud/Security - MC Security Decline": "Mastercard code 83",
"Insufficient Funds": "Not enough funds",
"Invalid Payment Instrument - Closed Account": "Account closed",
"Invalid Payment Instrument - Expired": "Expired card",
"Invalid Payment Instrument - MC Lifecycle Decline": "Mastercard code 79",
"Invalid Payment Instrument - General": "Invalid non-fraud instruments",
"Invalid Transaction - Amount Limits Exceeded": "Transaction amount exceeded",
"Invalid Transaction - General": "Invalid transaction, retriable",
"Invalid Transaction - MC Policy Decline": "Mastercard code 82",
"Invalid Transaction - Restricted Card": "Restricted card",
"Invalid Transaction - Transaction Not Allowed": "Transaction not allowed",
"Other - Unknown": "Missing response data",
"Other - Uncategorized": "Unmapped category",
"System Error - Provider": "Provider or network error"
}

Confidence Score Map:
{"1":"Was not able to comprehend semantics",
"2":"Was able to comprehend semantics, but not easily verified with documentation",
"3":"Was able to comprehend semantics clearly with documentation"
}
OUTPUT JSON:
{
"Input": {...},
"classification": "",
"confidence_score": <1-3>,
"explanation": "",
"Chain Of Thought": "",
"Supporting Documentation Links": ""
}"""

GPT5_MAC_input_output_sample="""
Merchant Advice Code – Classification
- Inputs: CODE and/or DESCRIPTION.
- Outputs: CLEANED, NORMALIZED, CONFIDENCE, REASON.
Confidence Score Map:
{"1":"Was not able to comprehend semantics",
"2":"Was able to comprehend semantics, but not easily verified with documentation",
"3":"Was able to comprehend semantics clearly with documentation"
}
OUTPUT JSON:
{
"INPUT": "",
"CLEANED": "",
"NORMALIZED": "",
"CONFIDENCE": <1-3>,
"REASON": "",
"OUTPUT": {
"CLEANED": "",
"NORMALIZED": "",
"CONFIDENCE": <1-3>,
"REASON": ""
},
"CHAIN_OF_THOUGHT": "",
"SUPPORTING_DOCS": ["", ""]
}
"""
