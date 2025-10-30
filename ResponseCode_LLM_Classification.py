# Databricks notebook source
import os, sys
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
from pyspark.sql import Window
from pyspark.sql.types import *
from Common.DeltaTableHelper import *
from Common.DataLakeURIs import *
from Common.Utils import *
from pyspark.sql.functions import col, split
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, date_format
from delta.tables import *
from Common.Telemetry import *
from Common.DeltaTableHelper import *
from Prompts import *

# COMMAND ----------
!pip install openai
!pip install azure.identity

# COMMAND ----------
# Databricks widgets for model selection and iteration input
dbutils.widgets.dropdown("model", "GPT5", ["GPT5", "GPT4o"], "Select Model")
dbutils.widgets.text("iteration", "2", "Iteration (int)")
dbutils.widgets.text("batch_size", "10", "Iteration (int)")

# Retrieve widget values
model = dbutils.widgets.get("model") #2 and above
itn = int(dbutils.widgets.get("iteration"))
batch_size = int(dbutils.widgets.get("batch_size"))

itn = itn
print(model,itn,batch_size)


# COMMAND ----------
import json
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DecimalType, DateType


# COMMAND ----------
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DecimalType, DateType

schema = StructType([
    StructField("iteration", IntegerType(), True),
    StructField("id", StringType(), True),
    StructField("model", StringType(), True),
    StructField("created", IntegerType(), True),
    StructField("usage_prompt_tokens", IntegerType(), True),
    StructField("usage_completion_tokens", IntegerType(), True),
    StructField("usage_total_tokens", IntegerType(), True),
    StructField("finish_reason", StringType(), True),
    StructField("date", DateType(), True),
    StructField("content", StringType(), True),
    StructField("system_prompt", StringType(), True),
    StructField("user_message", StringType(), True),
    StructField("cost_input_usd", DecimalType(10, 5), True),
    StructField("cost_output_usd", DecimalType(10, 5), True),
    StructField("total_cost_usd", DecimalType(10, 5), True)
])

# COMMAND ----------
import os
import base64
from openai import AzureOpenAI

if model == "GPT4o":
    endpoint = os.getenv("ENDPOINT_URL", "https://pdpdataclassifier.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview")
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", dbutils.secrets.get(scope='key-vault-secret', key='AZUREOPENAIAPIKEYGPT4o'))
    # Initialize Azure OpenAI client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

    SYSTEM_PROMPT=GPT4o_SYS_PROMPT #system prompt
    input_output_sample=GPT4o_RC_input_output_sample #user prompts
elif model == "GPT5":
    endpoint = os.getenv("ENDPOINT_URL", "https://pdp-test-ai-service-9.openai.azure.com/")
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY",dbutils.secrets.get(scope='key-vault-secret', key='AZUREOPENAIAPIKEYGPT5'))
    client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",)
    SYSTEM_PROMPT=GPT5_SYS_PROMPT #system prompt
    input_output_sample=GPT5_RC_input_output_sample #user prompts

# COMMAND ----------
# Llm_input_df will be the input to the model. This frame will be chuncked such that 10 records will be included as input into the model
# Put filters on this dataframe to test with sample data

from pyspark.sql.functions import col

# Load input and output dataframes
input_df = spark.read.format("delta").load("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_input").fillna("Unknown")
output_df = spark.read.format("delta").load("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output").fillna("Unknown").filter(col("iteration") == lit(itration)  )

# Select the required columns
input_cols = ["PaymentNetworkName", "ProviderName", "ResponseCodeFromNetwork", "ResponseCodeFromProvider"]
output_cols = ["PaymentNetworkName", "ProviderName", "ResponseCodeFromNetwork", "ResponseCodeFromProvider"]

# Perform anti-join to find rows in input_df not present in output_df
## Change filters to change LLM inputs
llm_input_df = input_df.select(input_cols).subtract(output_df.select(output_cols))
    .orderBy( lower(col("ProviderName")).desc() 
    ,col("ResponseCodeFromProvider").desc())
    .filter(col("ProviderName") != lit("Unknown"))
    .filter(col("ResponseCodeFromProvider") != lit("NA"))
    .filter(col("PaymentNetworkName") != lit("Unknown"))
    .filter(col("ProviderName")==lit("PayU")).limit(10)

print(SYSTEM_PROMPT,input_output_sample)
display(llm_input_df)


# COMMAND ----------

import json

# Function to convert DataFrame to JSON in batches of 50 rows
def df_to_json_batches(df, batch_size=10):
    json_batches = []
    total_rows = df.count()
    for start in range(0, total_rows, batch_size):
        batch_df = df.offset(start).limit(batch_size)
        batch_json = batch_df.toJSON().collect()
        json_batches.append(batch_json)
    return json_batches

# COMMAND ----------

def call_Classfier(SYSTEM_PROMPT,user_message,model):
    
    if model == "GPT5":
        completion = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=128000,
            stop=None,
            stream=False
        )
    elif model=="GPT4o":
        completion = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=16384,#max_tokens=16384,
            temperature=0,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )      
    
    return completion


# COMMAND ----------
# Compute cost of token usage
def compute_gpt4o_cost(prompt_tokens, completion_tokens):
    # Pricing (per 1K tokens)
    input_rate = 0.0025  # $ per 1K input tokens
    output_rate = 0.01   # $ per 1K output tokens

    cost_input = (prompt_tokens / 1000) * input_rate
    cost_output = (completion_tokens / 1000) * output_rate
    total_cost = cost_input + cost_output

    return {
        "cost_input_usd": decimal.Decimal(np.round(cost_input, 4)),
        "cost_output_usd": decimal.Decimal(np.round(cost_output, 4)),
        "total_cost_usd": decimal.Decimal(np.round(total_cost, 4))
    }


# COMMAND ----------
def clean_ClassificationOutput(raw_output):
    return re.sub(r"json|```", "", raw_output).strip()


# COMMAND ----------
def parse_ClassificationOutput(str):
    data = json.loads(clean_output)
    records = []
    for item in data:
        flat = {
        "ProviderName": item["Input"].get("ProviderName", ""),
        "PaymentNetworkName": item["Input"].get("PaymentNetworkName", ""),
        "ResponseCodeFromProvider": item["Input"].get("ResponseCodeFromProvider", ""),
        "ResponseCodeFromNetwork": item["Input"].get("ResponseCodeFromNetwork", ""),
        "classification": item.get("classification", ""),
        "confidence_score": item.get("confidence_score", None),
        "explanation": item.get("explanation", ""),
        "Chain_Of_Thought": item.get("Chain Of Thought", ""),
        "Supporting_Documentation_Links": item.get("Supporting Documentation Links", ""),
        "id":metadata["id"]
        }
        records.append(flat)

    df = spark.createDataFrame(records)
    #display(df)
    return df

# COMMAND ----------

#if dbutils.fs.ls("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output"):
    #dbutils.fs.rm("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output", recurse=True)

#if dbutils.fs.ls("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output_metadata"):
    #dbutils.fs.rm("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output_metadata", recurse=True)

# COMMAND ----------
# Convert llm_input_df to JSON batches
json_batches = df_to_json_batches(llm_input_df,batch_size)

for i in range(0, len(json_batches)):
    input_data=str(json_batches[i]).replace("['","").replace("']","").replace("', '",",\n").replace("']","").replace('","','",\n"')
    input_data="""### Input: \n{\n"""+input_data+"}\n"
    user_message=input_data+input_output_sample
    completion=call_Classfier(SYSTEM_PROMPT,user_message,model)
    #print(input_data)
    metadata = {
        "iteration":itration,
        "id": completion.id,
        "model": completion.model,
        "created": completion.created,
        "usage_prompt_tokens": completion.usage.prompt_tokens,
        "usage_completion_tokens": completion.usage.completion_tokens,
        "usage_total_tokens": completion.usage.total_tokens,
        "finish_reason": completion.choices[0].finish_reason, # stop : no action required. length: more tokens required
        "date":date.today(),
        "content":completion.choices[0].message.content,
        "system_prompt":SYSTEM_PROMPT,
        "user_message":user_message
    }
    print(metadata["finish_reason"])
    # Example usage with API response
    usage = completion.usage
    cost = compute_gpt4o_cost(usage.prompt_tokens, usage.completion_tokens)
    #print(cost)
    metadata.update(cost)
    raw_output = completion.choices[0].message.content
    clean_output = clean_ClassificationOutput(raw_output)
    #print(clean_output)
    try:
        df = parse_ClassificationOutput(clean_output).withColumn("iteration",lit(itration))
    except:
        print("Error parsing output")
        print(clean_output)
        continue
    keys=metadata.keys()
    values=metadata.values()
    metadata_df = spark.createDataFrame([metadata], schema)
    metadata_df.write.format("delta").mode("append").save("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output_metadata")
    if df.count()==batch_size:
        df.write.format("delta").mode("append").save("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output")
    else:
        print("Model Output truncated truncated the Input!!!!")
        exit(1)

print(clean_output) 
df.display()
