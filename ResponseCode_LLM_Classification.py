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

# Retrieve widget values
model = dbutils.widgets.get("model") #2 and above
itration = int(dbutils.widgets.get("iteration"))
itration = itration
print(model,itration)


# COMMAND ----------

import json
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DecimalType, DateType


# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DecimalType, DateType

schema = StructType([
    StructField("itration", IntegerType(), True),
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
elif model == "GPT5": #pdp-test-ai-service-9
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

print(SYSTEM_PROMPT,input_output_sample)

# COMMAND ----------

# MAGIC %md
# MAGIC ####llm_input_df will be the input to the model. This frame will be chuncked such that 10 records will be included as input into the model
# MAGIC Put filters on this dataframe to test with sample data

# COMMAND ----------

from pyspark.sql.functions import col

# Load input and output dataframes
input_df = spark.read.format("delta").load("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_input").fillna("Unknown")
output_df = spark.read.format("delta").load("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output").fillna("Unknown").filter(col("iteration") == lit(itration)  )

# Select the required columns
input_cols = ["PaymentNetworkName", "ProviderName", "ResponseCodeFromNetwork", "ResponseCodeFromProvider"]
output_cols = ["PaymentNetworkName", "ProviderName", "ResponseCodeFromNetwork", "ResponseCodeFromProvider"]

# Perform anti-join to find rows in input_df not present in output_df
llm_input_df = input_df.select(input_cols).subtract(output_df.select(output_cols)).orderBy( lower(col("ProviderName")).desc() ,col("ResponseCodeFromProvider").desc()).filter(col("ProviderName") != lit("Unknown")  ).filter(col("ResponseCodeFromProvider") != lit("NA")  ).filter(col("PaymentNetworkName") != lit("Unknown")).limit(10)


display(llm_input_df)

# COMMAND ----------

llm_input_df.count()

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

# MAGIC %sql
# MAGIC
# MAGIC --select  ProviderName,PaymentNetworkName,ResponseCodeFromProvider,ResponseCodeFromNetwork,count(1) from delta.`abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_input` group by all having count(1)>1

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC --select  ProviderName,PaymentNetworkName,ResponseCodeFromProvider,ResponseCodeFromNetwork,ResponseCodeDetails from delta.`abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_input`  where ProviderName='BillDesk' and PaymentNetworkName='MASTERCARD' and ResponseCodeFromProvider='TRPPE0008:Internal error'
# MAGIC

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
    display(df)
    return df

# COMMAND ----------

#if dbutils.fs.ls("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output"):
    #dbutils.fs.rm("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output", recurse=True)

#if dbutils.fs.ls("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output_metadata"):
    #dbutils.fs.rm("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output_metadata", recurse=True)

# COMMAND ----------

# Convert llm_input_df to JSON batches
json_batches = df_to_json_batches(llm_input_df)

for i in range(0, len(json_batches)):
    input_data=str(json_batches[i]).replace("['","").replace("']","").replace("', '",",\n").replace("']","").replace('","','",\n"')
    input_data="""### Input: \n{\n"""+input_data+"}\n"
    user_message=input_data+input_output_sample
    completion=call_Classfier(SYSTEM_PROMPT,user_message,model)
    #print(input_data)
    metadata = {
        "itration":itration,
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
    df = parse_ClassificationOutput(clean_output).withColumn("itration",lit(itration))
    keys=metadata.keys()
    values=metadata.values()
    metadata_df = spark.createDataFrame([metadata], schema)
    #df.write.format("delta").mode("append").save("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output")
    #metadata_df.write.format("delta").mode("append").save("abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output_metadata")


# COMMAND ----------

print(clean_output)

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC
# MAGIC --select * from delta.`abfss://main@paydatalaketest.dfs.core.windows.net/silver/DataClassification/classification_output` where providername='Worldpay' limit 1--and PaymentNetworkName='VISA' 

# COMMAND ----------

metadata_df.display()

# COMMAND ----------

df.display()
