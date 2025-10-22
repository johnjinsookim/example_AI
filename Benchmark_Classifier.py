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
from sklearn.metrics import accuracy_score, classification_report


# COMMAND ----------

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from pyspark.sql.functions import row_number
from pyspark.sql.window import Window

#Insert paths to data lake
classification_input_path = "" 
classification_output_path = ""

df_lables = spark.read.format("delta").load(classification_input_path).fillna("NA")
df_llm_output = spark.read.format("delta").load(classification_output_path).fillna("NA")


# COMMAND ----------

NormalizedResponseCode_Mapping = {
"3DS Authentication Required":"3DS Authentication Required",
"Abandoned":"Abandoned",
"Approved":"Approved",
"Do Not Honor":"Do Not Honor",
"Fraud/Security - 3DS Authentication Failed":"Fraud/Security",
"Fraud/Security - General":"Fraud/Security",
"Fraud/Security - Invalid CVV":"Fraud/Security",
"Fraud/Security - Invalid PIN":"Fraud/Security",
"Fraud/Security - Known":"Fraud/Security",
"Fraud/Security - Lost/Stolen":"Fraud/Security",
"Fraud/Security - MC Security Decline":"Fraud/Security",
"Insufficient Funds":"Insufficient Funds",
"Invalid Payment Instrument - Closed Account":"Invalid Payment Instrument",
"Invalid Payment Instrument - Expired":"Invalid Payment Instrument",
"Invalid Payment Instrument - General":"Invalid Payment Instrument",
"Invalid Payment Instrument - MC Lifecycle Decline":"Invalid Payment Instrument",
"Invalid Transaction - Amount Limits Exceeded":"Invalid Transaction",
"Invalid Transaction - General":"Invalid Transaction",
"Invalid Transaction - MC Policy Decline":"Invalid Transaction",
"Invalid Transaction - Restricted Card":"Invalid Transaction",
"Invalid Transaction - Transaction Not Allowed":"Invalid Transaction",
"Other - Unknown":"Other",
"Other - Uncategorized":"Other",
"System Error - Microsoft":"System Error",
"System Error - Provider":"System Error",
}

# COMMAND ----------

from itertools import chain

mapping_expr = F.create_map([lit(x) for x in chain(*NormalizedResponseCode_Mapping.items())])

df_llm_output = df_llm_output.withColumn("classification_responsecode", mapping_expr.getItem(col("classification")))
df_llm_output.display()

# COMMAND ----------

join_columns = ['ProviderName', 'PaymentNetworkName', 'ResponseCodeFromNetwork', 'ResponseCodeFromProvider']

# COMMAND ----------

df_join = df_llm_output.join(df_lables, join_columns, 'inner').fillna("Unknown")


# COMMAND ----------
"""We replaced confidence score with a scale from 1-3 since scores introduce false precision"""

# MAGIC %md
# MAGIC # precision: Of all the predictions the model made for a given class, how many were correct?
# MAGIC #            High precision = few false positives.
# MAGIC #            Example: If GPT says "Insufficient Funds" 10 times, but only 8 are correct → precision = 0.8
# MAGIC
# MAGIC # recall: Of all the actual true examples of a given class, how many did the model correctly identify?
# MAGIC #         High recall = few false negatives.
# MAGIC #         Example: If there are 12 real "Insufficient Funds" cases, and GPT catches 8 → recall = 0.67
# MAGIC
# MAGIC # f1-score: Harmonic mean of precision & recall. Balances both metrics.
# MAGIC #           F1 = 2 * (precision * recall) / (precision + recall)
# MAGIC #           Useful when dataset is imbalanced (some labels appear much more than others).
# MAGIC
# MAGIC # support: The number of actual occurrences of the class in the dataset.
# MAGIC #          Example: If there are 12 real "Insufficient Funds" examples → support = 12
# MAGIC

# COMMAND ----------

df_join_eval = df_join.filter(~col("responsecodedetails").isin('Other - Uncategorized','Other - Unknown'))
print("Accuracy:", accuracy_score(df_join_eval.select("ResponseCode").toPandas()["ResponseCode"], df_join_eval.select("classification_responsecode").toPandas()["classification_responsecode"]))
print("\nClassification Report:\n", classification_report(df_join_eval.select("ResponseCode").toPandas()["ResponseCode"], df_join_eval.select("classification_responsecode").toPandas()["classification_responsecode"]))

# COMMAND ----------

#matched data
df_join.filter(col("classification")==col("ResponseCodeDetails")).display()

# COMMAND ----------

#matched data
df_join.filter(col("classification_responsecode")==col("ResponseCode")).display()

# COMMAND ----------

#Lablel as others
df_join.filter(col("responsecodedetails").isin('Other - Uncategorized','Other - Unknown')) \
.select( 'ResponseCodeFromProvider','ResponseCodeFromNetwork',"confidence_score","classification", "ResponseCodeDetails",'classification_responsecode','responsecode','ResponseCodeFromProvider', 'PaymentNetworkName', 'ResponseCodeFromNetwork', 'ResponseCodeFromProvider',"Chain_Of_Thought","explanation").display()

# COMMAND ----------

#unmatched classification with label and which is not otheres
df_join.filter(col("classification_responsecode")!=col("ResponseCode")) \
    .filter(~col("responsecodedetails").isin('Other - Uncategorized','Other - Unknown')) \
.select( 'ResponseCodeFromProvider','ResponseCodeFromNetwork',"confidence_score","classification", "ResponseCodeDetails",'classification_responsecode','responsecode','ResponseCodeFromProvider', 'PaymentNetworkName', 'ResponseCodeFromNetwork', 'ResponseCodeFromProvider',"Chain_Of_Thought","explanation").orderBy(col("confidence_score").desc(),col("ProviderName")).display()

# COMMAND ----------

df_join = df_join.filter( ~(col("responsecodedetails").isin('Other - Uncategorized','Other - Unknown')) )
df_join.filter(col("classification_responsecode")!=col("ResponseCode")).groupBy("confidence_score").count().display()
