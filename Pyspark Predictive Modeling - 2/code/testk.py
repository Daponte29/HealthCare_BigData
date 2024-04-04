

#EVENT STATISTICS WITH PYSPARK SESSION
import pyspark
import time
import pip
import pandas as pd
import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import datediff, to_date, max as max_, lit, col, collect_list, row_number, concat_ws, format_number, concat, monotonically_increasing_id
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

path1 = r"C:\Users\nolot\OneDrive\Desktop\Big Data Healthcare\HW2\hw2-2\hw2\data\events.csv"
schema1 = StructType([
    StructField("patientid", IntegerType(), True),
    StructField("eventid", StringType(), True),
    StructField("eventdesc", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("value", FloatType(), True)])


df = spark.read.csv(path1, header=False, schema=schema1)
