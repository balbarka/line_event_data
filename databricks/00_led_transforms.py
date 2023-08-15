# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Line event data observations
# MAGIC
# MAGIC When pulling raw data from a PLC you will have at a minimum the following fields
# MAGIC  - time as timestamp
# MAGIC  - tag as string
# MAGIC  - value as double
# MAGIC
# MAGIC Let's create some dummy data for two well behaved PLCs so we can see what transforms do:

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime

# Set the random seed for reproducibility
np.random.seed(42)

# Create a DataFrame
m1_data = {'t': pd.date_range(start=datetime.datetime(2020, 1, 1, 12, 0, 0), periods=100, freq='30S') +
                pd.to_timedelta(np.random.normal(4, 1.5, 100), unit='s'),
                'tag': ['m1'] * 100,
                'value': np.sort(np.random.uniform(0, 20, 100))}
m2_data = {'t': pd.date_range(start=datetime.datetime(2020, 1, 1, 12, 0, 0), periods=200, freq='15S') +
                pd.to_timedelta(np.random.normal(4, 2.5, 200), unit='s'),
                'tag': ['m2'] * 200,
                'value': np.sort(np.random.uniform(0, 10, 200))}

raw_plc_pdf = pd.concat([pd.DataFrame(m1_data), pd.DataFrame(m2_data)])

# Display the DataFrame
display(raw_plc_pdf)

# COMMAND ----------

# We can now save this data as a spark dataframe
spark.sql("CREATE DATABASE IF NOT EXISTS main.led")
raw_plc_sdf = spark.createDataFrame(raw_plc_pdf)
raw_plc_sdf.write.mode('overwrite').saveAsTable('main.led.raw_plc')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC An assumption for some time series analyses is there is a fix time series interval for each observation. We have two intervals of 15 and 30 seconds in our data. A reasonable binning for this would be a 15 second interval. With this new interval, we'll also want to create some simple aggregates of the observations in the interval.
# MAGIC
# MAGIC **NOTE**: Since these aggregates are relatively expensive, we'll likely want to materialize sometime after these aggregates, but before iterative modeling tasks. This is highly dependent upon use case and data engineering strategy. In the example here, we'll materialize.

# COMMAND ----------

from pyspark.sql.functions import ceil, col, cast, date_trunc, array_sort
from pyspark.sql import window as W
from pyspark.sql.functions import count, struct, min, max, collect_set, avg
from pyspark.sql.types import StructField

plc_bin_sdf = raw_plc_sdf.withColumn("t_15", 
                                     (ceil((date_trunc('second', col('t') + 
                                            expr('INTERVAL 999 millisecond'))).cast("long") / 15) * 15).cast("timestamp")) \
                             .groupBy(['tag', 't_15']) \
                             .agg(struct(count("value").alias("cnt"),
                                         avg("value").alias("avg"),
                                         min("value").alias("min"),
                                         max("value").alias("max")).alias('vals'),
                                  array_sort(collect_set(struct(col('t'),
                                                     col('value')))).alias('obs'))
plc_bin_sdf.write.mode('overwrite').saveAsTable('main.led.plc_bin')
display(plc_bin_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imputation
# MAGIC
# MAGIC A lot of libraries will handle missing observations for you. But in case they don't, it is sometimes useful to do your own imputation (the process of filling in missing or incomplete values in a dataset with estimated or predicted values based on the available data) in spark sql.
# MAGIC
# MAGIC In this example, we'll do a simple average of observation before and observation after, but there are many other options available.
# MAGIC
# MAGIC **NOTE**: in many cases you will want to use obs instead of vals and not assign the same mean to all missing values, but we are using vals and simple average to make the example more succint.

# COMMAND ----------

from pyspark.sql.functions import explode, array, lit

# create range of timestamps with 15s interval
timestamps = pd.date_range(start='2020-01-01 12:00:00', periods=200, freq='15s')

# create a spark dataframe with a single timestamp column
t_range = spark.createDataFrame(
            pd.DataFrame(data=timestamps, columns=["t_15"])) \
            .withColumn('tag', explode(array(lit('m1'), lit('m2'))))

display(t_range)

# COMMAND ----------

from pyspark.sql.functions import lead, lag, col, first, last, element_at, unix_timestamp, when
from pyspark.sql.window import Window

# Define a window specification
w_lag = Window.partitionBy("tag").orderBy(col("t_15")).rowsBetween(1, Window.unboundedFollowing)
w_lead = Window.partitionBy("tag").orderBy(col("t_15")).rowsBetween(Window.unboundedPreceding, -1)

# Add a lead and lag columns for use in missing vals
# Compute missing vals
plc_bin_impute = t_range.join(plc_bin_sdf, ['t_15', 'tag'], 'left') \
                      .withColumn("lead_obs", element_at(last(col("obs"), ignorenulls=True).over(w_lead), -1)) \
                      .withColumn("lag_obs", element_at(first(col("obs"), ignorenulls=True).over(w_lag), 1)) \
                      .select('t_15',
                              'tag',
                              when(col('vals').isNotNull(), col('vals')).otherwise(
                                   struct(lit(0).alias("cnt"),
                                          (col('lead_obs')['value'] + 
                                            (((col('t_15') - 
                                               expr('INTERVAL 7 seconds 500 milliseconds')).cast('double') - 
                                              col('lead_obs')['t'].cast('double')) /
                                             (col('lag_obs')['t'].cast('double') - col('lead_obs')['t'].cast('double'))) * 
                                          (col('lag_obs')['value'] - col('lead_obs')['value'])).alias("avg"),
                                          lit(None).alias("min"),
                                          lit(None).alias("max"))).alias('vals'),
                              'obs')

plc_bin_impute.write.mode('overwrite').saveAsTable('main.led.plc_bin_impute')
display(plc_bin_impute)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- example of how to explode to get real values
# MAGIC SELECT
# MAGIC     tag, 
# MAGIC     o.t as ts,
# MAGIC     o.value as value
# MAGIC FROM
# MAGIC     (SELECT tag, explode(obs) as o FROM main.led.plc_bin_impute) a;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Example of how to evaluate interval averages
# MAGIC
# MAGIC SELECT
# MAGIC     tag,
# MAGIC     t_15,
# MAGIC     vals.avg vals_avg
# MAGIC FROM
# MAGIC     main.led.plc_bin_impute;

# COMMAND ----------

# MAGIC %pip install dbl-tempo==0.1.12

# COMMAND ----------

?dat.asofJoin

# COMMAND ----------

?TSDF

# COMMAND ----------

from pyspark.sql.functions import explode
from tempo import TSDF

dat = spark.table('main.led.plc_bin_impute') \
                .withColumn('o', explode('obs')) \
                .select(col('o.t').alias('ts'),
                        col('tag'),
                        col('o.value').alias('value'))

m1_df = dat.filter(col('tag') == 'm1')
m1_df.write.mode('overwrite').saveAsTable('main.led.m1_df')

m2_df = dat.filter(col('tag') == 'm2')
m2_df.write.mode('overwrite').saveAsTable('main.led.m2_df')

m1_tsdf = TSDF(m1_df, ts_col="ts")
m2_tsdf = TSDF(m2_df, ts_col="ts")

rslt = m2_tsdf.asofJoin(m1_tsdf).df \
              .select('ts',
                      col('right_value').alias('m1'),
                      col('value').alias('m2'))

display(rslt)
