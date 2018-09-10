
import findspark

# Initialize and provide path
findspark.init("/usr/local/spark/spark-2.3.1-bin-hadoop2.7")

from pyspark.sql import SparkSession

def load_data():
	f = './canata.txt';
	rdd = sc.textFile(f)
	rdd.take(2)
	rdd.count()
	#a = range(100)
    
	#data = sc.parallelize(a)
	#data.take(5)


spark = SparkSession.builder \
   .master("local") \
   .appName("Linear Regression Model") \
   .config("spark.executor.memory", "1gb") \
   .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")
load_data()
