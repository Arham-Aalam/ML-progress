import org.apache.log4j._

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS


import org.apache.spark.sql.types.{IntegerType, DoubleType}

//import org.apache.spark.sql._
import org.apache.spark.sql.functions._

Logger.getLogger("org").setLevel(Level.ERROR)

val randomNum = scala.util.Random

val customerProducts = spark.read.format("csv").load("customer_products.csv")

//customerProducts.printSchema

val customerProductsWithHeaders = customerProducts.withColumnRenamed("_c0", "customerId").withColumnRenamed("_c1", "productId")

//customerProductsWithHeaders.printSchema


val randomVal = udf(() => { randomNum.nextInt(5) + 1 })

val collabData = customerProductsWithHeaders.withColumn("rating", randomVal())

//collabData.show(20)

val indexedProductId = new StringIndexer().setInputCol("productId").setOutputCol("indexedProductId")
//val productEncoder = new OneHotEncoderEstimator().setInputCols(Array("indexedProductId")).setOutputCols(Array("PID"))

val indexedCustomerId = new StringIndexer().setInputCol("customerId").setOutputCol("indexedCustomerId")
//val customerEncoder = new OneHotEncoderEstimator().setInputCols(Array("indexedCustomerId")).setOutputCols(Array("UID"))

val prodIdData = indexedProductId.fit(collabData).transform(collabData)
val custIdData = indexedCustomerId.fit(collabData).transform(collabData)

val p = prodIdData.withColumn("ProdID", prodIdData("indexedProductId").cast(IntegerType))
val c = custIdData.withColumn("CustID", custIdData("indexedCustomerId").cast(IntegerType))

val finalData = p.join(c, Seq("customerId", "productId", "rating"))

val Array(training, test) = finalData.randomSplit(Array(0.7, 0.3))

val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("CustID").setItemCol("ProdID").setRatingCol("rating")

val model = als.fit(training)

model.setColdStartStrategy("drop")

val predictions = model.transform(test)

val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")

//generate 10 products recommendation for each user
val userRecs = model.recommendForAllUsers(10)
userRecs.show(20)

// top 10 users recommendation for each movie
val movieRecs = model.recommendForAllItems(10)
movieRecs.show(20)










