import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

import org.apache.log4j._

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.types.{IntegerType, DoubleType}

import org.apache.spark.ml.feature.VectorAssembler

Logger.getLogger("org").setLevel(Level.ERROR)


// Loads data.
val productPrice = spark.read.format("csv").option("header", "true").load("product_price.csv")
val productDescription = spark.read.format("csv").option("header", "true").load("Product_description.csv")
val productPrice2 = productPrice.where(productPrice("features") !== "0.000").withColumn("featuress", productPrice("features").cast(DoubleType)).drop("features")

println("======================Price data Min and Max =================")
productPrice2.agg(min("featuress"), max("featuress")).show()
//println("count-------------")
//productPrice.where(productPrice("features") === "0.000").count()

println("================ K means for Tops ============")
val productDescriptionTops = productDescription.where(productDescription("primary-product-category-id") === "TOPS").drop("features")
//productDescription.show(20)

val productPriceTopData = productPrice2.join(productDescriptionTops, productPrice2("product_id") === productDescriptionTops("product-id"), "inner").select("product_id", "featuress", "primary-product-category-id")
productPriceTopData.show(20)

val assemblerTops = new VectorAssembler().setInputCols(Array("featuress")).setOutputCol("features")
val vectorDataTops = assemblerTops.transform(productPriceTopData).drop("featuress")

println("======================Tops data Min and Max ===================")
vectorDataTops.agg(min("features"), max("features")).show()
vectorDataTops.show(20)

var kCount = 6;

val kmeans4Tops = new KMeans().setK(kCount).setSeed(1L)
val modelOfTops = kmeans4Tops.fit(vectorDataTops)

val predictionsTops = modelOfTops.transform(vectorDataTops)

//Evaluate clustering by computing Silhouette score
val TopsEvaluator = new ClusteringEvaluator()

val TopsSilhouette = TopsEvaluator.evaluate(predictionsTops)
println(s"================Silhouette with squared euclidean distance = $TopsSilhouette")

var i=0
var maxs: Long = 0
var clusterNumber = 0
for(i <- 1 to kCount) {
    println(i + "th cluster count ====")
    var count = predictionsTops.where(predictionsTops("prediction") === i-1).count()
    println(count)
    if(count > maxs) {
        maxs = count
        clusterNumber = i-1
    }
}

//show clusters
//predictionsTops.collect().foreach(println)

println("=========== Range of Maximum cluster count ========")
predictionsTops.where(predictionsTops("prediction") === clusterNumber).agg(min("features"), max("features")).show()


//val assembler = new VectorAssembler().setInputCols(Array("featuress")).setOutputCol("features")
//val vectorData = assembler.transform(productPrice2).drop("featuress")

//vectorData.join(productDescription, vectorData("product_id") === productDescription("product-id"), "left").select("primary-product-category-id", "product_id").show(40)
//vectorData.join(productDescription, vectorData("product_id") === productDescription("product-id"), "left").select("primary-product-category-id", "product_id").count()
//vectorData.select("primary-product-category-id").where()

//vectorData.show(20)

//productDescription.show(20)

//val kmeans = new KMeans().setK(3).setSeed(1L)
//val model = kmeans.fit(vectorData)

//val predictions = model.transform(vectorData)

//Evaluate clustering by computing Silhouette score
//val evaluator = new ClusteringEvaluator()

//val silhouette = evaluator.evaluate(predictions)
//println(s"Silhouette with squared euclidean distance = $silhouette")

//show clusters
//predictions.collect().foreach(println)