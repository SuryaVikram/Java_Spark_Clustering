import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import org.apache.spark.api.java.JavaRDD;

import java.util.List;

public class KMeans_Q1 {
    public static void main(String args[]) {

        //create Spark Conf and Java Spark Context objects
        SparkConf sparkConf = new SparkConf().setAppName("KMeans").setMaster("local[4]");;
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        //gets the text file in Java RDD
        String path = "src/main/resources/twitter2D.txt";
        JavaRDD<String> textFile = jsc.textFile(path);

        //Reads the data from textfile, splits by ',' and saves location data as vector and tweet as string in Tuple
        JavaRDD<Tuple2<Vector, String>> parsedData = textFile.map(
                (String s) -> {
                    String[] sarray = s.split(",");
                    double[] values = new double[2];
                    for (int i = 0; i < 2; i++)
                        values[i] = Double.parseDouble(sarray[i]);
                    String tweet = sarray[sarray.length - 1];
                    Vector vals = Vectors.dense(values);
                    return new Tuple2<Vector, String>(vals,tweet);
                }
        );
        //cache parsed data to quick access
        parsedData.cache();

        //KMeans Model
        int numClusters=4;
        int numIterations=20;
        KMeansModel clusters = KMeans.train(parsedData.map(data->data._1).rdd(), numClusters, numIterations);

        //predicts on the vector in parsed data and save along with corresponding tweet in a tuple
        List<Tuple2<String, Integer>> tweetCluster = parsedData.map(data -> new Tuple2<String, Integer>(data._2, clusters.predict(data._1)))
                .sortBy(tc -> tc._2, true, 1).collect();

        //Print the tweet and its cluster
        for (Tuple2<String, Integer> tc:tweetCluster
             ) {
            System.out.println("Tweet "+ tc._1+ " is in cluster "+ tc._2);

        }


    }
}
