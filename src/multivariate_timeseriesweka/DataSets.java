/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka;

import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.filters.SummaryStats;
import static utilities.InstanceTools.createClassInstancesMap;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class DataSets {
    
    public static String dropboxPath = "E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\";

    //All of the multivariate datasets.
    public static String[] multivariateNames = {
        "AALTD",
        "ArabicDigit",
        //"ArticularyWord", //this is the only dataset that has sub types.
        "Cricket",
        "HandwritingAccelerometer",
        "HandwritingGyroscope",
        "JapaneseVowels",
        "MVMotion",
        "PEMS",
        "PenDigits",
        "UWaveGesture",
        "VillarData"
    };
    
    
    public static void createAndWriteSummaryStats() throws Exception{
        //load datasets
        for(String dataset : multivariate_timeseriesweka.DataSets.multivariateNames){
            
            Instances train;
            try {
                train = utilities.ClassifierTools.loadData(new File(multivariate_timeseriesweka.DataSets.dropboxPath + dataset + "/" + dataset +"_TRAIN.arff"));
            } catch (IOException ex) {
                continue; //if dataset doesn't exist move on.
            }
            
            OutFile out = new OutFile(multivariate_timeseriesweka.DataSets.dropboxPath + dataset +"_summarystats.txt");
            
            Instances[] channels = utilities.MultivariateInstanceTools.splitMultivariateInstances(train);
            
            out.writeLine("num instances " + train.numInstances());
            out.writeLine("num dimenions " + channels.length);
            out.writeLine("dimension length " + channels[0].numAttributes());
            out.writeLine("num classes " + train.numClasses());
            out.writeLine("[mean, variance, skewness, kurtosis, min, max]");
            
            //we calculate 6 stats mean, variance, skewness, kurtosis, min, max 
            double[][][] overallStatsByClass = new double[channels.length][train.numClasses()][6];
            double[][] overallStats = new double[channels.length][6];
            
            for(int i=0; i< channels.length; i++){
                overallStatsByClass[i] = calculateStatsForInstances(channels[i]);
            }
            
            for(int i=0; i< channels.length; i++){               
                for(int j =0; j< 6; j++){
                    for(int k=0; k<train.numClasses(); k++)
                        overallStats[i][j] += overallStatsByClass[i][k][j];
                    overallStats[i][j] /= train.numClasses();
                }
                out.writeLine("Channel " + i + " " + Arrays.toString(overallStats[i]));
            }
            
            for(int i=0; i< channels.length; i++){  
                out.writeLine("Channel "+ i);
                for (int j=0; j < overallStatsByClass[i].length; j++) {
                        out.writeLine("class: "+ j+ " " + Arrays.toString(overallStatsByClass[i][j]));
                }
            }
            out.closeFile();
        }         
    }
    
    public static double[][] calculateStatsForInstances(Instances dataset) throws Exception{
        //we want to bin our series by class first.
        Instances filter =new SummaryStats().process(dataset);

        Map<Double, Instances> instancesMap = createClassInstancesMap(filter);

        //we calculate 6 stats.
        double[][] seriesStats = new double[instancesMap.size()][6];

        for(Map.Entry<Double, Instances> pair : instancesMap.entrySet()){

            Instances inst = pair.getValue();
            double[][] data = utilities.InstanceTools.fromWekaInstancesArray(inst, true);
            double[] averagedStats = new double[6];
            //TODO triple check this.
            for(int i=0; i<averagedStats.length; i++){
                for(int j=0; j<inst.numInstances(); j++){
                    averagedStats[i] += data[j][i]; //column major. want to add along the access of 6 values.
                }
            }

            for(int i=0; i<averagedStats.length; i++)
                averagedStats[i] /= (double)inst.numInstances();

            seriesStats[(int)pair.getKey().doubleValue()] = averagedStats;
        }
        
        return seriesStats;
    }
    
    public static void main(String[] args) throws Exception {
        createAndWriteSummaryStats();
    }
}
