/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka.classifiers;

import multivariate_timeseriesweka.elasticmeasures.DTW_D;
import utilities.ClassifierResults;
import weka.core.Capabilities;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 * 
 * A wrapper class that extends the current kNN implementation 
 * to allow us to pass in our relational attribute measures.
 */
public class kNN extends weka.classifiers.lazy.kNN{
       
    public kNN(int k){
        super(k);
    }
    
    @Override
    public Capabilities getCapabilities(){
        Capabilities output = super.getCapabilities();
        output.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        return output;
    }
    
    public static void main(String[] args) {
       
        String dataset = "UWaveGesture";
        Instances train = utilities.ClassifierTools.loadData("E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\"+dataset+"\\"+dataset+"_TRAIN.arff");//load some train data.
        Instances test = utilities.ClassifierTools.loadData("E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\"+dataset+"\\"+dataset+"_TEST.arff");//load some test data.
        
        kNN nn = new kNN(1);
        nn.setDistanceFunction(new DTW_D());
        //nn.setDistanceFunction(new DTW_I());
        
        nn.buildClassifier(train);
        
        try{
        ClassifierResults results = utilities.ClassifierTools.constructClassifierResults(nn, test);
        System.out.println(results.acc);
        }catch(Exception ex){
            System.out.println(ex);
        }
    }
    
}
