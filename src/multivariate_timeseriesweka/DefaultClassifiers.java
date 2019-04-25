/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;
import multivariate_timeseriesweka.classifiers.DTW_A;
import multivariate_timeseriesweka.classifiers.kNN;
import multivariate_timeseriesweka.elasticmeasures.DTW_D;
import multivariate_timeseriesweka.elasticmeasures.DTW_I;
import weka.classifiers.Classifier;

/**
 *
 * @author raj09hxu
 */
public class DefaultClassifiers {
    
        
    public static final Map<String, Supplier<Classifier>> CLASSIFIERS;
    static {
        Map<String, Supplier<Classifier>> map = new HashMap();
        map.put("DTW_A", DefaultClassifiers::createDTW_A);
        map.put("DTW_D", DefaultClassifiers::createDTW_D);
        map.put("DTW_I", DefaultClassifiers::createDTW_I);
        CLASSIFIERS = Collections.unmodifiableMap(map);
    }
    
    public static Classifier createDTW_A(){
        DTW_A A = new DTW_A(1);
        A.setR(0.2); //20%
        return A;
    }
    
    public static Classifier createDTW_I(){
        kNN nn = new kNN(1);
        DTW_I I = new DTW_I();
        I.setR(0.2);
        nn.setDistanceFunction(I);
        return nn;
    }
    
    public static Classifier createDTW_D(){
        kNN nn = new kNN(1);
        DTW_D D = new DTW_D();
        D.setR(0.2);
        nn.setDistanceFunction(D);
        return nn;
    }
    
}
