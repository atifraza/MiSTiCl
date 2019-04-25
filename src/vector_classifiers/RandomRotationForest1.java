/*
 Adjusted Rotation Forest.

VERSION 1: 

1. Limit the max number of attributes per tree
    Test 1: make sure it still does the same thing when maxNumAttributes> numAtts in all cases
10/2/17: Run TunedRotationForest and RandomRotationForest1 with maxNumAttributes=10000 
should be no difference: 

    Test 2: check it still runs with problems where maxNumAttributes> numAtts
    Test 3: Compare accuracy on problems where maxNumAttributes> numAtts
    Test 4: Perform timing experiment on problems where maxNumAttributes> numAtts

2. Impose bagging and work out OOB Error
 */
package vector_classifiers;

import java.util.Random;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.TunedRotationForest;

/**
 *
 * @author ajb
 */
public class RandomRotationForest1 extends TunedRotationForest{
    private int maxNumAttributes=100;
    public RandomRotationForest1(){
        this.estimateAccFromTrain(false);
        this.tuneFeatures(false);
        this.tuneTree(false);
        this.setNumIterations(200);
    }
    public void setMaxNumAttributes(int m){
        maxNumAttributes=m;
    }
    @Override
    protected int [] attributesPermutation(int numAttributes, int classAttribute,
                                         Random random) {

    int [] permutation = new int[numAttributes-1];
    int i = 0;
//This just ignores the class attribute
    for(; i < classAttribute; i++){
      permutation[i] = i;
    }
    for(; i < permutation.length; i++){
      permutation[i] = i + 1;
    }

    permute( permutation, random );
    if(numAttributes>maxNumAttributes){
//TRUNCTATE THE PERMATION TO CONSIDER maxNumAttributes. 
// we could do this more efficiently, but this is the simplest way. 
        int[] temp = new int[maxNumAttributes];
       System.arraycopy(permutation, 0, temp, 0, maxNumAttributes);
       permutation=temp;
    }
    
    return permutation;
  }    
    public static void main(String[] args){
        
    }
}
