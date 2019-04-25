/*
This classifier is enhanced so that classifier builds a random forest with the 
facility to build by forward selection addition of trees to minimize OOB error.    

Further enhanced to include OOB error estimates and predictions

Further changes: 
1. set number of trees (m_numTrees) via grid search on a range (using OOB) that
defaults to 
{10 [Weka Default],100,200,.., 500 [R default],...,1000} (11 values)
2. set number of features  (max value m==numAttributes without class)
per tree (m_numFeatures) and m_numTrees through grid
search on a range 
1, 10, sqrt(m) [R default], log_2(m)+1 [Weka default], m [full set]}
(4 values)+add an option to choose randomly for each tree?
grid search is then just 55 values and because it uses OOB no CV is required
 */
package vector_classifiers;

import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.meta.Bagging;
import utilities.ClassifierResults;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author ajb
 */
public class TunedRandomForest extends RandomForest implements SaveParameterInfo, TrainAccuracyEstimate{
    boolean tune=true;
    boolean tuneFeatures=true;
    boolean debug=false;
    int[] numTreesRange;
    int[] numFeaturesRange;
    String trainPath="";
    Random rng;
    ArrayList<Double> accuracy;
    boolean crossValidate=true;
    boolean findTrainAcc=true;  //If there is no tuning, this will find the estimate with the fixed values
    private ClassifierResults res =new ClassifierResults();
    
    /* Needs investigation, temporary solution*/
    Bagging m_bagger;
   
    public void setCrossValidate(boolean b){
        crossValidate=b;
    }
    public void setTrainAcc(boolean b){
        findTrainAcc=b;
    }

    
    public TunedRandomForest(){
        super();
        m_NumIterations=500;
        m_numExecutionSlots=1; 
        m_bagger=new EnhancedBagging();
        rng=new Random();
        accuracy=new ArrayList<>();
    }
    public void setSeed(int s){
        super.setSeed(s);
        rng=new Random();
        rng.setSeed(s);
    }
    
    public void debug(boolean b){
        this.debug=b;
    }
    
    public void tuneTree(boolean b){
        tune=b;
    }
    public void tuneFeatures(boolean b){
        tuneFeatures=b;
    }
    public void setNumIterationsRange(int[] d){
        numTreesRange=d;
    }
    public void setNumFeaturesRange(int[] d){
        numFeaturesRange=d;
    }
 @Override
    public void writeCVTrainToFile(String train) {
        trainPath=train;
        findTrainAcc=true;
    }    
    @Override
    public boolean findsTrainAccuracyEstimate(){ return findTrainAcc;}
    
    @Override
    public ClassifierResults getTrainResults(){
//Temporary : copy stuff into res.acc here
//        res.acc=ensembleCvAcc;
//TO DO: Write the other stats        
        return res;
    }        

    @Override
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",TrainAcc,"+res.acc+",numTrees,"+this.getNumIterations()+",NumFeatures,"+this.getNumFeatures();
        for(double d:accuracy)
            result+=","+d;
        return result;
    }
    protected final void setDefaultGridSearchRange(int m){
//This only involves 100 parameter searches.   
        if(debug)
            System.out.println("Setting defaults ....");
        numTreesRange=new int[10];
        numTreesRange[0]=10; //Weka default
        for(int i=1;i<numTreesRange.length;i++)
            numTreesRange[i]=100*i;
        if(tuneFeatures){
            if(m>10){// Space evenly possible values
                numFeaturesRange=new int[10];
                numFeaturesRange[0]=(int)Math.sqrt(m);
                numFeaturesRange[1]=(int) Utils.log2(m)+1;
                for(int i=2;i<10;i++)
                    numFeaturesRange[i]=((i-1)*m)/10;
            }
            else{
                numFeaturesRange=new int[m];
                for(int i=0;i<m;i++)
                    numFeaturesRange[i]=i+1;
            }
        }
        else
            numFeaturesRange=new int[]{(int)Math.sqrt(m)};
            
  }

    protected final void setDefaultFeatureRange(int m){
//This only involves 55 or 44 parameter searches, unlike RBF that uses 625 by default.   
        if(debug)
            System.out.println("Setting default features....");

        if(tuneFeatures){
            if(m>10)//Include defaults for Weka (Utils.log2(m)+1) and R version  (int)Math.sqrt(m)
                numFeaturesRange=new int[]{1,10,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
            else
                numFeaturesRange=new int[]{1,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
        }
        else
            numFeaturesRange=new int[]{(int)Math.sqrt(m)};
            
  }
    
    @Override
    public void buildClassifier(Instances data) throws Exception{
        long startTime=System.currentTimeMillis(); 
//********* 1: Set up the main classifier with standard Weka calls ***************/      
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
//this is only used if CV is used to find parameters or estimate acc from train data        
        int folds=10;
        if(folds>data.numInstances())
            folds=data.numInstances();
        super.setSeed(rng.nextInt());
        
/******* 2. Tune parameters if required: 
 tunes number of features and number of trees  using either cross validation
 if crossValidate==true or out of bag error otherwise
 * 
 * NOTE: the number of trees could be found incrementally, just start with the smallest
 * number and add in each time rather than rebuild. It would massively speed up the search
 * this has been implemented for the EnhancedBagger,but is not yet used. 
 * Obviously cannot do this for the number of attributes
 */
        if(tune){
            if(numTreesRange==null)
                setDefaultGridSearchRange(data.numAttributes()-1);
            else if(numFeaturesRange==null)
                setDefaultFeatureRange(data.numAttributes()-1);
  //Its a local nested class! urggg            
            class Pair{
                int x,y;
                Pair(int a, int b){
                    x=a;
                    y=b;
                }
            }
            ClassifierResults tempres = null;
            res.acc = -1;//is initialised using default constructor above, left it there 
            CrossValidator cv = new CrossValidator();
            if(crossValidate){
                cv.setSeed(rng.nextInt()); 
                cv.setNumFolds(folds);
                cv.buildFolds(data);
            }
            //to avoid any unforeseen clashes (jamesl) 
            ArrayList<Pair> ties=new ArrayList<>();
            for(int numFeatures:numFeaturesRange){//Need to start from scratch for each
                for(int numTrees:numTreesRange){//Need to start from scratch for each                   t= new RandomForest();
                    RandomForest t= new RandomForest();
                    t.setSeed(rng.nextInt());
                    t.setNumFeatures(numFeatures);
                    t.setNumIterations(numTrees);
                    
                //new (jamesl)
                    if(crossValidate){  
                        tempres = cv.crossValidateWithStats(t, data);
                    }else{
                        Instances temp=new Instances(data);
                        t.buildClassifier(temp);
                        tempres = new ClassifierResults();
                        tempres.acc = 1 - t.measureOutOfBagError();   
                    }
                    if(debug)
                        System.out.println(" numTrees ="+numTrees+" Acc = "+tempres.acc);
                    
                    accuracy.add(tempres.acc);
                    if (tempres.acc > res.acc) {
                        res = tempres;
                        ties=new ArrayList<>();//Remove previous ties
                        ties.add(new Pair(numFeatures,numTrees));
                    }
                    else if(tempres.acc == res.acc){//Sort out ties
                        ties.add(new Pair(numFeatures,numTrees));
                    }
                }
            }
            int bestNumTrees;
            int bestNumAtts;
            Pair best=ties.get(rng.nextInt(ties.size()));
            bestNumAtts=best.x;
            bestNumTrees=best.y;
            
            this.setNumIterations(bestNumTrees);
            this.setNumFeatures(bestNumAtts);
//            res.acc=1-bestErr; //removed with cv change, saved from cver (jamesl) 
            if(debug)
                System.out.println("Best num atts ="+bestNumAtts+" best num trees = "+bestNumTrees+" best train acc = "+res.acc);
        }
        else //Override WEKA's default which is worse than sqrt(m)
            setNumFeatures(Math.max(1,(int)Math.sqrt(data.numAttributes()-1)));
        
/******** 3. Build final classifier ***************/        
/*Cant call super.buildClassifier as it resets the bagger to Bagging instead of 
 EnhancedBagging so instead straight cut and paste from RandomForest, with 
 Bagging changed to EnhancedBagging and default size changed
*/
        m_bagger = new EnhancedBagging();
        RandomTree rTree = new RandomTree();
        // set up the random tree options
        rTree.setKValue(getNumFeatures());
        rTree.setMaxDepth(getMaxDepth());
        // set up the bagger and build the forest
        m_bagger.setClassifier(rTree);
        m_bagger.setSeed(rng.nextInt());
        m_bagger.setNumIterations(m_NumIterations);
        m_bagger.setCalcOutOfBag(true);
        m_bagger.setNumExecutionSlots(m_numExecutionSlots);
        m_bagger.buildClassifier(data);       

/*** 4. Find the estimates of the train acc, either through CV or OOB  ****/
// do this after the main build in case OOB is used, because we need the main 
//classifier for that
//NOTE IF THE CLASSIFIER IS TUNED THIS WILL BE A POSSIBLE SOURCE OF BIAS
//It should really be nested up a level.         
 
    
        if(findTrainAcc){   //Need find train acc, either through CV or OOB
            if(crossValidate){  
                RandomForest t= new RandomForest();
                t.setNumFeatures(this.getNumFeatures());
                t.setNumIterations(this.getNumIterations());
                t.setSeed(rng.nextInt());
 
                //new (jamesl) 
                CrossValidator cv = new CrossValidator();
                cv.setSeed(rng.nextInt()); //trying to mimick old seeding behaviour below
                cv.setNumFolds(folds);
                cv.buildFolds(data);
                res = cv.crossValidateWithStats(t, data);
            }
            else{
                res.acc=1-this.measureOutOfBagError();
//Get OOB probabilities. This is not possible with the standard 
//random forest bagger, hence the use of EnhancedBagger
                System.out.println("BAGGER CLASS = "+m_bagger.getClass().getName());
                
                ((EnhancedBagging)m_bagger).findOOBProbabilities();
                double[][] OOBPredictions=((EnhancedBagging)m_bagger).OOBProbabilities;
                for(int i=0;i<data.numInstances();i++)
                    res.storeSingleResult(data.instance(i).classValue(),OOBPredictions[i]);
            }
        }
        
        res.buildTime=System.currentTimeMillis()-startTime;
        if(trainPath!=""){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(data.relationName()+",TunedRandF,Train");
            f.writeLine(getParameters());
            f.writeLine(res.acc+"");
            f.writeLine(res.writeInstancePredictions());
        } 
        
    }

    public void addTrees(int n, Instances data) throws Exception{
        EnhancedBagging newTrees =new EnhancedBagging();
        RandomTree rTree = new RandomTree();
        // set up the random tree options
        rTree.setKValue(getNumFeatures());
        rTree.setMaxDepth(getMaxDepth());
//Change this so that it is reproducable
        Random r= new Random();
        newTrees.setSeed(r.nextInt());
        newTrees.setClassifier(rTree);
        newTrees.setNumIterations(n);
        newTrees.setCalcOutOfBag(true);
        newTrees.setNumExecutionSlots(m_numExecutionSlots);
        newTrees.buildClassifier(data);
        newTrees.findOOBProbabilities();
//Merge with previous
        m_bagger.aggregate(newTrees);
        m_bagger.finalizeAggregation();
//Update OOB Error, as this is seemingly not done in the bagger
        
        m_NumIterations+=n;
        m_bagger.setNumIterations(m_NumIterations); 
        ((EnhancedBagging)m_bagger).mergeBaggers(newTrees);
        
    }
    public double getBaggingPercent(){
      return m_bagger.getBagSizePercent();
    }

    private class EnhancedBagging extends Bagging{
        /* Needs investigation, temporary solution*/
        protected double m_OutOfBagError;
// 
        @Override
        public void buildClassifier(Instances data)throws Exception {
            super.buildClassifier(data);
            m_data=data;
//            System.out.println(" RESET BAGGER");

        }
        double[][] OOBProbabilities;
        int[] counts;
        public void mergeBaggers(EnhancedBagging other){
            for (int i = 0; i < m_data.numInstances(); i++) {
                for (int j = 0; j < m_data.numClasses(); j++) {
                      OOBProbabilities[i][j]=counts[i]*OOBProbabilities[i][j]+other.counts[i]*other.OOBProbabilities[i][j];
                      OOBProbabilities[i][j]/=counts[i]+other.counts[i];
                }
                counts[i]=counts[i]+other.counts[i];
            }
//Merge  m_inBags index i is classifier, j the instance
            boolean[][] inBags = new boolean[m_inBag.length+other.m_inBag.length][];
            for(int i=0;i<m_inBag.length;i++)
                inBags[i]=m_inBag[i];
            for(int i=0;i<other.m_inBag.length;i++)
                inBags[m_inBag.length+i]=other.m_inBag[i];
            m_inBag=inBags;
            findOOBError();
        }
        public void findOOBProbabilities() throws Exception{
            OOBProbabilities=new double[m_data.numInstances()][m_data.numClasses()];
            counts=new int[m_data.numInstances()];
            for (int i = 0; i < m_data.numInstances(); i++) {
                for (int j = 0; j < m_Classifiers.length; j++) {
                    if (m_inBag[j][i])
                      continue;
                    counts[i]++;
                    double[] newProbs = m_Classifiers[j].distributionForInstance(m_data.instance(i));
                // average the probability estimates
                    for (int k = 0; k < m_data.numClasses(); k++) {
                        OOBProbabilities[i][k] += newProbs[k];
                    }
                }
                for (int k = 0; k < m_data.numClasses(); k++) {
                    OOBProbabilities[i][k] /= counts[i];
                }
            }
        }
        
        public double findOOBError(){
            double correct = 0.0;
            for (int i = 0; i < m_data.numInstances(); i++) {
                double[] probs = OOBProbabilities[i];
                int vote =0;
                for (int j = 1; j < probs.length; j++) {
                  if(probs[vote]<probs[j])
                      vote=j;
            }
            if(m_data.instance(i).classValue()==vote) 
                correct++;
            }
            m_OutOfBagError=1- correct/(double)m_data.numInstances();
//            System.out.println(" NEW OOB ERROR ="+m_OutOfBagError);
            return m_OutOfBagError;
        }
        
 //       public double getOOBError
    }
    public double findOOBError() throws Exception{
        ((EnhancedBagging)m_bagger).findOOBProbabilities();
        return ((EnhancedBagging)m_bagger).findOOBError();
    }
    public double[][] findOOBProbabilities() throws Exception{
        ((EnhancedBagging)m_bagger).findOOBProbabilities();
        return ((EnhancedBagging)m_bagger).OOBProbabilities;
    }
    public double[][] getOBProbabilities() throws Exception{
        return ((EnhancedBagging)m_bagger).OOBProbabilities;
    }
  
    public static void jamesltests() {
        //tests to confirm correctness of cv changes
        //summary: pre/post change avg accs over 50 folds
        // train: 0.9689552238805973 vs 0.9680597014925374
        // test: 0.9590670553935859 vs 0.9601943634596699
        
        //post change trainaccs/testaccs on 50 folds of italypowerdemand: 
//                trainacc=0.9680597014925374
//        folds:
//        [0.9552238805970149, 0.9552238805970149, 1.0, 1.0, 0.9552238805970149, 
//        0.9701492537313433, 0.9552238805970149, 0.9402985074626866, 0.9552238805970149, 1.0, 
//        0.9552238805970149, 0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 
//        0.9552238805970149, 0.9402985074626866, 0.9850746268656716, 1.0, 0.9552238805970149, 
//        0.9850746268656716, 0.9701492537313433, 0.9701492537313433, 0.9552238805970149, 1.0, 
//        0.9701492537313433, 0.9701492537313433, 0.9552238805970149, 0.9552238805970149, 0.9850746268656716, 
//        0.9402985074626866, 0.9850746268656716, 1.0, 0.9850746268656716, 0.9850746268656716, 
//        0.9402985074626866, 0.9552238805970149, 0.9253731343283582, 0.9701492537313433, 0.9701492537313433, 
//        0.9701492537313433, 1.0, 0.9402985074626866, 0.9701492537313433, 0.9552238805970149, 
//        0.9402985074626866, 0.9701492537313433, 0.9552238805970149, 0.9850746268656716, 0.9701492537313433]
//
//                testacc=0.9601943634596699
//        folds:
//        [0.9543245869776482, 0.9271137026239067, 0.9582118561710399, 0.966958211856171, 0.9718172983479106, 
//        0.9650145772594753, 0.9582118561710399, 0.9689018464528668, 0.9494655004859086, 0.966958211856171, 
//        0.9620991253644315, 0.9698736637512148, 0.9659863945578231, 0.9659863945578231, 0.966958211856171, 
//        0.9718172983479106, 0.9416909620991254, 0.9640427599611273, 0.9368318756073858, 0.9698736637512148,
//        0.966958211856171, 0.9494655004859086, 0.9582118561710399, 0.9698736637512148, 0.9620991253644315, 
//        0.9650145772594753, 0.9640427599611273, 0.9601554907677357, 0.9319727891156463, 0.967930029154519, 
//        0.9523809523809523, 0.967930029154519, 0.9591836734693877, 0.9727891156462585, 0.9572400388726919, 
//        0.9329446064139941, 0.9718172983479106, 0.9620991253644315, 0.9689018464528668, 0.9514091350826045, 
//        0.9630709426627794, 0.966958211856171, 0.9543245869776482, 0.9718172983479106, 0.9698736637512148, 
//        0.9552964042759962, 0.9727891156462585, 0.9329446064139941, 0.9630709426627794, 0.9650145772594753]

        //pre change trainaccs/testaccs on 50 folds of italypowerdemand: 
//                trainacc=0.9689552238805973
//        folds:
//        [0.9402985074626866, 0.9701492537313433, 1.0, 1.0, 0.9253731343283582, 
//        0.9850746268656716, 0.9850746268656716, 0.9552238805970149, 0.9552238805970149, 1.0, 
//        0.9402985074626866, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9253731343283582, 0.9850746268656716, 1.0, 0.9701492537313433, 
//        0.9850746268656716, 0.9850746268656716, 0.9701492537313433, 0.9701492537313433, 1.0, 
//        0.9552238805970149, 0.9552238805970149, 0.9552238805970149, 0.9701492537313433, 0.9850746268656716, 
//        0.9552238805970149, 0.9850746268656716, 1.0, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9552238805970149, 0.9402985074626866, 0.9701492537313433, 0.9552238805970149, 
//        0.9850746268656716, 1.0, 0.9402985074626866, 0.9701492537313433, 0.9402985074626866, 
//        0.9253731343283582, 0.9701492537313433, 0.9552238805970149, 0.9552238805970149, 0.9552238805970149]
//
//                testacc=0.9590670553935859
//        folds:
//        [0.9514091350826045, 0.9290573372206026, 0.9591836734693877, 0.967930029154519, 0.9708454810495627, 
//        0.9689018464528668, 0.9650145772594753, 0.9708454810495627, 0.9358600583090378, 0.967930029154519, 
//        0.9640427599611273, 0.9640427599611273, 0.9630709426627794, 0.9659863945578231, 0.9543245869776482, 
//        0.9689018464528668, 0.9514091350826045, 0.9659863945578231, 0.9659863945578231, 0.9611273080660836,
//        0.9689018464528668, 0.9504373177842566, 0.9504373177842566, 0.9698736637512148, 0.9630709426627794, 
//        0.9620991253644315, 0.9582118561710399, 0.966958211856171, 0.9543245869776482, 0.9640427599611273, 
//        0.9514091350826045, 0.9533527696793003, 0.9659863945578231, 0.9689018464528668, 0.9572400388726919, 
//        0.967930029154519, 0.9689018464528668, 0.9698736637512148, 0.9698736637512148, 0.9582118561710399, 
//        0.9601554907677357, 0.966958211856171, 0.9378036929057337, 0.9689018464528668, 0.9650145772594753, 
//        0.8794946550048591, 0.9737609329446064, 0.9319727891156463, 0.9484936831875608, 0.9689018464528668]


        System.out.println("ranftestsWITHCHANGES");
        
        String dataset = "ItalyPowerDemand";
        
        Instances train = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
        Instances test = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TEST");
        
        int rs = 50;
        
        double[] trainAccs = new double[rs];
        double[] testAccs = new double[rs];
        double trainAcc =0;
        double testAcc =0;
        for (int r = 0; r < rs; r++) {
            Instances[] data = InstanceTools.resampleTrainAndTestInstances(train, test, r);
            
            TunedRandomForest ranF = new TunedRandomForest();
            ranF.setCrossValidate(true);
            ranF.setTrainAcc(true);
            try {
                ranF.buildClassifier(data[0]);
            } catch (Exception ex) {
                Logger.getLogger(TunedRandomForest.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            trainAccs[r] = ranF.res.acc;
            trainAcc+=trainAccs[r];
            
            testAccs[r] = ClassifierTools.accuracy(data[1], ranF);
            testAcc+=testAccs[r];
            
            System.out.print(".");
        }
        trainAcc/=rs;
        testAcc/=rs;
        
        System.out.println("\nacc="+trainAcc);
        System.out.println(Arrays.toString(trainAccs));
        
        System.out.println("\nacc="+testAcc);
        System.out.println(Arrays.toString(testAccs));
    }
      
    
    public static void main(String[] args) {
  //      jamesltests();
  //      testBinMaker();
 //       System.exit(0);
        DecimalFormat df = new DecimalFormat("##.###");
        try{
                String s="ItalyPowerDemand";
                System.out.println(" PROBLEM ="+s);
                Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TRAIN");
                Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TEST");
                TunedRandomForest rf=new TunedRandomForest();
               rf.buildClassifier(train);
                System.out.println(" bag percent ="+rf.getBaggingPercent()+" OOB error "+rf.measureOutOfBagError());
/*                
                for(int i=0;i<5;i++){
                    System.out.println(" Number f trees ="+rf.getNumTrees()+" num elements ="+rf.numElements());
                    System.out.println(" bag percent ="+rf.getBaggingPercent()+" OOB error "+rf.measureOutOfBagError());
                    double[][] probs=rf.findOOBProbabilities();
/*s
                    for (int j = 0; j < probs.length; j++) {
                        double[] prob = probs[j];
                        for (int k = 0; k < prob.length; k++) {
                            System.out.print(","+prob[k]);
                        }
                        System.out.println("");
                        
                    }

                    rf.addTrees(50, train);
                }
                int correct=0;
                for(Instance ins:test){
                    double[] pred=rf.distributionForInstance(ins);
                    double cls=rf.classifyInstance(ins);
                    if(cls==ins.classValue())
                        correct++;
                }
                System.out.println(" ACC = "+((double)correct)/test.numInstances());
//                System.out.println(" calc out of bag? ="+rf.m_bagger.m_CalcOutOfBag);
                System.exit(0);
                double a =ClassifierTools.singleTrainTestSplitAccuracy(rf, train, test);
                System.out.println(" error ="+df.format(1-a));
//                tsbf.buildClassifier(train);
 //               double c=tsbf.classifyInstance(test.instance(0));
 //               System.out.println(" Class ="+c);
*/
        }catch(Exception e){
            System.out.println("Exception "+e);
            e.printStackTrace();
            System.exit(0);
        }
    }
  
}
