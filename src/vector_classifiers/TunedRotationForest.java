/*
 */
package vector_classifiers;

import fileIO.OutFile;
import java.io.File;
import weka.classifiers.trees.*;
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
import weka.classifiers.Evaluation;
import weka.classifiers.meta.RotationForest;
import utilities.ClassifierResults;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author ajb
 */
public class TunedRotationForest extends RotationForest implements SaveParameterInfo,TrainAccuracyEstimate{
    boolean tuneTree=true;
    int[] numTreesRange;
    int[] numFeaturesRange;
    String trainPath="";
    boolean tuneFeatures=false;
    boolean debug=false;
    boolean findTrainAcc=true;
    Random rng;
    ArrayList<Double> accuracy;
    private ClassifierResults res =new ClassifierResults();
    
    
    public TunedRotationForest(){
        super();
        this.setNumIterations(200);
        rng=new Random();
        accuracy=new ArrayList<>();
        
    }   
    
    public void setSeed(int s){
        super.setSeed(s);
        rng=new Random();
        rng.setSeed(s);
    }
     public void tuneFeatures(boolean b){
        tuneFeatures=b;
    }
     public void debug(boolean b){
        this.debug=b;
    }
     public void justBuildTheClassifier(){
        estimateAccFromTrain(false);
        tuneFeatures(false);
        tuneTree(false);
        debug=false;
    }

     public void estimateAccFromTrain(boolean b){
        this.findTrainAcc=b;
    }
  
    public void tuneTree(boolean b){
        tuneTree=b;
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
        String result="BuildTime,"+res.buildTime+",TrainAcc,"+res.acc+",numTrees,"+this.getNumIterations()+",NumFeatures,"+this.getMaxGroup();
        for(double d:accuracy)
            result+=","+d;
        
        return result;
    }
    protected final void setDefaultGridSearchRange(int m){
        numTreesRange=new int[11];
        numTreesRange[0]=10;
        for(int i=1;i<numTreesRange.length;i++)
            numTreesRange[i]=(50)*i;  //This may be slow!
        
        if(tuneFeatures){
            if(m>10){// Space evenly possible values
                numFeaturesRange=new int[10];
                numFeaturesRange[0]=3;
                for(int i=1;i<10;i++)
                    numFeaturesRange[i]=3+i;
            }
            else{
                numFeaturesRange=new int[m-3];
                for(int i=0;i<m-3;i++)
                    numFeaturesRange[i]=m+i;
            }
        }
        else
            numFeaturesRange=new int[]{3};
            
    }

    protected final void setDefaultFeatureRange(int m){
//This only involves 55 or 44 parameter searches, unlike RBF that uses 625 by default.   
        if(debug)
            System.out.println("Setting default features....");

        if(tuneFeatures){
            if(m>10)//Include defaults for Weka (Utils.log2(m)+1) and R version  (int)Math.sqrt(m)
                numFeaturesRange=new int[]{3,10,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
            else
                numFeaturesRange=new int[]{3,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
        }
        else
            numFeaturesRange=new int[]{3};
  }
    
    
    
    @Override
    public void buildClassifier(Instances data) throws Exception{
//        res.buildTime=System.currentTimeMillis(); //removed with cv changes  (jamesl) 
        long startTime=System.currentTimeMillis(); 
        //now calced separately from any instance on ClassifierResults, and added on at the end
                
                
        int folds=10;
        if(folds>data.numInstances())
            folds=data.numInstances();
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        super.setSeed(rng.nextInt());
        if(tuneTree){
            if(numTreesRange==null)
                setDefaultGridSearchRange(data.numAttributes()-1);
            else if(numFeaturesRange==null)
                setDefaultFeatureRange(data.numAttributes()-1);

            double bestErr=1.0;
  //Its a local nested class! urggg            
            class Pair{
                int x,y;
                Pair(int a, int b){
                    x=a;
                    y=b;
                }
            }
            
            CrossValidator cv = new CrossValidator();
            cv.setSeed(rng.nextInt()); 
            cv.setNumFolds(folds);
            cv.buildFolds(data);
            ClassifierResults tempres = null;
            res.acc = -1;//is initialised using default constructor above, left it there 
            //to avoid any unforeseen clashes (jamesl) 
            
            ArrayList<Pair> ties=new ArrayList<>();
            for(int numFeatures:numFeaturesRange){//Need to start from scratch for each
                if(debug)
                    System.out.println(" numFeatures ="+numFeatures);
                for(int numTrees:numTreesRange){//Need to start from scratch for each
                
                    RotationForest t= new RotationForest();
                    t.setMaxGroup(numFeatures);
                    t.setMinGroup(numFeatures);
                    t.setNumIterations(numTrees);
                    t.setSeed(rng.nextInt());
                    
                //new
                    tempres = cv.crossValidateWithStats(t, data);
                    
                    if(debug)
                        System.out.println("\t numTrees ="+numTrees+" Acc = "+tempres.acc);
                    
                    accuracy.add(tempres.acc);
                    if (tempres.acc > res.acc) {
                        res = tempres;
                        ties=new ArrayList<>();//Remove previous ties
                        ties.add(new Pair(numFeatures,numTrees));
                    }
                    else if(tempres.acc == res.acc){//Sort out ties
                        ties.add(new Pair(numFeatures,numTrees));
                    }
                //endofnew
                    
                //old
//                    Instances temp=new Instances(data);
//                    Evaluation eval=new Evaluation(temp);
//                    eval.crossValidateModel(t, temp, folds, rng);
//                    double e=eval.errorRate();
//                    if(debug)
//                        System.out.println("\t numTrees ="+numTrees+" Acc = "+(1-e));
//                    accuracy.add(1-e);
//                    if(e<bestErr){
//                        bestErr=e;
//                        ties=new ArrayList<>();//Remove previous ties
//                        ties.add(new Pair(numFeatures,numTrees));
//                    }
//                    else if(e==bestErr){//Sort out ties
//                        ties.add(new Pair(numFeatures,numTrees));
//                    }
                //endofold
                }
            }
            int bestNumTrees=0;
            int bestNumAtts=0;
 
            Pair best=ties.get(rng.nextInt(ties.size()));
            bestNumAtts=best.x;
            bestNumTrees=best.y;
            this.setNumIterations(bestNumTrees);
            this.setMaxGroup(bestNumAtts);
            this.setMinGroup(bestNumAtts);
//            res.acc=1-bestErr; //removed with cv change, saved from cver (jamesl) 
            if(debug)
                System.out.println("Best num atts ="+bestNumAtts+" best num trees="+bestNumTrees+" "+bestNumTrees+" best Acc ="+res.acc);
        }
/*If there is no parameter search, then there is no train CV available.        
this gives the option of finding one. It is inefficient
*/        
        else if(findTrainAcc){
            RotationForest t= new RotationForest();
            t.setMaxGroup(this.getMaxGroup());
            t.setMinGroup(this.getMinGroup());
            t.setNumIterations(this.getNumIterations());
            t.setSeed(rng.nextInt());
            
            //new (jamesl) 
            CrossValidator cv = new CrossValidator();
            cv.setSeed(rng.nextInt()); //trying to mimick old seeding behaviour below
            cv.setNumFolds(folds);
            cv.buildFolds(data);
            
            res = cv.crossValidateWithStats(t, data);
            //endofnew
            
            //old
//            Instances temp=new Instances(data);
//            Evaluation eval=new Evaluation(temp);
//            t.setSeed(rng.nextInt());
//            eval.crossValidateModel(t, temp, folds, rng);
//            res.acc=1-eval.errorRate();
            //endofold
        }
        
        super.buildClassifier(data);
        res.buildTime=System.currentTimeMillis()-startTime;
        if(trainPath!=""){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(data.relationName()+",TunedRotF,Train");
            f.writeLine(getParameters());
            f.writeLine(res.acc+"");
            f.writeString(res.writeInstancePredictions());
            
        }
    }
  
    public static void jamesltests() {
        //tests to confirm correctness of cv changes
        //summary: pre/post change avg accs over 50 folds
        // train: 0.9770149253731346 vs 0.9752238805970153
        // test: 0.9672886297376094 vs 0.9673858114674441
        
        //post change trainaccs/testaccs on 50 folds of italypowerdemand: 
        //      avg trainacc=0.9752238805970153
        //folds: 
//        [0.9850746268656716, 0.9701492537313433, 1.0, 1.0, 0.9552238805970149, 
//        0.9850746268656716, 0.9850746268656716, 0.9701492537313433, 0.9552238805970149, 1.0, 
//        0.9552238805970149, 0.9850746268656716, 0.9701492537313433, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9552238805970149, 0.9701492537313433, 1.0, 0.9850746268656716, 
//        0.9850746268656716, 0.9701492537313433, 0.9552238805970149, 0.9850746268656716, 1.0, 
//        0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 
//        0.9701492537313433, 0.9850746268656716, 1.0, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9552238805970149, 0.9552238805970149, 0.9850746268656716, 0.9701492537313433, 
//        0.9850746268656716, 1.0, 0.9253731343283582, 0.9701492537313433, 0.9850746268656716, 
//        0.9402985074626866, 0.9701492537313433, 0.9552238805970149, 0.9552238805970149, 0.9850746268656716]

//              avg testacc=0.9673858114674441
//        folds: 
//        [0.9727891156462585, 0.9484936831875608, 0.966958211856171, 0.9708454810495627, 0.9747327502429544, 
//        0.9718172983479106, 0.9659863945578231, 0.9737609329446064, 0.9504373177842566, 0.9689018464528668, 
//        0.9689018464528668, 0.9708454810495627, 0.9718172983479106, 0.9737609329446064, 0.9698736637512148, 
//        0.9727891156462585, 0.9465500485908649, 0.9727891156462585, 0.9689018464528668, 0.9708454810495627, 
//        0.9718172983479106, 0.9727891156462585, 0.9630709426627794, 0.9718172983479106, 0.967930029154519, 
//        0.9659863945578231, 0.967930029154519, 0.9659863945578231, 0.9494655004859086, 0.9698736637512148, 
//        0.9504373177842566, 0.9718172983479106, 0.9708454810495627, 0.9698736637512148, 0.9659863945578231, 
//        0.9630709426627794, 0.9718172983479106, 0.9698736637512148, 0.9708454810495627, 0.9737609329446064, 
//        0.9689018464528668, 0.9698736637512148, 0.9514091350826045, 0.9698736637512148, 0.9650145772594753, 
//        0.967930029154519, 0.9737609329446064, 0.9650145772594753, 0.9718172983479106, 0.9689018464528668]
        
        //pre change trainaccs/testaccs on 50 folds of italypowerdemand: 
//            avg trainacc=0.9770149253731346
//        folds:
//        [0.9850746268656716, 1.0, 1.0, 1.0, 0.9552238805970149, 
//        0.9850746268656716, 0.9850746268656716, 0.9701492537313433, 0.9701492537313433, 1.0, 
//        0.9552238805970149, 0.9850746268656716, 0.9701492537313433, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9402985074626866, 0.9701492537313433, 1.0, 0.9850746268656716, 
//        0.9850746268656716, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 1.0, 
//        0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 
//        0.9701492537313433, 0.9850746268656716, 1.0, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9552238805970149, 0.9701492537313433, 0.9850746268656716, 0.9701492537313433, 
//        0.9850746268656716, 1.0, 0.9253731343283582, 0.9701492537313433, 0.9850746268656716, 
//        0.9402985074626866, 0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716]

//                avg testacc=0.9672886297376094
//        folds:
//        [0.9737609329446064, 0.9484936831875608, 0.9689018464528668, 0.9698736637512148, 0.9737609329446064, 
//        0.9708454810495627, 0.9650145772594753, 0.9718172983479106, 0.93488824101069, 0.966958211856171, 
//        0.966958211856171, 0.9698736637512148, 0.9708454810495627, 0.9718172983479106, 0.9698736637512148, 
//        0.9718172983479106, 0.9426627793974732, 0.9727891156462585, 0.967930029154519, 0.9708454810495627, 
//        0.9708454810495627, 0.9727891156462585, 0.9650145772594753, 0.9718172983479106, 0.9698736637512148, 
//        0.9698736637512148, 0.9698736637512148, 0.9689018464528668, 0.9601554907677357, 0.9708454810495627, 
//        0.9582118561710399, 0.9708454810495627, 0.9708454810495627, 0.9708454810495627, 0.9698736637512148, 
//        0.9572400388726919, 0.9727891156462585, 0.9727891156462585, 0.9718172983479106, 0.9737609329446064, 
//        0.9698736637512148, 0.9689018464528668, 0.9329446064139941, 0.9708454810495627, 0.9718172983479106, 
//        0.9708454810495627, 0.9737609329446064, 0.9620991253644315, 0.9718172983479106, 0.9727891156462585]

        
        
        System.out.println("rotftests");
        
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
            
            TunedRotationForest rotF = new TunedRotationForest();
            rotF.estimateAccFromTrain(true);
            
            try {
                rotF.buildClassifier(data[0]);
            } catch (Exception ex) {
                Logger.getLogger(TunedRotationForest.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            trainAccs[r] = rotF.res.acc;
            trainAcc+=trainAccs[r];
            
            testAccs[r] = ClassifierTools.accuracy(data[1], rotF);
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
        jamesltests();
    }
  
}
