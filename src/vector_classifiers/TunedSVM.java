/*
Tony's attempt to see the effect of parameter setting on SVM.

Two parameters: 
kernel para: for polynomial this is the weighting given to lower order terms
    k(x,x')=(<x'.x>+a)^d
regularisation parameter, used in the SMO 

m_C
 */
package vector_classifiers;

import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import utilities.ClassifierResults;
import weka.core.*;

/**
 *
 * @author ajb
 
 TunedSVM sets the margin c through a ten fold cross validation.
 
 If the kernel type is RBF, also set sigma through CV, same values as c
 
 NOTE: 
 1. CV could be done faster?
 2. Could use libSVM instead
 * 
 */
public class TunedSVM extends SMO implements SaveParameterInfo, TrainAccuracyEstimate {
    boolean setSeed=false;
    int seed;
    int min=-16;
    int max=16;
    double[] paraSpace;
    private static int MAX_FOLDS=10;
    private double[] paras;
    String trainPath="";
    boolean debug=false;
    Random rng;
    ArrayList<Double> accuracy;
    private boolean kernelOptimise=false;   //Choose between linear, quadratic and RBF kernel
    private boolean paraOptimise=true;
    private ClassifierResults res =new ClassifierResults();
    public TunedSVM(){
        kernelOptimise=false;
        kernel=KernelType.RBF;
        paraOptimise=true;
        setKernel(new RBFKernel());
        rng=new Random();
       accuracy=new ArrayList<>();
         
    }

    public void setSeed(int s){
        this.setSeed=true;
        seed=s;
        rng=new Random();
        rng.setSeed(seed);
    }
    
 @Override
    public void writeCVTrainToFile(String train) {
        trainPath=train;
    }    
//Think this always does para search?
//    @Override
//    public boolean findsTrainAccuracyEstimate(){ return findTrainAcc;}
    
    @Override
    public ClassifierResults getTrainResults(){
//Temporary : copy stuff into res.acc here
        return res;
    }     
    @Override
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",CVAcc,"+res.acc+",C,"+paras[0];
        if(paras.length>1)
            result+=",Gamma,"+paras[1];
       for(double d:accuracy)
            result+=","+d;
        
        return result;
    }
    
    
    public enum KernelType {LINEAR,QUADRATIC,RBF};
    KernelType kernel;
    public void debug(boolean b){
        this.debug=b;
    }

    public void setKernelType(KernelType type) {
        switch (type) {
            case LINEAR:                     
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                setKernel(p);
            break;
            case QUADRATIC:
                PolyKernel p2=new PolyKernel();
                p2.setExponent(2);
                setKernel(p2);
            break;
            case RBF:
                RBFKernel kernel2 = new RBFKernel();
                setKernel(kernel2);
            break;
        }
    }
    
    public void setParaSpace(double[] p){
        paraSpace=p;
    }
    public void setStandardParas(){
        paraSpace=new double[max-min+1];
        for(int i=min;i<=max;i++)
            paraSpace[i-min]=Math.pow(2,i);
    }
    public void optimiseKernel(boolean b){kernelOptimise=b;}
    public void optimiseParas(boolean b){paraOptimise=b;}
    
    static class ResultsHolder{
        double x,y;
        ClassifierResults res;
        ResultsHolder(double a, double b,ClassifierResults r){
            x=a;
            y=b;
            res=r;
        }
    }

    public void tuneRBF(Instances train) throws Exception {
        paras=new double[2];
        int folds=MAX_FOLDS;
        if(folds>train.numInstances())
            folds=train.numInstances();

        double minErr=1;
        this.setSeed(rng.nextInt());
        ArrayList<ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;
        for(double p1:paraSpace){
            for(double p2:paraSpace){
                SMO model = new SMO();
                RBFKernel kern = new RBFKernel();
                kern.setGamma(p2);
                model.setKernel(kern);
                model.setC(p1);
//                model.setBuildLogisticModels(true);
                model.setBuildCalibrationModels(true);
                CrossValidator cv = new CrossValidator();
                if (setSeed)
                  cv.setSeed(seed);
                cv.setNumFolds(folds);
                Instances temp=new Instances(train);//Just in case ...
                tempResults=cv.crossValidateWithStats(model,temp);

//                Evaluation eval=new Evaluation(temp);
//                eval.crossValidateModel(model, temp, folds, rng);
                double e=1-tempResults.acc;
                accuracy.add(tempResults.acc);

                if(debug)
                    System.out.println(" C= "+p1+" Gamma = "+p2+" Acc = "+(1-e));
                if(e<minErr){
                    minErr=e;
                    ties=new ArrayList<>();//Remove previous ties
                    ties.add(new ResultsHolder(p1,p2,tempResults));
                }
                else if(e==minErr){//Sort out ties
                        ties.add(new ResultsHolder(p1,p2,tempResults));
                }
            }
        }
        
        double bestC;
        double bestSigma;
        ResultsHolder best=ties.get(rng.nextInt(ties.size()));
        bestC=best.x;
        bestSigma=best.y;
        res=best.res;
        paras[0]=bestC;
        setC(bestC);
        ((RBFKernel)m_kernel).setGamma(bestSigma);
        paras[1]=bestSigma;
        if(debug)
            System.out.println("Best C ="+bestC+" best Gamma = "+bestSigma+" best train acc = "+res.acc);
    }
    
   public void tuneQuadratic(Instances train) throws Exception {
        paras=new double[1];
        int folds=MAX_FOLDS;
        if(folds>train.numInstances())
            folds=train.numInstances();
        double minErr=1;
        this.setSeed(rng.nextInt());
        ArrayList<ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;
        
        for(double d: paraSpace){
            Instances temp=new Instances(train);
            SMO model = new SMO();
            model.setKernel(m_kernel);
            model.setC(d);
//            model.setBuildLogisticModels(true);
            model.setBuildCalibrationModels(true);
            CrossValidator cv = new CrossValidator();
            if (setSeed)
              cv.setSeed(seed);
            cv.setNumFolds(folds);
            tempResults=cv.crossValidateWithStats(model,temp);

//                Evaluation eval=new Evaluation(temp);
//                eval.crossValidateModel(model, temp, folds, rng);
            double e=1-tempResults.acc;
            accuracy.add(tempResults.acc);

            if(e<minErr){
                minErr=e;
               ties=new ArrayList<>();//Remove previous ties
                ties.add(new ResultsHolder(d,0.0,tempResults));
            }
            else if(e==minErr){//Sort out ties
                ties.add(new ResultsHolder(d,0.0,tempResults));
            }
        }
        ResultsHolder best=ties.get(rng.nextInt(ties.size()));
        setC(best.x);
        res=best.res;
    }
     
    public void selectKernel(Instances train) throws Exception {
        KernelType[] ker=KernelType.values();
        double[] rbfParas=new double[2];
        double rbfCVAcc=0;
        double linearBestC=0;
        double linearCVAcc=0;
        double quadraticBestC=0;
        double quadraticCVAcc=0;
        for(KernelType k:ker){
            TunedSVM temp=new TunedSVM();
            Kernel kernel;
            switch(k){
                case LINEAR:                     
                    PolyKernel p=new PolyKernel();
                    p.setExponent(1);

                    temp.setKernel(p);
                    temp.tuneQuadratic(train);
                    linearCVAcc=temp.res.acc;
                    linearBestC=temp.getC();
                break;
                case QUADRATIC:
                    PolyKernel p2=new PolyKernel();
                    p2.setExponent(2);
                    temp.setKernel(p2);
                    temp.tuneQuadratic(train);
                    quadraticCVAcc=temp.res.acc;
                    quadraticBestC=temp.getC();
                break;
                case RBF:
                    RBFKernel kernel2 = new RBFKernel();
                    temp.setKernel(kernel2);
                    temp.tuneRBF(train);
                    rbfCVAcc=temp.res.acc;
                    rbfParas[0]=temp.getC();
                    rbfParas[1]=((RBFKernel)temp.m_kernel).getGamma();
                    break;
            }
        }
//Choose best, ineligantly
        if(linearCVAcc> rbfCVAcc && linearCVAcc> quadraticCVAcc){//Linear best
            PolyKernel p=new PolyKernel();
            p.setExponent(1);
            setKernel(p);
            setC(linearBestC);
            paras=new double[1];
            paras[0]=linearBestC;
            res.acc=linearCVAcc;
        }else if(quadraticCVAcc> linearCVAcc && quadraticCVAcc> rbfCVAcc){ //Quad best
            PolyKernel p=new PolyKernel();
            p.setExponent(2);
            setKernel(p);
            setC(quadraticBestC);
            paras=new double[1];
            paras[0]=quadraticBestC;
            res.acc=quadraticCVAcc;
        }else{   //RBF
            RBFKernel kernel = new RBFKernel();
            kernel.setGamma(rbfParas[1]);
            setKernel(kernel);
            setC(rbfParas[0]);
            paras=rbfParas;
            res.acc=rbfCVAcc;
        }
    }
    
    
    @Override
    public void buildClassifier(Instances train) throws Exception {
        res.buildTime=System.currentTimeMillis();
        if(paraSpace==null)
            setStandardParas();
        
        if(kernelOptimise)
            selectKernel(train);
        else if(paraOptimise){
            if(this.getKernel() instanceof RBFKernel)
                tuneRBF(train); //Does MORE CV 
            else
                tuneQuadratic(train);
        }
        
        super.buildClassifier(train);
        
        res.buildTime=System.currentTimeMillis()-res.buildTime;
        if(trainPath!=null && trainPath!=""){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(train.relationName()+",TunedSVM,Train");
            f.writeLine(getParameters());
            f.writeLine(res.acc+"");
            f.writeLine(res.writeInstancePredictions());
        }        
    }
    public static void main(String[] args) {
        String sourcePath="C:\\Users\\ajb\\Dropbox\\UCI Problems\\";
        String problemFile="monks-1";
        DecimalFormat df = new DecimalFormat("###.###");
        Instances all=ClassifierTools.loadData(sourcePath+problemFile+"/"+problemFile);
        Instances[] split=InstanceTools.resampleInstances(all,0,0.5);
        Instances train=split[0];
        Instances test=split[1];
        try{
            TunedSVM svml=new TunedSVM();
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(1);
            svml.setKernel(kernel);
            TunedSVM svmq=new TunedSVM();
            kernel = new PolyKernel();
            kernel.setExponent(2);
            svmq.setKernel(kernel);
            TunedSVM svmrbf=new TunedSVM();
            RBFKernel kernel2 = new RBFKernel();
            kernel2.setGamma(1/(double)(all.numAttributes()-1));
            svmrbf.setKernel(kernel2);
           svml.buildClassifier(train);
            System.out.println("BUILT LINEAR = "+svml);
            System.out.println(" Optimal C ="+svml.getC());
             
            svmq.buildClassifier(train);
            System.out.println("BUILT QUAD");
            System.out.println(" Optimal C ="+svmq.getC());
           svmrbf.buildClassifier(train);
            System.out.println("BUILT RBF");
            System.out.println(" Optimal C ="+svmrbf.getC());
            double accL=0,accQ=0,accRBF=0;
            accL=ClassifierTools.accuracy(test, svml);
           accQ=ClassifierTools.accuracy(test, svmq);
           accRBF=ClassifierTools.accuracy(test,svmrbf);

            System.out.println("ACC on "+problemFile+": Linear = "+df.format(accL)+", Quadratic = "+df.format(accQ)+", RBF = "+df.format(accRBF));
                
         }catch(Exception e){
            System.out.println(" Exception builing a classifier = "+e);
            System.exit(0);
        }
    }

    
    protected class PolynomialKernel extends PolyKernel{
        double a=0; //Parameter
    @Override   
    protected double evaluate(int id1, int id2, Instance inst1)
        throws Exception {

        double result;
        if (id1 == id2) {
          result = dotProd(inst1, inst1);
        } else {
          result = dotProd(inst1, m_data.instance(id2));
        }
//Only change from base class        
        result += a;
        if (m_exponent != 1.0) {
          result = Math.pow(result, m_exponent);
        }
        return result;
      }

    
}
    
}
