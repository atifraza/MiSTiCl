/**
*   Input format: ARFF files. 
* Either a single file, 
* *    problemPath/problemName/problemName.arff
which is randomly split by propInTrain
* or a train/test file
* *    problemPath/problemName/problemName_TRAIN.arff
* *    problemPath/problemName/problemName_TEST.arff
*  in which case the zero fold is as in file, other folds are resampled with 
* the same train/test splits.  
* 
* Perform an experiment. The base operation is building a classifier on a problem file with a given fold. 
* 
*/
package development;

import vector_classifiers.RandomRotationForest1;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import vector_classifiers.TunedSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.TunedRotationForest;
import vector_classifiers.HESCA;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import weka.classifiers.trees.RandomForest;
import vector_classifiers.TunedRandomForest;
import weka.core.Instances;


public class ICDMRandomRotationForest{
    static boolean debug=true;
    public static String[] classifiers={"RotF","RandRotF"};
    public static double propInTrain=0.5;
    public static int folds=30; 

    
    static String[] UCIContinuousFileNames={"abalone","acute-inflammation","acute-nephritis","adult","annealing","arrhythmia","audiology-std","balance-scale","balloons","bank","blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car","cardiotocography-10clases","cardiotocography-3clases",
        "chess-krvk","chess-krvkp","congressional-voting","conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding",
        "connect-4","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli","energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography","magic","mammographic",
        "miniboone","molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks","parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor","ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit","statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe","titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases","wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo"};
    
    String problemPath=""; //The root location of the problem files. 
    String resultsPath="";  //The directory where results will be written. If results are already there, nothing will happen
    
    static String[] files=UCIContinuousFileNames;
/*
Section ? Table ?
    
 Can we better scale Rotation Forest for large feature spaces?    
 
This function performs all the resamples in a single thread. It is written
this way for ease of comprehension, but in reality we distributed each resample
   
    To run a single problem/classifier/fold combination, 
    
    do this for UCI
    
    or this for UCR-UEA
    
    
    */    
    public static void randomRotationForest1(String problem, boolean singleFile){
        if(singleFile){//one ARFF with all the data as with UCI data
            
        }
        else{ //Train/Test ARFF as with UCR-UEA data
            
        }
            
        
    }

    public static Classifier setClassifier(String classifier, int fold){
//RandF or RotF
        TunedRandomForest randF;
        TunedRotationForest r;
        switch(classifier){
//Full Rotation Forest with no tuning            
           case "RotF":
                r=new TunedRotationForest();
                r.setNumIterations(200);
                r.justBuildTheClassifier();
                return r;
            case "RandRotF1":
//Full Rotation Forest with no tuning            
                RandomRotationForest1 r3=new RandomRotationForest1();
                r3.setNumIterations(200);
                r3.setMaxNumAttributes(100);
                r3.justBuildTheClassifier();
                return r3;
                
                
            case "RotFCV":
                r = new TunedRotationForest();
                r.setNumIterations(200);
                r.tuneFeatures(false);
                r.tuneTree(false);
                r.estimateAccFromTrain(true);
                return r;
            case "RandFCV":
                randF = new TunedRandomForest();
                randF.tuneTree(false);
                randF.tuneFeatures(false);
                randF.setNumIterations(500);
                randF.debug(debug);
                randF.setSeed(fold);
                randF.setTrainAcc(true);
                randF.setCrossValidate(true);
                return randF;
            case "RandFOOB":
                randF = new TunedRandomForest();
                randF.tuneTree(false);
                randF.tuneFeatures(false);
                randF.setNumIterations(500);
                randF.debug(debug);
                randF.setSeed(fold);
                randF.setTrainAcc(true);
                randF.setCrossValidate(false);
                return randF; 

            default:
            throw new RuntimeException("Unknown classifier = "+classifier+" in Feb 2017 class");
        }
    }
    public static void singleClassifierAndFoldSingleDataSet(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=ICDMRandomRotationForest.setClassifier(classifier,fold);
        Instances all=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem);
        all.randomize(new Random());
        
        Instances[] split=InstanceTools.resampleInstances(all, fold, propInTrain);
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions"+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdirs();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
            if(c instanceof TrainAccuracyEstimate)
                ((TrainAccuracyEstimate)c).writeCVTrainToFile(predictions+"/trainFold"+fold+".csv");
            if(c instanceof HESCA){
                System.out.println("Turning on file read ");
                  ((HESCA)c).setResultsFileLocationParameters(DataSets.resultsPath, problem, fold);
                  ((HESCA)c).setBuildIndividualsFromResultsFiles(true);
            }
            double acc =singleClassifierAndFoldSingleDataSet(split[0],split[1],c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
            
 //       of.writeString("\n");
        }
    }
    
    public static double singleClassifierAndFoldSingleDataSet(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        double acc=0;
        int act;
        int pred;
// Save internal info for ensembles
//        if(c instanceof SaveableEnsemble)
//           ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
        OutFile p=null;
        try{              
            c.buildClassifier(train);
            StringBuilder str = new StringBuilder();
            DecimalFormat df=new DecimalFormat("##.######");
            for(int j=0;j<test.numInstances();j++)
            {
                act=(int)test.instance(j).classValue();

                test.instance(j).setClassMissing();//Just in case ....
                double[] probs=c.distributionForInstance(test.instance(j));
                pred=0;
                for(int i=1;i<probs.length;i++){
                    if(probs[i]>probs[pred])
                        pred=i;
                }
                if(act==pred)
                    acc++;
                str.append(act);
                str.append(",");
                str.append(pred);
                str.append(",,");
                for(double d:probs){
                    str.append(df.format(d));
                    str.append(",");
                }
                str.append("\n");
            }
            acc/=test.numInstances();
           
            p=new OutFile(resultsPath+"/testFold"+fold+".csv");
            if(p==null) throw new Exception(" file wont open!! "+resultsPath+"/testFold"+fold+".csv");
            p.writeLine(train.relationName()+","+c.getClass().getName()+",test");
            if(c instanceof SaveParameterInfo){
              p.writeLine(((SaveParameterInfo)c).getParameters());
            }else
                p.writeLine("No parameter info");
            p.writeLine(acc+"");
            p.writeLine(str.toString());
        }catch(Exception e)
        {
                e.printStackTrace();
                System.out.println(" Error ="+e+" in method singleClassifierAndFold in class Feb2017");
                System.out.println(" Classifier = "+c.getClass().getName());
                System.out.println(" Results path="+resultsPath);
                System.out.println(" Outfile = "+p);
                System.out.println(" Train Split = "+train.toSummaryString());
                System.out.println(" Test Split = "+test.toSummaryString());
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }    
    
/** Run a given classifier/problem/fold combination with associated file set up
 @param args: 
 * args[0]: Classifier name. Create classifier with setClassifier
 * args[1]: Problem name
 * args[2]: Fold number. This is assumed to range from 1, hence we subtract 1
 * (this is because of the scripting we use to run the code on the cluster)
 *          the standard archive folds are always fold 0
 * 
 * NOTES: 
 * 1. this assumes you have set DataSets.problemPath to be where ever the 
 * data is, and assumes the data is in its own directory with two files, 
 * args[1]_TRAIN.arff and args[1]_TEST.arff 
 * 2. assumes you have set DataSets.resultsPath to where you want the results to
 * go It will NOT overwrite any existing results (i.e. if a file of non zero 
 * size exists)
 * 3. This method just does the file set up then calls the next method. If you 
 * just want to run the problem, go to the next method
* */
    public static void singleClassifierAndFoldTrainTestSplit(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=setClassifier(classifier,fold);
        Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
            if(c instanceof TrainAccuracyEstimate)
                ((TrainAccuracyEstimate)c).writeCVTrainToFile(predictions+"/trainFold"+fold+".csv");
            double acc =singleClassifierAndFoldTrainTestSplit(train,test,c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
            
 //       of.writeString("\n");
        }
    }
/**
 * 
 * @param train: the standard train fold Instances from the archive 
 * @param test: the standard test fold Instances from the archive
 * @param c: Classifier to evaluate
 * @param fold: integer to indicate which fold. Set to 0 to just use train/test
 * @param resultsPath: a string indicating where to store the results
 * @return the accuracy of c on fold for problem given in train/test
 * 
 * NOTES:
 * 1.  If the classifier is a SaveableEnsemble, then we save the internal cross 
 * validation accuracy and the internal test predictions
 * 2. The output of the file testFold+fold+.csv is
 * Line 1: ProblemName,ClassifierName, train/test
 * Line 2: parameter information for final classifier, if it is available
 * Line 3: test accuracy
 * then each line is
 * Actual Class, Predicted Class, Class probabilities 
 * 
 * 
 */    
    public static double singleClassifierAndFoldTrainTestSplit(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
        double acc=0;
        int act;
        int pred;
// Save internal info for ensembles
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
        try{              
            c.buildClassifier(data[0]);
            if(debug){
                if(c instanceof RandomForest)
                    System.out.println(" Number of features in MAIN="+((RandomForest)c).getNumFeatures());
            }
            StringBuilder str = new StringBuilder();
            DecimalFormat df=new DecimalFormat("##.######");
            for(int j=0;j<data[1].numInstances();j++)
            {
                act=(int)data[1].instance(j).classValue();
                data[1].instance(j).setClassMissing();//Just in case ....
                double[] probs=c.distributionForInstance(data[1].instance(j));
                pred=0;
                for(int i=1;i<probs.length;i++){
                    if(probs[i]>probs[pred])
                        pred=i;
                }
                if(act==pred)
                    acc++;
                str.append(act);
                str.append(",");
                str.append(pred);
                str.append(",,");
                for(double d:probs){
                    str.append(df.format(d));
                    str.append(",");
                }
                str.append("\n");
            }
            acc/=data[1].numInstances();
            OutFile p=new OutFile(resultsPath+"/testFold"+fold+".csv");
            p.writeLine(train.relationName()+","+c.getClass().getName()+",test");
            if(c instanceof SaveParameterInfo){
              p.writeLine(((SaveParameterInfo)c).getParameters());
            }else
                p.writeLine("No parameter info");
            p.writeLine(acc+"");
            p.writeLine(str.toString());
        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }    


    
    
    public static void collateResults(){
//3. Merge classifier files into a single file with average accuracies
        //NEED TO REWRITE FOR TRAIN TEST DIFF
        String basePath="C:\\Users\\ajb\\Dropbox\\Results\\Forest\\";
        OutFile acc=new OutFile(basePath+"CombinedAcc.csv");
        for(String cls:classifiers){
            acc.writeString(","+cls);
        }
        acc.writeString("\n");
        InFile[] allTest=new InFile[classifiers.length];
        for(int i=0;i<allTest.length;i++){
            String p=basePath+classifiers[i]+"Test.csv";
            if(new File(p).exists()){
                allTest[i]=new InFile(p);
//                System.out.println("File "+p+" opened ok");
            }
            else
                allTest[i]=null;//superfluous
//             p=basePath+classifiers[i]+"//"+classifiers[i]+"Train.csv";
        }
        for(int i=0;i<files.length;i++){
            acc.writeString(files[i]+",");
            String prev="First";
            for(int j=0;j<allTest.length;j++){
                if(allTest[j]==null){
                    acc.writeString(",");
                }
                else{//Find mean
                    try{
                        String r=allTest[j].readLine();
                        String[] res=r.split(",");
                        double mean=0;
                        for(int k=1;k<res.length;k++){
                            mean+=Double.parseDouble(res[k]);
                        }
                        if(res.length>1){
                            acc.writeString((mean/(res.length-1))+",");
                        }
                        else{
                            acc.writeString(",");
                        }
                        prev=r;
                    }catch(Exception ex){
                        System.out.println("failed to read line: "+ex+" previous line = "+prev);
                        System.exit(0);
                    }
                }
            }
            acc.writeString("\n");
        }
        for(InFile  inf:allTest)
            if(inf!=null)
                inf.closeFile();
        
    }
/**
 * nos cases, nos features, nos classes, nos cases/**
 * nos cases, nos features, nos classes, nos cases
 */   

    public static void summariseData(){
        
        OutFile out=new OutFile(DataSets.problemPath+"SummaryInfo.csv");
        out.writeLine("problem,numCases,numAtts,numClasses");
        for(String str:files){
            File f=new File(DataSets.problemPath+str+"/"+str+".arff");
            if(f.exists()){
                Instances ins=ClassifierTools.loadData(DataSets.problemPath+str+"/"+str);
                out.writeLine(str+","+ins.numInstances()+","+(ins.numAttributes()-1)+","+ins.numClasses());
            }
            else
                out.writeLine(str+",,");
        }
    }
    public static void collateSVMParameters(){
        InFile c=new InFile("C:\\Research\\Papers\\2017\\ECML Standard Parameters\\Section 6 choosing parameters\\TunedSVMParameterC.csv");
        InFile g=new InFile("C:\\Research\\Papers\\2017\\ECML Standard Parameters\\Section 6 choosing parameters\\TunedSVMParameterGamma.csv");
        int[][] counts=new int[25][25];
        double[] vals={0.000015,0.000031,0.000061,0.000122,0.000244,0.000488,0.000977,0.001953,0.003906,0.007813,0.015625,0.031250,0.062500,0.125000,0.250000,0.500000,1.000000,2.000000,4.000000,8.000000,16.000000,32.000000,64.000000,128.000000,256.000000};
        for(int i=0;i<files.length;i++){
            String line=c.readLine();
            String gLine=g.readLine();
            String[] splitC=line.split(",");
            String[] splitG=gLine.split(",");
            System.out.print("\n Problem="+splitC[0]);
            int cPos=0,gPos;
            for(int j=1;j<splitC.length;j++){
                if(!splitC[j].equals("")){
                    //Look up
                    int k=0;
                    double v=Double.parseDouble(splitC[j]);
                    try{
                    while(vals[k]!=v)
                        k++;
                    cPos=k;
                    }catch(Exception e){
                        System.out.println(" EXCEPTION : ="+e+" v = "+v+" k="+k);
                    }
                    k=0;
                    v=Double.parseDouble(splitG[j]);
                    while(vals[k]!=v)
                        k++;
                    gPos=k;
                    counts[cPos][gPos]++;
                    
//                    System.out.print("c Pos="+cPos+" G pos ="+gPos);
                }
            }
        }
        OutFile svm=new OutFile("C:\\Research\\Papers\\2017\\ECML Standard Parameters\\Section 6 choosing parameters\\svmParaCounts.csv");
        for(int i=0;i<counts.length;i++){
            for(int j=0;j<counts[i].length;j++)
                svm.writeString(counts[i][j]+",");
            svm.writeString("\n");
        }
        
    }
    public static void timingExperiment(String classifier){
//Test times
        
        
    }
    
    public static void main(String[] args) throws Exception{
      boolean ucrData=true;
       files=DataSets.fileNames;
 //      collateResults(30,true,args);
//UCIRotFTimingExperiment();
  //             System.exit(0);
 //       collateTrain();

      
      classifiers=new String[]{"RotF","RandRotF1"};
        String dir="RepoScripts";
        String jarFile="Repo";
    System.exit(0);

//        collateTrainTestResults(30);

        if(ucrData)
            runTSCDataSet(args);
        else
            runUCIDataSet(args);
    }

    
    public static void runTSCDataSet(String[] args) {
        if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"TSCProblems/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/RepoResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            ICDMRandomRotationForest.singleClassifierAndFoldTrainTestSplit(args);
        }
        else{
            DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/RepoResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }

            String[] paras={"RandFCV","ItalyPowerDemand","1"};
//            paras[0]="RotFCV";
//            paras[2]="1";
            ICDMRandomRotationForest.singleClassifierAndFoldTrainTestSplit(paras);            
            long t1=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                ICDMRandomRotationForest.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t2=System.currentTimeMillis();
            paras[0]="RandFOOB";
            ICDMRandomRotationForest.singleClassifierAndFoldSingleDataSet(paras);            
            long t3=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                ICDMRandomRotationForest.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t4=System.currentTimeMillis();
            System.out.println("Standard = "+(t2-t1)+", Enhanced = "+(t4-t3));
            
       }        
    }
    
    
    
    public static void runUCIDataSet(String[] args) {
        if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"UCIContinuous/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/UCIResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            ICDMRandomRotationForest.singleClassifierAndFoldSingleDataSet(args);
        }
        else{
            DataSets.problemPath=DataSets.dropboxPath+"UCI Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/UCIResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }

            String[] paras={"","semeion","1"};
            DataSets.problemPath="C:/Data/UCI Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/UCIResults/";
            File file =new File("C:\\Users\\ajb\\Dropbox\\Results\\UCIResults");
            paras[0]="RotFCV";
            paras[2]="1";
            ICDMRandomRotationForest.singleClassifierAndFoldSingleDataSet(paras);            
            long t1=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                ICDMRandomRotationForest.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t2=System.currentTimeMillis();
            paras[0]="EnhancedRotF";
            ICDMRandomRotationForest.singleClassifierAndFoldSingleDataSet(paras);            
            long t3=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                ICDMRandomRotationForest.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t4=System.currentTimeMillis();
            System.out.println("Standard = "+(t2-t1)+", Enhanced = "+(t4-t3));
            
       }        
    }


    public static void UCIRotFTimingExperiment() throws Exception{
//Restrict to those with over 40 attributes
        OutFile times=new OutFile("c:/temp/RotFUCITimes.csv");
        for(String problem:UCIContinuousFileNames){
//See whether we want to do this one
            if(problem.equals("miniboone")||problem.equals("connect-4"))
                continue;
            Instances inst=ClassifierTools.loadData("C:/Data/UCIContinuous/"+problem+"/"+problem);
            
            if(inst.numAttributes()-1>40){

                System.out.println(" Problem "+problem+" has "+(inst.numAttributes()-1)+" number of attributes");
                times.writeString(problem+","+(inst.numAttributes()-1)+","+(inst.numInstances())+",");
                RotationForest rot1=new RotationForest();
                rot1.setNumIterations(200);
                RandomRotationForest1 rot2=new RandomRotationForest1();
                rot2.setNumIterations(200);
                rot2.tuneFeatures(false);
                rot2.tuneTree(false);
                rot2.estimateAccFromTrain(false);
//Identical apart from this            
                rot2.setMaxNumAttributes(40);
                long t1=System.currentTimeMillis();
                rot1.buildClassifier(inst);
                long t2=System.currentTimeMillis();
                System.out.println(" Full RotF time = "+((t2-t1)/1000));
                times.writeString((t2-t1)+",");
                t1=System.currentTimeMillis();
                rot2.buildClassifier(inst);
                t2=System.currentTimeMillis();
                System.out.println(" truncated RotF time = "+((t2-t1)/1000));
                times.writeLine((t2-t1)+",");
                
                
                
            }            
        }
        
    }


    public static void UCRRotFTimingExperiment() throws Exception{
//Restrict to those with over 40 attributes
        OutFile times=new OutFile("c:/temp/RotFUCITimes.csv");
        for(String problem:DataSets.fileNames){
//See whether we want to do this one
            Instances inst=ClassifierTools.loadData("C:/Data/TSC Problems/"+problem+"/"+problem+"_TRAIN");
            if(problem.equals("HandOutlines"))
                continue;
            
            if(inst.numAttributes()-1>100){

                System.out.println(" Problem "+problem+" has "+(inst.numAttributes()-1)+" number of attributes");
                times.writeString(problem+","+(inst.numAttributes()-1)+","+(inst.numInstances())+",");
                RotationForest rot1=new RotationForest();
                rot1.setNumIterations(200);
                RandomRotationForest1 rot2=new RandomRotationForest1();
                rot2.setNumIterations(200);
                rot2.tuneFeatures(false);
                rot2.tuneTree(false);
                rot2.estimateAccFromTrain(false);
//Identical apart from this            
                rot2.setMaxNumAttributes(100);
                long t1=System.currentTimeMillis();
                rot1.buildClassifier(inst);
                long t2=System.currentTimeMillis();
                System.out.println(" Full RotF time = "+((t2-t1)/1000));
                times.writeString((t2-t1)+",");
                t1=System.currentTimeMillis();
                rot2.buildClassifier(inst);
                t2=System.currentTimeMillis();
                System.out.println(" truncated RotF time = "+((t2-t1)/1000));
                times.writeLine((t2-t1)+",");
                
                
                
            }            
        }
        
    }



}

