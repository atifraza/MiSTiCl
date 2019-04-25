/*
class to collate standard results files over multiple classifiers and problems

Usage 
(assuming Collate.jar has this as the main class): 
java -jar Collate.jar ResultsDirectory/ ProblemDirectory/ NumberOfFolds Classifier1 Classifier2 .... ClassifierN NoParasC1 NoParasC2 .... NoParasCn
e.g. java -jar -Xmx6000m Collate.jar Results/ UCIContinuous/ 30 RandF RotF 2 2 

collates the results for 30 folds for RandF and RotF in the directory for Results 
on all the problems in UCIContinous (as defined by having a directory in the folder)

Stage 1: take all the single fold files, work out the diagnostics on test data: 
Accuracy, BalancedAccuracy, NegLogLikelihood, AUROC and F1 and store the TrainCV accuracy. 
all done by call to collateFolds();
Combine folds into a single file for each statistic in ResultsDirectory/ClassifierName
these are
Counts: counts.csv, number per problem (max number is NumberOfFolds, it does not check for more).
Diagnostics: TestAcc.csv, TestF1.csv, TestBAcc.csv, TestNLL.csv, TestAUROC.csv, TrainCVAcc.csv
Timings: Timings.csv
Parameter info: Parameter1.csv, Parameter2.csv...AllTuningAccuracies.csv (if tuning occurs, all tuning values).

Stage 2: 
Output: Classifier Summary: call to method averageOverFolds() 
Creates average and standard deviation over all folds based on the files created at stage 1 with the addition of the mean difference
per fold.
All put in a single directory.

Stage 3
Final Comparison Summary: call to method basicSummaryComparisons();        
a single file in ResultsDirectory directory called summaryTests<ClassifierNames>.csv
contains pairwise comparisons of all the classifiers. 

1. All Pairwise Comparisons for TestAccDiff, TestAcc, TestBAcc, TestNLL.csv and TestAUROC

1. Wins/Draws/Loses
2. Mean (and std dev) difference
3. Two sample tests of the mean values 







 */
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import statistics.distributions.BinomialDistribution;
import statistics.tests.OneSampleTests;
import utilities.ClassifierResults;

/**
 *
 * @author ajb
 */
public class CollateResults {
    public static File[] dir;
    static String basePath;
    static String[] classifiers;
    static ArrayList<String> problems;
    static String problemPath;
    static int folds;
    static int numClassifiers;
    static int[] numParas;
    static DecimalFormat df=new DecimalFormat("##.######");
    static double[][] data;
  
/** 
 * Arguments: 
1. ResultsDirectory/ 
2. Either ProblemDirectory/ or ProblemFiles.csv or ProblemFiles.txt 
*                       Basically checks for an extension and if its there reads a file. 
* 
3. NumberOfFolds 
4-4+nosClassifiers                          Classifier1 Classifier2 .... ClassifierN 
4+nosClassifiers+1 to 4+2*nosClassifiers    NoParasC1 NoParasC2 .... NoParasCn
 * 
 */
    public static void readData(String[] args){
        basePath=args[0];
        System.out.println("Base path = "+basePath);
        problemPath=args[1];
        System.out.println("Problem path = "+problemPath);
         folds =Integer.parseInt(args[2]);
        System.out.println("Folds = "+folds);
        numClassifiers=(args.length-3)/2;
        classifiers=new String[numClassifiers];
        for(int i=0;i<classifiers.length;i++)
            classifiers[i]=args[i+3];
        numParas=new int[classifiers.length];
        for(int i=0;i<classifiers.length;i++)
            numParas[i]=Integer.parseInt(args[i+3+classifiers.length]);
//Get problem files
        File f=new File(problemPath);
            problems=new ArrayList<>();
        if(problemPath.contains(".txt") || problemPath.contains(".csv")){//Read from file
            if(!f.exists())
                System.out.println("Error loading problems  from file ="+problemPath);
            else{
                InFile inf=new InFile(problemPath);
                String prob=inf.readLine();
                while(prob!=null){
                    problems.add(prob);
                    prob=inf.readLine();
                }
            }
        }
        else{
            if(!f.isDirectory()){
                System.out.println("Error in problem path ="+problemPath);
            }

            dir=f.listFiles();
            for(File p:dir){
                if(p.isDirectory()){
                    problems.add(p.getName());
                }
            }
        }
        Collections.sort(problems);

    }    
 
/**
 * Stage 1: take all the single fold files, work out the diagnostics on test data: 
Accuracy, BalancedAccuracy, NegLogLikelihood, AUROC and F1 and store the TrainCV accuracy. 
all done by call to collateFolds();
Combine folds into a single file for each statistic in ResultsDirectory/ClassifierName
these are
Counts: counts.csv, number per problem (max number is NumberOfFolds, it does not check for more).
Diagnostics: TestAcc.csv, TestF1.csv, TestBAcc.csv, TestNLL.csv, TestAUROC.csv, TrainCVAcc.csv
Timings: Timings.csv
Parameter info: Parameter1.csv, Parameter2.csv...AllTuningAccuracies.csv (if tuning occurs, all tuning values).

 */    
    public static void collateFolds(){
//        String[] allStats={"TestAcc","TrainCVAcc","TestNLL","TestBACC","TestAUROC","TestF1"};      

        for(int i=0;i<classifiers.length;i++){
            String cls=classifiers[i];
            System.out.println("Processing classifier ="+cls);
            File f=new File(basePath+cls);
            if(f.isDirectory()){ //Check classifier directory exists. 
//Write collated results for this classifier to a single file                
                OutFile clsResults=new OutFile(basePath+cls+"/"+cls+"TestAcc.csv");
                OutFile f1Results=new OutFile(basePath+cls+"/"+cls+"TestF1.csv");
                OutFile BAccResults=new OutFile(basePath+cls+"/"+cls+"TestBAcc.csv");
                OutFile nllResults=new OutFile(basePath+cls+"/"+cls+"TestNLL.csv");
                OutFile AUROCResults=new OutFile(basePath+cls+"/"+cls+"TestAUROC.csv");
                OutFile trainResults=new OutFile(basePath+cls+"/"+cls+"TrainCVAcc.csv");
                OutFile[] paraFiles=new OutFile[numParas[i]];
                for(int j=0;j<paraFiles.length;j++)
                    paraFiles[j]=new OutFile(basePath+cls+"/"+cls+"Parameter"+(j+1)+".csv");
                OutFile timings=new OutFile(basePath+cls+"/"+cls+"Timings.csv");
                OutFile allAccSearchValues=new OutFile(basePath+cls+"/"+cls+"AllTuningAccuracies.csv");
                OutFile missing=null;
                OutFile counts=new OutFile(basePath+cls+"/"+cls+"Counts.csv");;
                
                int missingCount=0;
                for(String name:problems){            
                    clsResults.writeString(name);
                    trainResults.writeString(name);
                    f1Results.writeString(name);
                    BAccResults.writeString(name);
                    nllResults.writeString(name);
                    AUROCResults.writeString(name);
                    allAccSearchValues.writeString(name);
                    timings.writeString(name);
                    for(OutFile out:paraFiles)
                        out.writeString(name+",");
                    String path=basePath+cls+"//Predictions//"+name;
                    if(missing!=null && missingCount>0)
                        missing.writeString("\n");
                    missingCount=0;
                    for(int j=0;j<folds;j++){
    //Check fold exists
                        f=new File(path+"//testFold"+j+".csv");

                        if(f.exists() && f.length()>0){//This could fail if file only has partial probabilities on the line
    //Read in test accuracy and store                    
    //Check fold exists
    //Read in test accuracy and store
                            InFile inf=null;
                            String[] trainRes=null;
                            try{
                                inf=new InFile(path+"//testFold"+j+".csv");
                                inf.readLine();
                                trainRes=inf.readLine().split(",");//Stores train CV and parameter info
                                clsResults.writeString(","+inf.readDouble());
                                if(trainRes.length>1){//There IS parameter info
                                    //First is train time build
                                    timings.writeString(","+df.format(Double.parseDouble(trainRes[1])));
                                    //second is the trainCV testAcc
                                    if(trainRes.length>3){
                                        trainResults.writeString(","+Double.parseDouble(trainRes[3]));
                                        //Then variable list of numParas
                                        int pos=5;
                                        for(int k=0;k<numParas[i];k++){
                                            if(trainRes.length>pos){
                                                paraFiles[k].writeString(Double.parseDouble(trainRes[pos])+",");
                                                pos+=2;    
                                            }
                                            else
                                                paraFiles[k].writeString(",");
                                        }
    //                                    write the rest to the para search file
                                        while(pos<trainRes.length)
                                            allAccSearchValues.writeString(","+trainRes[pos++]);    
                                    }
                                }
                                else{
                                    trainResults.writeString(",");
                                    for(int k=0;k<numParas[i];k++)
                                        paraFiles[k].writeString(",");
                                }
//Read in the rest into a ClassifierResults object
                                inf.closeFile();
                                ClassifierResults res=new ClassifierResults();
                                res.loadFromFile(path+"//testFold"+j+".csv");
                                res.findAllStats();
                                f1Results.writeString(","+res.f1);
                                BAccResults.writeString(","+res.balancedAcc);
                                nllResults.writeString(","+res.nll);
                                AUROCResults.writeString(","+res.meanAUROC);
                                
                            }catch(Exception e){
                                System.out.println(" Error "+e+" in "+path);
                                if(trainRes!=null){
                                    System.out.println(" second line read has "+trainRes.length+" entries :");
                                    for(String str:trainRes)
                                        System.out.println(str);
                                }
                                System.exit(1);
                            }finally{
                                if(inf!=null)
                                    inf.closeFile();

                            }
                        }
                        else{
                            if(missing==null)
                                missing=new OutFile(basePath+cls+"//"+cls+"MissingFolds.csv");
                            if(missingCount==0)
                                missing.writeString(name);
                            missingCount++;
                           missing.writeString(","+j);
                        }
                    }
                    counts.writeLine(name+","+(folds-missingCount));
                    clsResults.writeString("\n");
                    trainResults.writeString("\n");
                    f1Results.writeString("\n");
                    BAccResults.writeString("\n");
                    nllResults.writeString("\n");
                    AUROCResults.writeString("\n");
                    timings.writeString("\n");
                    allAccSearchValues.writeString("\n");
                    
                    for(int k=0;k<paraFiles.length;k++)
                        paraFiles[k].writeString("\n");
                }
                clsResults.closeFile();
                trainResults.closeFile();
                    for(int k=0;k<paraFiles.length;k++)
                        paraFiles[k].closeFile();
            }
            else
                System.out.println("Classifier "+cls+" has no results directory: "+basePath+cls);
        }
        
    }

/** Stage 2: 
Output: Classifier Summary: call to method averageOverFolds() 
Creates average and standard deviation over all folds based on the files created at stage 1 with the addition of the mean difference
per fold.
All put in a single directory.
* **/
public static void averageOverFolds(){
        
        String name=classifiers[0];
        for(int i=1;i<classifiers.length;i++)
            name+=classifiers[i];
        File nf=new File(basePath+name);
        if(!nf.isDirectory())
            nf.mkdir();
        String[] allStats={"TestAcc","TrainCVAcc","TestNLL","TestBAcc","TestAUROC","TestF1","Timings"};      
        OutFile[] means=new OutFile[allStats.length];
        for(int i=0;i<means.length;i++)
            means[i]=new OutFile(basePath+name+"/"+allStats[i]+name+".csv"); 
        OutFile[] stDev=new OutFile[allStats.length];
        for(int i=0;i<stDev.length;i++)
            stDev[i]=new OutFile(basePath+name+"/"+allStats[i]+"StDev"+name+".csv"); 
        OutFile count=new OutFile(basePath+name+"/Counts"+name+".csv");

//Headers        
        for(int i=0;i<classifiers.length;i++){
            for(OutFile of:means)
                of.writeString(","+classifiers[i]);
            for(OutFile of:stDev)
                of.writeString(","+classifiers[i]);
            count.writeString(","+classifiers[i]);
        }
        for(OutFile of:means)
            of.writeString("\n");
        for(OutFile of:stDev)
            of.writeString("\n");
        count.writeString("\n");
//Do counts first
        InFile[] allClassifiers=new InFile[classifiers.length];
        for(int i=0;i<allClassifiers.length;i++){
            String p=basePath+classifiers[i]+"/"+classifiers[i]+"Counts.csv";
            if(new File(p).exists())
                allClassifiers[i]=new InFile(p);
            else
                allClassifiers[i]=null;//superfluous
        }
        for(String str:problems){
            count.writeString(str);
            for(int i=0;i<allClassifiers.length;i++){
                if(allClassifiers[i]!=null){
                   allClassifiers[i].readString();
                   count.writeString(","+allClassifiers[i].readInt());
                }
                else{
                    count.writeString(",");
                }
            }
            count.writeString("\n");

        }
        
        for(int j=0;j<allStats.length;j++){
//Open files with data for all folds        
            for(int i=0;i<allClassifiers.length;i++){
                String p=basePath+classifiers[i]+"/"+classifiers[i]+allStats[j]+".csv";
                if(new File(p).exists())
                    allClassifiers[i]=new InFile(p);
                else
                    allClassifiers[i]=null;//superfluous
            }
//Find mean for             
            for(String str:problems){
                means[j].writeString(str);
                stDev[j].writeString(str);
                String prev="First";
                for(int i=0;i<allClassifiers.length;i++){
                    if(allClassifiers[i]==null){
                        means[j].writeString(",");
                        stDev[j].writeString(",");
                    }
                    else{//Find mean
                        try{
                            String r=allClassifiers[i].readLine();
                            String[] res=r.split(",");
                            double mean=0;
                            double sumSquare=0;
                            for(int m=1;m<res.length;m++){
                                double d=Double.parseDouble(res[m]);
                                mean+=d;
                                sumSquare+=d*d;
                            }
                            if(res.length>1){
                                int size=(res.length-1);
                                mean=mean/size;
                                double stdDev=sumSquare/size-mean*mean;
                                stdDev=Math.sqrt(stdDev);
                                means[j].writeString(","+df.format(mean));
                                stDev[j].writeString(","+df.format(stdDev));
                            }
                            else{
                                means[j].writeString(",");
                                stDev[j].writeString(",");
                            }
                            prev=r;
                        }catch(Exception ex){
                            System.out.println("failed to read line: "+ex+" previous line = "+prev+" file index ="+j+" classifier index ="+i);
                        }
                    }        
                }               
                means[j].writeString("\n");
                stDev[j].writeString("\n");
                if(j==0)
                    count.writeString("\n");
             }
            for(InFile  inf:allClassifiers)
                if(inf!=null)
                    inf.closeFile();
        }
    }

public static void basicSummaryComparisons(){
//Only compares first two
    DecimalFormat df = new DecimalFormat("###.#####");
    if(classifiers.length<=1)
        return;
    String name=classifiers[0];
    for(int i=1;i<classifiers.length;i++)
        name+=classifiers[i];
    OutFile s=new OutFile(basePath+"summaryTests"+name+".csv");
    String[] allStatistics={"TestAcc","TestBAcc","TestNLL","TestAUROC"};
    data=new double[problems.size()][classifiers.length];
    s.writeLine(name);
    for(String str:allStatistics){
        s.writeLine("**************"+str+"********************");
        System.out.println("Loading "+basePath+name+"/"+str+name+".csv");
        InFile f=new InFile(basePath+name+"/"+str+name+".csv");
        f.readLine();
        for(int i=0;i<problems.size();i++){
            String ss=f.readLine();
            String[] d=ss.split(",");
            for(int j=0;j<classifiers.length;j++)
                data[i][j]=-1; 

            for(int j=0;j<d.length-1;j++){
                    try{
                    double v=Double.parseDouble(d[j+1]);
                    data[i][j]=v;
                    }catch(Exception e){
// yes yes I know its horrible, but this is text parsing, not rocket science
//                            System.out.println("No entry for classifier "+j);
                    }
            }
//                for(int j=0;j<classifiers.length;j++)
//                    System.out.println(" Classifier "+j+" has data "+data[i][j]);       
        }
        for(int x=0;x<classifiers.length-1;x++){
            for (int y=x+1; y < classifiers.length; y++) {//Compare x and y
                int wins=0,draws=0,losses=0;
                int sigWins=0,sigLosses=0;
                double meanDiff=0;
                double sumSq=0;
                double count=0;
                for(int i=0;i<problems.size();i++){
                    if(data[i][x]!=-1 && data[i][y]!=-1){
                        if(data[i][x]>data[i][y])
                            wins++;
                        else if(data[i][x]==data[i][y])
                            draws++;
                        else
                            losses++;
                        meanDiff+=data[i][x]-data[i][y];
                        sumSq+=(data[i][x]-data[i][y])*(data[i][x]-data[i][y]);
                        count++;
                    }
                }
//                    DecimalFormat df = new DecimalFormat("##.#####");
                System.out.println(str+","+classifiers[x]+","+classifiers[y]+",WIN/DRAW/LOSE,"+wins+","+draws+","+losses);  
                BinomialDistribution bin=new BinomialDistribution();
                bin.setParameters(wins+losses,0.5);
                double p=bin.getCDF(wins);
                if(p>0.5) p=1-p;
                s.writeLine(str+","+classifiers[x]+","+classifiers[y]+",WIN/DRAW/LOSE,"+wins+","+draws+","+losses+", p =,"+df.format(p));  
                System.out.println(str+","+classifiers[x]+","+classifiers[y]+",COUNT,"+count+",MeanDiff,"+df.format(meanDiff/count)+",StDevDiff,"+df.format((sumSq-(meanDiff*meanDiff)/count))+" p ="+df.format(p));  
        //3. Find out how many are statistically different within folds
//Do paired T-tests from fold files
            InFile first=new InFile(basePath+classifiers[x]+"/"+classifiers[x]+str+".csv");
            InFile second=new InFile(basePath+classifiers[y]+"/"+classifiers[y]+str+".csv");
            for(int i=0;i<problems.size();i++){
//Read in both: Must be the same number to proceed
                String[] probX=first.readLine().split(",");
                String[] probY=second.readLine().split(",");
                if(probX.length<=folds || probY.length<=folds)
                    continue;   //Skip this problem
                double[] diffs=new double[folds];
                boolean notAllTheSame=false;
                for(int j=0;j<folds;j++){
                    diffs[j]=Double.parseDouble(probX[j+1])-Double.parseDouble(probY[j+1]);
                    if(!notAllTheSame && !probX[j+1].equals(probY[j+1]))
                        notAllTheSame=true;
                }

                if(notAllTheSame){
                    OneSampleTests test=new OneSampleTests();
                    String res=test.performTests(diffs);
                    System.out.println("Results = "+res);
                    String[] results=res.split(",");
                    double tTestPValue=Double.parseDouble(results[2]);
                    if(tTestPValue>=0.95) sigWins++;
                    else if(tTestPValue<=0.05) sigLosses++;
                }
                else
                    System.out.println("**************ALL THE SAME problem = "+probX[0]+" *************");
          }
            s.writeLine(str+","+classifiers[x]+","+classifiers[y]+",SIGWIN/SIGLOSS,"+sigWins+","+sigLosses);  
            System.out.println(str+","+classifiers[x]+","+classifiers[y]+",SIGWIN/SIGLOSS,"+sigWins+","+sigLosses);  
        //2. Overall mean difference
            s.writeLine(str+","+classifiers[x]+","+classifiers[y]+",COUNT,"+count+",MeanDiff,"+df.format(meanDiff/count)+",StDevDiff,"+df.format((sumSq-(meanDiff*meanDiff)/count)));  
            System.out.println(str+","+classifiers[x]+","+classifiers[y]+",COUNT,"+count+",MeanDiff,"+df.format(meanDiff/count)+",StDevDiff,"+df.format((sumSq-(meanDiff*meanDiff)/count)));  
        }


    }

//Do pairwise tests over all common datasets. 
//1.    First need to condense to remove any with one missing
        ArrayList<double[]> res=new ArrayList<>();
        for(int i=0;i<data.length;i++){
            int j=0;
            while(j<data[i].length && data[i][j]!=-1)
                j++;
            if(j==data[i].length)
                res.add(data[i]);
        }
        System.out.println("REDUCED DATA SIZE = "+res.size());

        double[][] d2=new double[res.size()][];
        for(int i=0;i<res.size();i++)
            d2[i]=res.get(i);

 //2. Do pairwise tests       
        StringBuilder resultsString=MultipleClassifiersPairwiseTest.runSignRankTest(d2,classifiers);
        s.writeString(resultsString.toString());
        System.out.println(resultsString);

    }
    s.closeFile();
}
    
    public static void collate(String[] args){
//STAGE 1: Read from arguments, find problems        
        readData(args);
        System.out.println(" number of classifiers ="+numClassifiers);
//STAGE 2: Collate the individual fold files into one        
        collateFolds();
//STAGE 3: Summarise over folds 
        averageOverFolds();
//STAGE 4: Do statical comparisons
        basicSummaryComparisons();        
        
    }
//First argument: String path to results directories
//Second argument: path to directory with problem allStats to look for
//Third argument: number of folds    
//Next x arguments: x Classifiers to collate    
//Next x arguments: number of numParas stored for each classifier    
    public static void main(String[] args) {
        if(args.length>1)
            collate(args);
        else{    
            String[] str={"C:/Research/Results/","C:/Users/ajb/Dropbox/UCI Problems/ProblemTest.csv","30","TunedRandF","TunedRotF","2","2"};
            collate(str);
        
        }
    }
    
}
