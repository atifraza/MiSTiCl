package vector_classifiers;

import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import timeseriesweka.classifiers.ensembles.weightings.TrainAcc;
import timeseriesweka.classifiers.ensembles.weightings.ModuleWeightingScheme;
import timeseriesweka.classifiers.ensembles.weightings.TrainAccByClass;
import timeseriesweka.classifiers.ensembles.voting.MajorityVote;
import timeseriesweka.classifiers.ensembles.voting.NaiveBayesCombiner;
import timeseriesweka.classifiers.ensembles.voting.ModuleVotingScheme;
import development.MultipleClassifiersPairwiseTest;
import fileIO.OutFile;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import timeseriesweka.classifiers.cote.HiveCoteModule;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.DebugPrinting;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import utilities.SaveParameterInfo;
import utilities.StatisticalUtilities;
import utilities.TrainAccuracyEstimate;
import utilities.ClassifierResults;
import timeseriesweka.classifiers.ensembles.EnsembleFromFile;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import timeseriesweka.filters.SAX;
import utilities.Timer;
import weka.filters.Filter;

/**
 *
 * In it's current form, can either build ensemble and classify from results files 
 * of its members, (call setResultsFileLocationParameters(...)) else by default
 * will build/train the ensemble members normally. It will not do any file writing
 (e.g trainFold# files of its members) aside from the overall ensemble train file as 
 defined by/related to SaveParameterInfo, if that has been set up (setCVPath(...))
 * 
 * 
 * @author James Large (james.large@uea.ac.uk) , Jason Lines (j.lines@uea.ac.uk)
 *      
 */

public class HESCA extends EnsembleFromFile implements HiveCoteModule, SaveParameterInfo, DebugPrinting, TrainAccuracyEstimate {
    
    protected ModuleWeightingScheme weightingScheme = new TrainAcc();
    protected ModuleVotingScheme votingScheme = new MajorityVote();
    protected EnsembleModule[] modules;
    
    protected boolean setSeed = false;
    protected int seed;
    
    protected Classifier[] classifiers; //todo classifiers essentially stored in the modules
    //can get rid of this reference at some point 
    
    
    protected String[] classifierNames;
    protected String[] classifierParameters;
    
    protected final SimpleBatchFilter transform;
    protected Instances train;
    
    //TrainAccuracyEstimate
    protected boolean writeEnsembleTrainingFile = false;
    protected String outputEnsembleTrainingPathAndFile;
    
    protected boolean performEnsembleCV = false;
    protected CrossValidator cv = null;
    //private ArrayList<ArrayList<Integer>> cvfoldsInfo; //populated if doing ensemble cv
    private ClassifierResults ensembleTrainResults = null;//data generated during buildclassifier if above = true 
    private ClassifierResults ensembleTestResults = null;//data generated during testing
    
    //train/test data info
    protected int numTrainInsts;
    protected int numAttributes;
    protected int numClasses;
    protected int testInstCounter = 0;
    protected int numTestInsts = -1;
    protected Instance prevTestInstance = null;
    
        
    protected static class ErrorReport {
        private boolean anyErrors; 
        private String errorLog;
        
        public ErrorReport(String msgHeader) {
            errorLog = msgHeader;
        }
        
        public void log(String errorMsg) {
            anyErrors = true;
            errorLog += errorMsg;
        }
        
        public void throwIfErrors() throws Exception {
            if (anyErrors)
                throw new Exception(errorLog);
        }
        
        public boolean isEmpty() { return !anyErrors; };
        public String getLog() { return errorLog; };
    }
    
    public HESCA() {
        this.ensembleIdentifier = "HESCA";
        
        this.transform = null;
        this.setDefaultClassifiers();
    }
    
    public HESCA(SimpleBatchFilter transform) {
        this.ensembleIdentifier = "HESCA";
        
        this.transform = transform;
        this.setDefaultClassifiers();
    }
    
    public HESCA(Classifier[] classifiers, String[] classifierNames) {
        this.ensembleIdentifier = "HESCA";
        
        this.transform = null;
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
        
        setDefaultClassifierParameters();
    }
    
    public HESCA(SimpleBatchFilter transform, Classifier[] classifiers, String[] classifierNames) {
        this.ensembleIdentifier = "HESCA";
        
        this.transform = transform;
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
        
        setDefaultClassifierParameters();
    }
    
    public Classifier[] getClassifiers(){ return classifiers;}
    
    public void setClassifiers(Classifier[] classifiers, String[] classifierNames) {
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
        setDefaultClassifierParameters();
    }
    
    public void setClassifiers(Classifier[] classifiers, String[] classifierNames, String[] parameters) {
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
        this.classifierParameters = parameters;
    }
    
    protected final void setDefaultClassifierParameters(){
        classifierParameters = new String[classifiers.length];
        for (String s : classifierParameters)
            s = "internalHESCA";
    }
    
    public final void setDefaultClassifiers(){
        this.classifiers = new Classifier[8];
        this.classifierNames = new String[8];
        
        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        classifiers[0] = k;
        classifierNames[0] = "NN";
            
        classifiers[1] = new NaiveBayes();
        classifierNames[1] = "NB";
        
        classifiers[2] = new J48();
        classifierNames[2] = "C4.5";
        
        SMO svml = new SMO();
        svml.turnChecksOff();
        PolyKernel kl = new PolyKernel();
        kl.setExponent(1);
        svml.setKernel(kl);
        if(setSeed)
            svml.setRandomSeed(seed);
        classifiers[3] = svml;
        classifierNames[3] = "SVML";
        
        SMO svmq =new SMO();
//Assumes no missing, all real valued and a discrete class variable        
        svmq.turnChecksOff();
        PolyKernel kq = new PolyKernel();
        kq.setExponent(2);
        svmq.setKernel(kq);
        if(setSeed)
           svmq.setRandomSeed(seed);
        classifiers[4] =svmq;
        classifierNames[4] = "SVMQ";
        
        RandomForest r=new RandomForest();
        r.setNumIterations(500);
        if(setSeed)
           r.setSeed(seed);            
        classifiers[5] = r;
        classifierNames[5] = "RandF";
            
            
        RotationForest rf=new RotationForest();
        rf.setNumIterations(50);
        if(setSeed)
           rf.setSeed(seed);
        classifiers[6] = rf;
        classifierNames[6] = "RotF";
        
        classifiers[7] = new BayesNet();
        classifierNames[7] = "bayesNet";   
        
        
        setDefaultClassifierParameters();
    }
    
    public void setPerformCV(boolean b) {
        performEnsembleCV = b;
    }
            
    public void setRandSeed(int seed){
        this.setSeed = true;
        this.seed = seed;
    }

//Constants that determine the number of CV folds
//    The number of folds is all pretty arbitrary, but practically makes little difference
//especially if RandomForest and RotationForest use OOB error    
    public static int MAX_NOS_FOLDS=100;
    public static int FOLDS1=20;
    public static int FOLDS2=10;
    public static int NUM_CASES_THRESHOLD1=300;
    public static int NUM_CASES_THRESHOLD2=200;
    public static int NUM_CASES_THRESHOLD3=100;
    public static int NUM_ATTS_THRESHOLD1=200;
    public static int NUM_ATTS_THRESHOLD2=500;

  
    public static int findNumFolds(Instances train){
//        int numFolds = train.numInstances();
//        if(train.numInstances()>=NUM_CASES_THRESHOLD1)
//            numFolds=FOLDS2;
//        else if(train.numInstances()>=NUM_CASES_THRESHOLD2 && train.numAttributes()>=NUM_ATTS_THRESHOLD1)
//            numFolds=FOLDS2;
//        else if(train.numAttributes()>=NUM_ATTS_THRESHOLD2)
//            numFolds=FOLDS2;
//        else if (train.numInstances()>=NUM_CASES_THRESHOLD3) 
//            numFolds=FOLDS1;
//        return numFolds;
        return 10;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        printlnDebug("**HESCA TRAIN**");
        
        //transform data if specified
        if(this.transform==null){
            this.train = data;
        }else{
            this.train = Filter.useFilter(data, transform);
        }
        
        //init
        this.numTrainInsts = train.numInstances();
        this.numClasses = train.numClasses();
        this.numAttributes = train.numAttributes();
        
        //set up modules
        initialiseModules();
        weightingScheme.defineWeightings(modules, numClasses);
        votingScheme.trainVotingScheme(modules, numClasses);
                
        //if writing results of this ensemble (e.g to be read later as an individual module of a meta ensemble), 
        //write the full ensemble trainFold# file
        if(this.performEnsembleCV) {
            doEnsembleCV(data); //combine modules to find overall ensemble trainpreds 
            findEnsembleBuildTime();
            if(this.writeEnsembleTrainingFile) 
                writeEnsembleCVResults(data);
        }
        
        this.testInstCounter = 0; //prep for start of testing
    }
    
    /**
     * this does cv writing for TrainAccuracyEstimate, where the output path (outputEnsembleTrainingPathAndFile, set by 'writeCVTrainToFile(...)') 
     * may be different to the ensemble write path (resultsFilesDirectory, set by 'setResultsFileLocationParameters(...))
     */
    protected void writeEnsembleCVResults(Instances data) throws IOException {
        StringBuilder output = new StringBuilder();

        output.append(data.relationName()).append(",").append(ensembleIdentifier).append(",train\n");
        output.append(this.getParameters()).append("\n");
        output.append(ensembleTrainResults.acc).append("\n");
        double[] predClassVals=ensembleTrainResults.getPredClassVals();
        for(int i = 0; i < numTrainInsts; i++){
            //realclassval,predclassval,[empty],probclass1,probclass2,...
            output.append(data.instance(i).classValue()).append(",").append(predClassVals[i]).append(",");
            double[] distForInst=ensembleTrainResults.getDistributionForInstance(i);
            for (int j = 0; j < numClasses; j++) 
                output.append(",").append(distForInst[j]);
            
            output.append("\n");
        }

        new File(this.outputEnsembleTrainingPathAndFile).getParentFile().mkdirs();
        FileWriter fullTrain = new FileWriter(this.outputEnsembleTrainingPathAndFile);
        fullTrain.append(output);
        fullTrain.close();
    }

    protected void initialiseModules() throws Exception {
        this.modules = new EnsembleModule[classifierNames.length];
        for (int m = 0; m < modules.length; m++)
            modules[m] = new EnsembleModule(classifierNames[m], classifiers[m], classifierParameters[m]);
        
        //prep cv         
        if (willNeedToDoCV()) {
            int numFolds = setNumberOfFolds(train); //through TrainAccuracyEstimate interface
            
            cv = new CrossValidator();
            if (setSeed)
                cv.setSeed(seed);
            cv.setNumFolds(numFolds);
            cv.buildFolds(train);
        }
        
        //currently will only have file reading ON or OFF (not load some files, train the rest) 
        //having that creates many, many, many annoying issues, especially when classifying test cases
        if (readIndividualsResults) {
            if (!resultsFilesParametersInitialised)
                throw new Exception("Trying to load HESCA modules from file, but parameters for results file reading have not been initialised");
            loadModules(); //will throw exception if a module cannot be loaded (rather than e.g training that individual instead)
        } 
        else 
            trainModules();
        
        for (int m = 0; m < modules.length; m++) {
            modules[m].trainResults.setNumClasses(numClasses);
            modules[m].trainResults.setNumInstances(numTrainInsts);
        }
    }
    
    protected boolean willNeedToDoCV() {
        //if we'll need ensemble cv, we want the cv fold info
        if (performEnsembleCV)
            return true;
        
        //or if any of the modules dont have cv data already
        for (EnsembleModule m : modules)
            if (!(m.getClassifier() instanceof TrainAccuracyEstimate))
                return true;
        
        return false;
    }
    
    protected void trainModules() throws Exception {
        
        for (EnsembleModule module : modules) {
            if (module.getClassifier() instanceof TrainAccuracyEstimate) {
                module.getClassifier().buildClassifier(train);
                
                //these train results should also include the buildtime
                module.trainResults = ((TrainAccuracyEstimate)module.getClassifier()).getTrainResults();
                
                if (writeIndividualsResults) { //if we're doing trainFold# file writing
                    String params = module.getParameters();
                    if (module.getClassifier() instanceof SaveParameterInfo)
                        params = ((SaveParameterInfo)module.getClassifier()).getParameters();
                    writeResultsFile(module.getModuleName(), params, module.trainResults, "train"); //write results out
                    printlnDebug(module.getModuleName() + " writing train file data gotten through TrainAccuracyEstimate...");
                }

                // |  from before the TrainAccuracyEstimate interface was made, kept just in case
                // v
//                if (needIndividualTrainPreds()) { //need the preds/dists of the train set (cv)
//                    throw new Exception("This setup requires individual train preds/dists, but SaveCVAccuracy not updated yet to extract existing trainpreds from individuals");
//                     
////                  ClassifierResults res = ((SaveParameterInfo)module.getClassifier()).getModuleResults(); //implement
//                    //if that's not null, hooray
//                    //else either throw exeption or perform cv manually again, to decide
////                    
////                    if (writeIndividualsResults) { //if we're doing trainFold# file writing
////                        writeResultsFile(module.getModuleName(), module.getParameters(), module.trainResults, "train"); //write results out
////                        printlnDebug(module.getModuleName() + " writing train file with full preds from SaveParameterInfo...");
////                    }
//                } else {//only need the accuracy
//                    String params = ((SaveParameterInfo)module.getClassifier()).getParameters();
//                    double cvacc = Double.parseDouble(params.split(",")[1]);
//                    module.trainResults = new ClassifierResults(cvacc, numClasses);
//                    
//                    if (writeIndividualsResults) { //if we're doing trainFold# file writing
//                        writeResultsFile(module.getModuleName(), params, module.trainResults, "train"); //write results out
//                        printlnDebug(module.getModuleName() + " writing train file without preds from SaveCVAccuracy...");
//                    }
//                }
            }
            else {
                printlnDebug(module.getModuleName() + " performing cv...");
                module.trainResults = cv.crossValidateWithStats(module.getClassifier(), train);

                //assumption: classifiers that maintain a classifierResults object, which may be the same object that module.trainResults refers to,
                //and which this subsequent building of the final classifier would tamper with, would have been handled as an instanceof TrainAccuracyEstimate above
                Timer buildTimer = new Timer();
                module.getClassifier().buildClassifier(train);
                module.trainResults.buildTime = buildTimer.timeSoFar();
                module.setParameters("BuildTime,"+module.trainResults.buildTime+","+module.getParameters());
                
                if (writeIndividualsResults) { //if we're doing trainFold# file writing
                    writeResultsFile(module.getModuleName(), module.getParameters(), module.trainResults, "train"); //write results out
                    printlnDebug(module.getModuleName() + " writing train file with full preds from scratch...");
                }
            }
        }
    }
    
    protected void loadModules() throws Exception {
        //will look for all files and report all that are missing, instead of bailing on the first file not found
        //just helps debugging/running experiments a little 
        ErrorReport errors = new ErrorReport("Errors while loading modules from file. Directory given: " + resultsFilesDirectory);
        
        //for each module
        for(int m = 0; m < this.modules.length; m++){
            boolean trainResultsLoaded = false;
            boolean testResultsLoaded = false; 
            
            //try and load in the train/test results for this module
            File moduleTrainResultsFile = findResultsFile(classifierNames[m], "train");
            if (moduleTrainResultsFile != null) { 
                printlnDebug(classifierNames[m] + " train loading... " + moduleTrainResultsFile.getAbsolutePath());
                
                modules[m].trainResults = loadResultsFile(moduleTrainResultsFile, numClasses);

                trainResultsLoaded = true;
            }

            File moduleTestResultsFile = findResultsFile(classifierNames[m], "test");
            if (moduleTestResultsFile != null) { 
                //of course these results not actually used at all during training, 
                //only loaded for future use when classifying with ensemble
                printlnDebug(classifierNames[m] + " test loading..." + moduleTestResultsFile.getAbsolutePath());

                modules[m].testResults = loadResultsFile(moduleTestResultsFile, numClasses);

                numTestInsts = modules[m].testResults.numInstances();
                testResultsLoaded = true;
            }
            
            if (!trainResultsLoaded) {
                errors.log("\nTRAIN results files for '" + classifierNames[m] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ");
            }
            else if (needIndividualTrainPreds() && modules[m].trainResults.predictedClassProbabilities.isEmpty()) {
                errors.log("\nNo pred/distribution for instance data found in TRAIN results file for '" + classifierNames[m] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "'. ");
            }
        
            if (!testResultsLoaded) {
                errors.log("\nTEST results files for '" + classifierNames[m] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ");
            }
            else if (modules[m].testResults.numInstances()==0) {
                errors.log("\nNo prediction data found in TEST results file for '" + classifierNames[m] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "'. ");
            }
            else if (modules[m].testResults.numInstances()==0) {
                errors.log("\nNo distribution for instance data found in TEST results file for '" + classifierNames[m] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "'. ");
            }
        }
        
        errors.throwIfErrors();
    }
    
    protected boolean needIndividualTrainPreds() {
        return performEnsembleCV || weightingScheme.needTrainPreds || votingScheme.needTrainPreds;
    }
    
    protected void findEnsembleBuildTime() { 
        //HESCA's buildtime is defined as a sum of the buildtimes of all its members
        //there will naturally be some extra overhead, and does not consider cv if it is required,
        //however there are too many edge cases involving the question of whether we're reading results from
        //file or not etc. to simply define the build time as the time from the start to the end of buildClassifier()
        this.ensembleTrainResults.buildTime = 0;
        for (int m = 0; m < modules.length; m++) {
            if (modules[m].trainResults.buildTime == -1) {//if individual's time is not known
                this.ensembleTrainResults.buildTime = -1; //cannot know ensemble's time
                break;
            }
            this.ensembleTrainResults.buildTime += modules[m].trainResults.buildTime;
        }
    }

    protected void doEnsembleCV(Instances data) throws Exception {
        double[] preds = new double[numTrainInsts];
        double[][] dists = new double[numTrainInsts][];
        double[] accPerFold = new double[cv.getNumFolds()]; //for variance
        
        
        double actual, pred, correct = 0;
        double[] dist;
        
        //for each train inst
        for (int fold = 0; fold < cv.getNumFolds(); fold++) {
            for (int i = 0; i < cv.getFoldIndices().get(fold).size(); i++) {
                int instIndex = cv.getFoldIndices().get(fold).get(i);
                
                dist = votingScheme.distributionForTrainInstance(modules, instIndex); 

                pred = indexOfMax(dist);
                actual = data.instance(instIndex).classValue();
                //and make ensemble prediction
                if(pred==actual) {
                    correct++;
                    accPerFold[fold]++;
                }

                preds[instIndex] = pred;
                dists[instIndex] = dist;
            }
            
            accPerFold[fold] /= cv.getFoldIndices().get(fold).size();
        }
        
        double acc = correct/numTrainInsts;
        double stddevOverFolds = StatisticalUtilities.standardDeviation(accPerFold, false, acc);
        
        ensembleTrainResults = new ClassifierResults(acc, data.attributeToDoubleArray(data.classIndex()), preds, dists, stddevOverFolds, numClasses);
        ensembleTrainResults.setNumClasses(numClasses);
        ensembleTrainResults.setNumInstances(numTrainInsts);
    }
    
    /**
     * If building individuals from scratch, i.e not read results from files, call this
     * after testing is complete to build each module's testResults (accessible by module.testResults)
     * 
     * This will be done internally anyway if writeIndividualTestFiles(...) is called, this method
     * is made public only so that results can be accessed from memory during the same run if wanted 
     */
    public void finaliseIndividualModuleTestResults(double[] testSetClassVals) throws Exception {
        for (EnsembleModule module : modules)
            module.testResults.finaliseResults(testSetClassVals); //converts arraylists to double[]s and preps for writing
    }
    
    /**
     * If building individuals from scratch, i.e not read results from files, call this
     * after testing is complete to build each module's testResults (accessible by module.testResults)
     * 
     * This will be done internally anyway if writeIndividualTestFiles(...) is called, this method
     * is made public only so that results can be accessed from memory during the same run if wanted 
     */
    public void finaliseEnsembleTestResults(double[] testSetClassVals) throws Exception {
        this.ensembleTestResults.finaliseResults(testSetClassVals);
        ensembleTestResults.setNumClasses(numClasses);
        ensembleTestResults.setNumInstances(numTestInsts);        
    }
    
    /**
     * @param throwExceptionOnFileParamsNotSetProperly added to make experimental code smoother, 
     *  i.e if false, can leave the call to writeIndividualTestFiles(...) in even if building from file, and this 
     *  function will just do nothing. else if actually intending to write test results files, pass true
     *  for exceptions to be thrown in case of genuine missing parameter settings
     * @throws Exception 
     */
    public void writeIndividualTestFiles(double[] testSetClassVals, boolean throwExceptionOnFileParamsNotSetProperly) throws Exception {
        if (!writeIndividualsResults || !resultsFilesParametersInitialised) {
            if (throwExceptionOnFileParamsNotSetProperly)
                throw new Exception("to call writeIndividualTestFiles(), must have called setResultsFileLocationParameters(...) and setWriteIndividualsResultsFiles()");
            else 
                return; //do nothing
        }
        
        finaliseIndividualModuleTestResults(testSetClassVals);
        
        for (EnsembleModule module : modules)
            writeResultsFile(module.getModuleName(), module.getParameters(), module.testResults, "test");
    }
    
    /**
     * @param throwExceptionOnFileParamsNotSetProperly added to make experimental code smoother, 
     *  i.e if false, can leave the call to writeIndividualTestFiles(...) in even if building from file, and this 
     *  function will just do nothing. else if actually intending to write test results files, pass true
     *  for exceptions to be thrown in case of genuine missing parameter settings
     * @throws Exception 
     */
    public void writeEnsembleTrainTestFiles(double[] testSetClassVals, boolean throwExceptionOnFileParamsNotSetProperly) throws Exception {
        if (!resultsFilesParametersInitialised) {
            if (throwExceptionOnFileParamsNotSetProperly)
                throw new Exception("to call writeEnsembleTrainTestFiles(), must have called setResultsFileLocationParameters(...)");
            else 
                return; //do nothing
        }
        
        if (ensembleTrainResults != null) //performed cv
            writeResultsFile(ensembleIdentifier, getParameters(), ensembleTrainResults, "train");
        
        this.ensembleTestResults.finaliseResults(testSetClassVals);
        writeResultsFile(ensembleIdentifier, getParameters(), ensembleTestResults, "test");
    }

    public EnsembleModule[] getModules() {
        return modules;
    }
    
    public CrossValidator getCrossValidator() {
        return cv;
    }
    
    public String[] getClassifierNames() {
        return classifierNames;
    }

    public double[] getEnsembleCvPreds() {
        return ensembleTrainResults.getPredClassVals();
    }

    public double getEnsembleCvAcc() {
        return ensembleTrainResults.acc;
    }

    public String getEnsembleIdentifier() {
        return ensembleIdentifier;
    }
    
    public void setEnsembleIdentifier(String ensembleIdentifier) {
        this.ensembleIdentifier = ensembleIdentifier;
    }

    public double[][] getPosteriorIndividualWeights() {
        double[][] weights = new double[modules.length][];
        for (int m = 0; m < modules.length; ++m) 
            weights[m] = modules[m].posteriorWeights;
        
        return weights;
    }

    public ModuleVotingScheme getVotingScheme() {
        return votingScheme;
    }

    public void setVotingScheme(ModuleVotingScheme votingScheme) {
        this.votingScheme = votingScheme;
    }
    
    public ModuleWeightingScheme getWeightingScheme() {
        return weightingScheme;
    }

    public void setWeightingScheme(ModuleWeightingScheme weightingScheme) {
        this.weightingScheme = weightingScheme;
    }

    public double[] getIndividualCvAccs() {
        double [] accs = new double[modules.length];
        for (int i = 0; i < modules.length; i++) 
            accs[i] = modules[i].trainResults.acc;

        return accs;
    }
    
    //    public double[] getPriorIndividualWeights() {
//        return priorIndividualWeights;
//    }
//    
//    public void setPriorIndividualWeights(double[] priorWeights) {
//        this.priorIndividualWeights = priorWeights;
//    }
//            
//    private void setDefaultPriorWeights() {
//        priorIndividualWeights = new double[classifierNames.length];
//        for (int i = 0; i < priorIndividualWeights.length; i++)
//            priorIndividualWeights[i] = 1;
//    }
    

//    @Override
    public double[][] getIndividualCvPredictions() {
        double [][] preds = new double[modules.length][];
        for (int i = 0; i < modules.length; i++) 
            preds[i] = modules[i].trainResults.getPredClassVals();
        return preds;
    }
    
    public SimpleBatchFilter getTransform(){
        return this.transform;
    }
    
    @Override
    public void writeCVTrainToFile(String path) {
        outputEnsembleTrainingPathAndFile=path;
        performEnsembleCV=true;
    }    
    @Override
    public boolean findsTrainAccuracyEstimate(){ return performEnsembleCV;}
    
    @Override
    public ClassifierResults getTrainResults(){ 
        return ensembleTrainResults;
    }        
     
    public ClassifierResults getTestResults(){
        return ensembleTestResults;
    }
    
    @Override
    public String getParameters(){
        StringBuilder out = new StringBuilder();
        
        if (ensembleTrainResults != null) //cv performed
            out.append("BuildTime,").append(ensembleTrainResults.buildTime).append(",Trainacc,").append(ensembleTrainResults.acc).append(",");
        else 
            out.append("BuildTime,").append("-1").append(",Trainacc,").append("-1").append(",");
        
        out.append(weightingScheme.toString()).append(",").append(votingScheme.toString()).append(",");
        
        for(int m = 0; m < modules.length; m++){
            out.append(modules[m].getModuleName()).append("(").append(modules[m].priorWeight);
            for (int j = 0; j < modules[m].posteriorWeights.length; ++j)
                out.append("/").append(modules[m].posteriorWeights[j]);
            out.append("),");
        }
        
        return out.toString();
    }
    
//    public void readParameters(String paramLine) { 
//        String[] classifiers = paramLine.split(",");
//        
//        String[] classifierNames = new String[classifiers.length];
//        double[] priorWeights = new double[classifiers.length];
//        double[] postWeights = new double[classifiers.length];
//        
//        for (int i = 0; i < classifiers.length; ++i) {
//            String[] parts = classifiers[i].split("(");
//            classifierNames[i] = parts[0];
//            String[] weights = parts[1].split("/");
//            priorWeights[i] = Integer.parseInt(weights[0]);
//            for (int j = 1; j < weights.length; ++j)
//                postWeights[j-1] = Integer.parseInt(weights[j]);
//        }
//        
//    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            Instances converted = Filter.useFilter(rawContainer, transform);
            ins = converted.instance(0);
        }
        
        if (ensembleTestResults == null || (testInstCounter == 0 && prevTestInstance == null)) {//definitely the first call, not e.g the first inst being classified for the second time
            printlnDebug("\n**TEST**");
            
            ensembleTestResults = new ClassifierResults(numClasses);
        }
        
        if (readIndividualsResults && testInstCounter >= numTestInsts) //if no test files loaded, numTestInsts == -1
            throw new Exception("Received more test instances than expected, when loading test results files, found " + numTestInsts + " test cases");
                
        double[] dist;
        if (readIndividualsResults) //have results loaded from file
            dist = votingScheme.distributionForTestInstance(modules, testInstCounter);
        else //need to classify them normally
            dist = votingScheme.distributionForInstance(modules, ins);
        
        ensembleTestResults.storeSingleResult(dist);
        
        if (prevTestInstance != instance)
            ++testInstCounter;
        prevTestInstance = instance;
        
        return dist;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        return indexOfMax(dist);
    }
    
    protected static double indexOfMax(double[] dist) {
        double max = dist[0];
        double maxInd = 0;
        
        for (int i = 1; i < dist.length; ++i) {
            if (dist[i] > max) {
                max = dist[i];
                maxInd = i;
            }
        }
        return maxInd;
    }
    
//    protected static double indexOfMax(double[] dist) throws Exception {  
//        double  bsfWeight = -(Double.MAX_VALUE);
//        ArrayList<Integer>  bsfClassVals = null;
//        
//        for (int c = 0; c < dist.length; c++) {
//            if(dist[c] > bsfWeight){
//                bsfWeight = dist[c];
//                bsfClassVals = new ArrayList<>();
//                bsfClassVals.add(c);
//            }else if(dist[c] == bsfWeight){
//                bsfClassVals.add(c);
//            }
//        }
//
//        if(bsfClassVals == null)
//            throw new Exception("bsfClassVals == null, NaN problem");
//
//        double pred; 
//        //if there's a tie for highest voted class after all module have voted, settle randomly
//        if(bsfClassVals.size()>1)
//            pred = bsfClassVals.get(new Random(0).nextInt(bsfClassVals.size()));
//        else
//            pred = bsfClassVals.get(0);
//        
//        return pred;
//    }
    
    /**
     * @return the predictions of each individual module, i.e [0] = first module's vote, [1] = second...
     */
    public double[] classifyInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            Instances converted = Filter.useFilter(rawContainer, transform);
            ins = converted.instance(0);
        }
        
        double[] predsByClassifier = new double[this.classifiers.length];
                
        for(int i=0;i<classifiers.length;i++){
            predsByClassifier[i] = classifiers[i].classifyInstance(ins);
        }
        
        return predsByClassifier;
    }
    
    /**
     * @return the distributions of each individual module, i.e [0] = first module's dist, [1] = second...
     */
    public double[][] distributionForInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            Instances converted = Filter.useFilter(rawContainer, transform);
            ins = converted.instance(0);
        }
        
        double[][] distsByClassifier = new double[this.classifiers.length][];
                
        for(int i=0;i<classifiers.length;i++){
            distsByClassifier[i] = classifiers[i].distributionForInstance(ins);
        }
        
        return distsByClassifier;
    }
    
//    /**
//     * classifiers/cnames are optional, leave null for default classifiers
//     * 
//     * todo: is a bodge job from previous code, before 'the big refactor'
//     * 
//     * clean it up at some point and use the up to date methods
//     */
//    public static void buildAndWriteFullIndividualTrainTestResults(Instances defaultTrainPartition, Instances defaultTestPartition, 
//            String resultOutputDir, String datasetIdentifier, String ensembleIdentifier, int resampleIdentifier, 
//            Classifier[] classifiers, String[] cNames,
//            SimpleBatchFilter transform, boolean setSeed, boolean resample, boolean writeEnsembleTrainResults) throws Exception{
//        HESCA h;
//        if(classifiers != null) 
//            h = new HESCA(transform, classifiers, cNames);
//        else 
//            h = new HESCA(transform);
//        
//        Instances train = new Instances(defaultTrainPartition);
//        Instances test = new Instances(defaultTestPartition);
//        if(resample && resampleIdentifier >0){
//            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, resampleIdentifier);
//            train = temp[0];
//            test = temp[1];
//        }
//        
//        if (setSeed)
//            h.setRandSeed(resampleIdentifier);
//                
//        if (writeEnsembleTrainResults)
//            h.writeCVTrainToFile(resultOutputDir+h.ensembleIdentifier+"/Predictions/"+datasetIdentifier+"/trainFold"+resampleIdentifier+".csv");
//        
//        h.buildClassifier(train);
//        
//        StringBuilder[] byClassifier = new StringBuilder[h.classifiers.length+1];
//        for(int c = 0; c < h.classifiers.length+1; c++){
//            byClassifier[c] = new StringBuilder();
//        }
//        int[] correctByClassifier = new int[h.classifiers.length+1];
//        
//        double cvSum = 0;
//        for(int c = 0; c < h.classifiers.length; c++){
//            cvSum+= h.modules[c].trainResults.acc;
//        }
//        int correctByEnsemble = 0;
//                
//        double act;
//        double pred;
//        double[] preds;
//        double[][] dists;
//        
//        
//        double[] distForIns = null;
//        double bsfClassVal = -1;
//        double bsfClassWeight = -1;
//        for(int i = 0; i < test.numInstances(); i++){
//            act = test.instance(i).classValue();
//            
//            dists = h.distributionForInstanceByConstituents(test.instance(i));
//            preds = new double[dists.length];
//            
//            for (int j = 0; j < dists.length; j++)
//                preds[j] = indexOfMax(dists[j]);
//                
//            distForIns = new double[test.numClasses()];
//            bsfClassVal = -1;
//            bsfClassWeight = -1;
//            for(int c = 0; c < h.classifiers.length; c++){
//                byClassifier[c].append(act).append(",").append(preds[c]).append(",");
//                for (int j = 0; j < test.numClasses(); j++)
//                    byClassifier[c].append(",").append(dists[c][j]);
//                byClassifier[c].append("\n");
//                
//                if(preds[c]==act){
//                    correctByClassifier[c]++;
//                }
//                
//                distForIns[(int)preds[c]]+= h.modules[c].trainResults.acc;
//                if(distForIns[(int)preds[c]] > bsfClassWeight){
//                    bsfClassVal = preds[c];
//                    bsfClassWeight = distForIns[(int)preds[c]];
//                }
//            }
//            
//            if(bsfClassVal==act){
//                correctByEnsemble++;
//            }
//            byClassifier[h.classifiers.length].append(act+","+bsfClassVal+",");
//            for(int cVal = 0; cVal < distForIns.length; cVal++){
//                byClassifier[h.classifiers.length].append(","+distForIns[cVal]/cvSum);
//            }
//            byClassifier[h.classifiers.length].append("\n");
//        }
//        
//        h.setResultsFileLocationParameters(resultOutputDir, datasetIdentifier, resampleIdentifier); //dat hack... set after building/testing
//        
//        for (EnsembleModule module : h.modules)
//            h.writeResultsFile(module.getModuleName(), module.getParameters(), module.trainResults, "train");
//        
//        FileWriter out;
//        String outPath;
//        for(int c = 0; c < h.classifiers.length; c++){
//            outPath = h.resultsFilesDirectory+h.classifierNames[c]+"/Predictions/"+h.datasetName;
//            new File(outPath).mkdirs();
//            out = new FileWriter(outPath+"/testFold"+h.resampleIdentifier+".csv");
//            out.append(h.datasetName+","+h.ensembleIdentifier+h.classifierNames[c]+",test\n");
//            out.append("noParamInfo\n");
//            out.append((double)correctByClassifier[c]/test.numInstances()+"\n");
//            out.append(byClassifier[c]);
//            out.close();            
//        }
//        
//        outPath = h.resultsFilesDirectory+h.ensembleIdentifier+"/Predictions/"+h.datasetName;
//        new File(outPath).mkdirs();
//        out = new FileWriter(outPath+"/testFold"+h.resampleIdentifier+".csv");
//        out.append(h.datasetName+","+h.ensembleIdentifier+",test\n");
//        out.append("noParamInfo\n");
//        out.append((double)correctByEnsemble/test.numInstances()+"\n");
//        out.append(byClassifier[h.classifiers.length]);
//        out.close(); 
//        
//    }
    
    /**
     * classifiers/cnames are optional, leave null for default classifiers
     * 
     * todo: is a bodge job from previous code, before 'the big refactor'
     * 
     * clean it up at some point and use the up to date methods
     */
    public static void buildAndWriteFullIndividualTrainTestResults(Instances defaultTrainPartition, Instances defaultTestPartition, 
            String resultOutputDir, String datasetIdentifier, String ensembleIdentifier, int resampleIdentifier, 
            Classifier[] classifiers, String[] cNames,
            SimpleBatchFilter transform, boolean setSeed, boolean resample, boolean writeEnsembleResults) throws Exception{

        
        Instances train = new Instances(defaultTrainPartition);
        Instances test = new Instances(defaultTestPartition);
        if(resample && resampleIdentifier >0){
            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, resampleIdentifier);
            train = temp[0];
            test = temp[1];
        }

        HESCA h;
        if(classifiers != null) 
            h = new HESCA(transform, classifiers, cNames);
        else 
            h = new HESCA(transform);
        
        if (setSeed)
            h.setRandSeed(resampleIdentifier);
        
        h.setWeightingScheme(new EqualWeighting());
        h.setResultsFileLocationParameters(resultOutputDir, datasetIdentifier, resampleIdentifier); //dat hack... set after building/testing
        h.setWriteIndividualsResultsFiles(true);
        
        if (writeEnsembleResults)
            h.setPerformCV(true);
            
        h.buildClassifier(train);
        
        for (Instance inst : test)
            h.distributionForInstance(inst); //will store results internally
        
        double[] classVals = test.attributeToDoubleArray(test.classIndex());
        h.writeIndividualTestFiles(classVals, true);
        if (writeEnsembleResults)
            h.writeEnsembleTrainTestFiles(classVals, true);
    }
    
    
    public static void exampleUseCase() throws Exception {
        String datasetName = "ItalyPowerDemand";
        
        Instances train = ClassifierTools.loadData("c:/tsc problems/"+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData("c:/tsc problems/"+datasetName+"/"+datasetName+"_TEST");
        
        SimpleBatchFilter transform = new SAX();
        HESCA hesca = new HESCA(transform); //default classifiers, can still use different transforms if wanted
        
        
        Classifier[] classifiers = new Classifier[] { new kNN() };
        String [] names = new String[] { "NN" };
        String [] params = new String[] { "k=100" };
        
        hesca = new HESCA(classifiers, names); //to specify classsifiers, either this
        hesca.setClassifiers(classifiers, names); //or this 
        hesca.setClassifiers(classifiers, names, params); //can also pass parameter strings for each if wanted, defaults to "internalHESCA" otherwise
        //does not check for \n characters or anything (which would break file format), just dont pass them
        
        hesca.setDefaultClassifiers();//default ones still old/untuned, for old results recreation/correctness testing purposes
        
        //default uses train acc weighting and majority voting (as before) 
        hesca.setWeightingScheme(new TrainAccByClass()); //or set new methods
        hesca.setVotingScheme(new MajorityVote()); //some voting schemes require dist for inst to be defined
        
        //some old results files may not have written them, will break in this case
        //buildAndWriteFullIndividualTrainTestResults(...) will produce files with dists if needed (outside of a normal
        //hesca experiemental run)
        
        int resampleID = 0;
        hesca.setRandSeed(resampleID);
        
        //use this to set the location for any results file reading/writing
        hesca.setResultsFileLocationParameters("hescaTest/", datasetName, resampleID);
        
        //include this to turn on file reading, will read from location provided in setResultsFileLocationParameters(...)
        hesca.setWriteIndividualsResultsFiles(true);
        //include this to turn on file writing for individuals trainFold# files 
        hesca.setBuildIndividualsFromResultsFiles(true);
        //can only have one of these (or neither) set to true at any one time (internally, setting one to true 
        //will automatically set the other to false)
        //if building classifiers from scratch and writing to file, you obviously arnt reading results files
        //likewise if reading from file, to write would only replace the existing files with identical copies
        
        hesca.buildClassifier(train);
        
        //test however you would a normal classifier, no file writing here
        System.out.println(ClassifierTools.accuracy(test, hesca));
        
        //use this after testing is complete to write the testFold# files of the individuals (if setResultsFileLocationParameters(...) has been called)
        //see the java doc on this func for reasoning behind the boolean
        boolean throwExceptionOnFileParamsNotSetProperly = false;
        hesca.writeIndividualTestFiles(test.attributeToDoubleArray(test.classIndex()), throwExceptionOnFileParamsNotSetProperly);
    }
    
    public static void test() throws Exception {
        System.out.println("test()");
        
        String dataset = "ItalyPowerDemand";
        
        Instances train = ClassifierTools.loadData("C:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
        Instances test = ClassifierTools.loadData("C:/tsc problems/"+dataset+"/"+dataset+"_TEST");
        
//        String[] classifierNames = new String[] {
//            "bayesNet",
//            "C4.5",
//            "NB",
//            "NN",
//            "RandF",
//            "RotF",
//            "SVML",
//            "SVMQ"
//        };
//  
//        Classifier[] classifiers = new Classifier[] {
//            null, null, null, null, null, null, null, null
//        };
        
//        HESCA hesca = new HESCA(classifiers, classifierNames);
        HESCA hesca = new HESCA();
        hesca.setResultsFileLocationParameters("C:/Users/xmw13bzu/Documents/Code/PHDCode/PHDCode/buildTimetests/", dataset, 0);
//        hesca.setResultsFileLocationParameters("buildTimetests/", dataset, -1);
        hesca.setBuildIndividualsFromResultsFiles(true);
        hesca.setPerformCV(true);
//        hesca.setWeightingScheme(new MCCWeighting());
//        hesca.setVotingScheme(new StackingOnPreds(new SMO())); 
//        hesca.setVotingScheme(new NaiveBayesCombiner());
        
        hesca.buildClassifier(train);
//        System.out.println(ClassifierTools.accuracy(test, hesca));

        double acc = .0;
        for (Instance instance : test) {
            if (instance.classValue() == hesca.classifyInstance(instance))
                acc++;
        }
        acc/=test.numInstances();
        System.out.println("acc = " + acc);
        
        hesca.finaliseEnsembleTestResults(test.attributeToDoubleArray(test.classIndex()));
        System.out.println(hesca.getTestResults().acc);
        
        hesca.writeEnsembleTrainTestFiles(test.attributeToDoubleArray(test.classIndex()), true);
        
//        hesca.writeIndividualTestFiles(test.attributeToDoubleArray(test.classIndex()), false);
    }
    
    public static void main(String[] args) throws Exception {
//        TunedRandomForest randF = new TunedRandomForest();
//        randF.tuneTree(false);
//        randF.tuneFeatures(false);
//        randF.setNumIterations(500);
//        randF.setSeed(0);
//        randF.setTrainAcc(true);
//        randF.setCrossValidate(false);
//        
//        TunedRotationForest rotF = new TunedRotationForest();
//        rotF.setNumIterations(200);
//        rotF.tuneFeatures(false);
//        rotF.tuneTree(false);
//        rotF.estimateAccFromTrain(true);
//        
//        buildBulkResultsFiles("bulkFilesTest/", new Classifier[] { new kNN(), new NaiveBayes() }, new String[] { "NN", "nbayes" });

        test();
//        debugTest();
//        ensembleVariationTests(true, completeUCIDatasets, "E:/JamesLPHD/HESCA/UCI/UCIResults/", "test");
//        ensembleVariationTests(true, completeUCIDatasets, "E:/20170301transfer/UCIResults/", "test");
//        ensembleVariationTests(false, smallishUCRdatasets, "C:/JamesLPHD/SmallUCRHESCAResultsFiles/", "metatests");
//        exampleUseCase(); //look here for how to use, below is my random testing
        
        
//        MultipleClassifiersPairwiseTest.runTests("schemeTestsByWeighttest.csv", "schemeTestsByWeighttestCliques.csv");
        System.exit(0);
        
        String dataset = "ItalyPowerDemand";
        
        Instances train = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
        Instances test = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TEST");
        
        String[] classifierNames = new String[] {
            "bayesNet",
            "C4.5",
            "NB",
            "NN",
            "RandF",
            "RotF",
            "SVML",
            "SVMQ"
        };

        Instances[] data = InstanceTools.resampleTrainAndTestInstances(train, test, 0);

        HESCA h = new HESCA();
//        h.setDebugPrinting(true);

        h.setResultsFileLocationParameters("C:/JamesLPHD/SmallUCRHESCAResultsFiles/", dataset, 0);
        h.setBuildIndividualsFromResultsFiles(true);
        h.buildClassifier(data[0]);

        double a = ClassifierTools.accuracy(data[1], h);
        
        System.out.println(h.buildEnsembleReport(true, true));
    }       
     
    private static int numResamples = 25;
    
    private static final String[] smallishUCRdatasets  = { 
//        "Adiac",
//        "ArrowHead",
//        "CBF",
//        "ChlorineConcentration",
//        "Coffee",
//        "CricketX",
//        "CricketY",
//        "CricketZ",
//        "DiatomSizeReduction",
//        "DistalPhalanxOutlineAgeGroup",
//        "DistalPhalanxOutlineCorrect",
//        "DistalPhalanxTW",
//        "ECG200",
        "ItalyPowerDemand",
//        "MiddlePhalanxOutlineAgeGroup",
//        "MiddlePhalanxOutlineCorrect",
//        "MiddlePhalanxTW",
//        "MoteStrain",
//        "ProximalPhalanxOutlineAgeGroup",
//        "ProximalPhalanxOutlineCorrect",
//        "ProximalPhalanxTW",
//        "SonyAIBORobotSurface1",
//        "SonyAIBORobotSurface2",
//        "SyntheticControl",
//        "TwoLeadECG"
    };
    
    private static final String[] completeUCIDatasets = { 
        "acute-inflammation","acute-nephritis","annealing","arrhythmia","balloons","blood","breast-cancer","breast-cancer-wisc",
"breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car","cardiotocography-10clases","cardiotocography-3clases","chess-krvkp","congressional-voting",
"conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli","energy-y1",
"energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian",
"heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere",
"iris","led-display","lenses","libras","low-res-spect","lung-cancer","lymphography","molec-biol-promoter",
"molec-biol-splice","monks-1","monks-2","monks-3","musk-1","ozone","parkinsons","pima",
"pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning","post-operative","primary-tumor",
"seeds","semeion","soybean","spect","spectf","statlog-australian-credit","statlog-german-credit","statlog-heart",
"statlog-image","statlog-landsat","statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe",
"titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases","waveform","waveform-noise","wine",
"wine-quality-red","yeast","zoo",
    };
    
    private static void buildBulkResultsFiles(String outpath, Classifier[] cs, String[] cn) throws Exception {
        System.out.println("buildBulkResultsFiles()");
        
        double done = 0;
        for (String dset : smallishUCRdatasets) {
            Instances train = ClassifierTools.loadData("c:/tsc problems/"+dset+"/"+dset+"_TRAIN");
            Instances test = ClassifierTools.loadData("c:/tsc problems/"+dset+"/"+dset+"_TEST");
            
            for (int r = 0; r < numResamples; ++r) 
                HESCA.buildAndWriteFullIndividualTrainTestResults(train, test, outpath, dset, "", r, cs, cn, null, true, true, true); 
            
            System.out.println(((++done)/smallishUCRdatasets.length));
        }
    }
    
    
    private static void ensembleVariationTests(boolean uci, String[] dsets, String individualResultsDirectory, String fileSuffix) throws Exception {
        System.out.println("schemeTests()");
        
        Classifier[] classifiers = null;
        String[] classifierNames = new String[] { "Default" };
//        classifiers = new Classifier[] { null,null,null };
//        classifierNames = new String[] {
//           "SVM", "RotFCV", "RandFCV" // 
//        };
        
        ModuleWeightingScheme[] weightSchemes = new ModuleWeightingScheme[] { 
//            new ConfusionEntropyByClass(), //broke
//            new ConfusionEntropy(),
//            new EqualWeighting(),
//            new MCCWeighting(),
            new TrainAcc(),
//            new TrainAccOrMCC(2.0),
//            new TrainAccOrMCC(3.0),
//            new TrainAccOrMCC(),
//            new TrainAccOrMCC(10.0),
//            new TrainAccByClass(),
//            new FScore(),//1.0 default 
//            new FScore(0.5),
//            new FScore(2.0),
        };
        
        ModuleVotingScheme[] voteSchemes = new ModuleVotingScheme[] { 
//            new MajorityVote(),
//            new MajorityVoteByCorrectedConfidence(),
//            new MajorityVoteByConfidence(),
//            new MajorityConfidence(),
//            new NP_MAX(),
//            new AverageOfConfidences(),
//            new AverageVoteByConfidence(),
//            new ProductOfConfidences(),
//            new ProductOfVotesByConfidence()
//            new BestIndividualOracle(),
//            new BestIndividualTrain()
//            new AbstractStacking(new NaiveBayes()),
//            new AbstractStacking(new J48()),
//            new AbstractStacking(new RandomForest()),
//            new AbstractStacking(new kNN()),
//            new AbstractStacking(new BayesNet()),
        };
        
        
        double[][][] accs = new double[voteSchemes.length][weightSchemes.length][dsets.length];
        
        for (int v = 0; v < voteSchemes.length; ++v) {
            ModuleVotingScheme vscheme = voteSchemes[v];
            
            for (int i = 0; i < weightSchemes.length; i++) {
                ModuleWeightingScheme wscheme = weightSchemes[i];

                for (int j = 0; j < dsets.length; j++) {
                    String dataset = dsets[j];

                    Instances train=null, test=null, all=null;
                    if (uci) {
                        all = ClassifierTools.loadData("C:/UCI Problems/"+dataset+"/"+dataset);
                    }else {//ucr
                        train = ClassifierTools.loadData("C:/TSC Problems/"+dataset+"/"+dataset+"_TRAIN");
                        test = ClassifierTools.loadData("C:/TSC Problems/"+dataset+"/"+dataset+"_TEST");
                    } 
                        
                    
                    double acc = .0;
                    for (int r = 0; r < numResamples; ++r)  {

                        Instances[] data=null;
                        if (uci)
                            data = InstanceTools.resampleInstances(all, r, 0.5);
                        else //ucr
                            data = InstanceTools.resampleTrainAndTestInstances(train, test, r);

                        HESCA h;
                        if (classifiers==null)
                            h = new HESCA();
                        else  
                            h = new HESCA(classifiers,classifierNames);
//                        h.setDebugPrinting(true);
                        h.setWeightingScheme(wscheme);
                        h.setVotingScheme(vscheme);
//                        h.getVotingScheme().setDebugPrinting(true);

                        h.setResultsFileLocationParameters(individualResultsDirectory, dataset, r);
                        h.setBuildIndividualsFromResultsFiles(true);
                        h.buildClassifier(data[0]);

                        double a = ClassifierTools.accuracy(data[1], h);
                        acc += a;
                        
//                        System.out.println(h.buildEnsembleReport(true, true));
                    }

                    acc /= numResamples;
                    accs[v][i][j] = acc;
                }

                System.out.println("\t" + wscheme.getClass().getName() + " done");
            }
            
            System.out.println(vscheme.getClass().getName() + " done");
        }
        
        OutFile out = new OutFile(arrayToStringNoSpace(classifierNames)+"_BYVOTE"+fileSuffix+".csv");
        for (int v = 0; v < voteSchemes.length; v++) {
            out.writeLine("" + voteSchemes[v]);
        
            for (int w = 0; w < weightSchemes.length; w++) 
                out.writeString("," + weightSchemes[w]);
            out.writeLine("");

            for (int d = 0; d < dsets.length; ++d) {
                out.writeString(dsets[d]);

                for (int w = 0; w < weightSchemes.length; w++)
                    out.writeString("," + accs[v][w][d]);
                out.writeLine("");
            }
            
            out.writeLine("\n\n");
        }
        out.closeFile();
        
        OutFile out2 = new OutFile(arrayToStringNoSpace(classifierNames)+"_BYWEIGHT"+fileSuffix+".csv");
         for (int w = 0; w < weightSchemes.length; w++) {
            out2.writeLine("" + weightSchemes[w]);
        
            for (int v = 0; v < voteSchemes.length; v++)
                out2.writeString("," + voteSchemes[v]);
            out2.writeLine("");

            for (int d = 0; d < dsets.length; ++d) {
                out2.writeString(dsets[d]);

                for (int v = 0; v < voteSchemes.length; v++)
                    out2.writeString("," + accs[v][w][d]);
                out2.writeLine("");
            }
            
            out2.writeLine("\n\n");
        }
        out2.closeFile();
        
        OutFile out3 = new OutFile(arrayToStringNoSpace(classifierNames)+"_COMBINED"+fileSuffix+".csv");
        for (int v = 0; v < voteSchemes.length; v++)
            for (int w = 0; w < weightSchemes.length; w++) 
                out3.writeString("," + voteSchemes[v] + "+" + weightSchemes[w]);
        out3.writeLine("");

        for (int d = 0; d < dsets.length; ++d) {
            out3.writeString(dsets[d]);

            for (int v = 0; v < voteSchemes.length; v++)
                for (int w = 0; w < weightSchemes.length; w++)
                    out3.writeString("," + accs[v][w][d]);
            out3.writeLine("");
        }
        out3.closeFile();
        
//////////   SUMMARY FILE,     below is all hacks to get old code to work, look away
        double[][] accscombined = new double[weightSchemes.length*voteSchemes.length][dsets.length];
        String[] combinedNames = new String[weightSchemes.length*voteSchemes.length];
        for (int v = 0; v < voteSchemes.length; v++)
            for (int w = 0; w < weightSchemes.length; w++) {
                accscombined[v*weightSchemes.length + w] = accs[v][w];
                combinedNames[v*weightSchemes.length + w] =  voteSchemes[v] + "+" + weightSchemes[w];
            }
      
        //avg ranks
        double[][] ranks = findRanks(accscombined);
        
        double[] rankmeans = new double[ranks.length];
        for (int c = 0; c < ranks.length; c++)
            rankmeans[c] = StatisticalUtilities.mean(ranks[c], false);
            
        //summary
        OutFile summ=new OutFile(arrayToStringNoSpace(classifierNames)+"_SUMMARY"+fileSuffix+".csv");
        for (int c = 0; c < combinedNames.length; c++)
            summ.writeString("," + combinedNames[c]);
        summ.writeLine("");
        
        summ.writeString("ranks:");
        for (int c = 0; c < rankmeans.length; c++)
            summ.writeString("," + rankmeans[c]);
        summ.writeString("\n");
        
        summ.writeString("accs:");
        for (int c = 0; c < accscombined.length; c++)
            summ.writeString("," + StatisticalUtilities.mean(accscombined[c], false));
        summ.writeString("\n\n");
        
        for (int c = 0; c < accscombined.length; c++)
            summ.writeString("," + combinedNames[c]);
        summ.writeLine("");
        
        for (int c1 = 0; c1 < accscombined.length; c1++) {
            summ.writeString(combinedNames[c1]);
            for (int c2 = 0; c2 < accscombined.length; c2++) {
                int wins=0, draws=0, losses=0;
                for (int d = 0; d < dsets.length; d++) {
                    
                    if (accscombined[c1][d] > accscombined[c2][d])
                        wins++;
                    else if ((accscombined[c1][d] == accscombined[c2][d]))
                        draws++;
                    else 
                        losses++;
                }
                summ.writeString(","+wins+"|"+draws+"|"+losses);
            }
            summ.writeLine("");
        }
        
        summ.writeString("\n\nPairwise tests:\n");
        summ.writeString(MultipleClassifiersPairwiseTest.runTests(arrayToStringNoSpace(classifierNames)+"_COMBINED"+fileSuffix+".csv").toString());
        
        summ.closeFile();
        
        
    }
    
    private static String arrayToStringNoSpace(String[] arr) {
        StringBuilder sb = new StringBuilder(arr[0]);
        
        for (int i = 1; i < arr.length; i++)
            sb.append(",").append(arr[i]);
        
        return sb.toString();
    }
    
    private static String[] stringToArray(String str, String delim) {
        String[] arr = str.trim().split(delim);
        return arr;
    }
    
    /**
     * @param accs [classifiers][acc on datasets]
     * @return [classifiers][rank on dataset]
     */
    private static double[][] findRanks(double[][] accs) {
        double[][] ranks = new double[accs.length][accs[0].length];
        
        for (int d = 0; d < accs[0].length; d++) {
            Double[] a = new Double[accs.length];
            for (int c = 0; c < accs.length; c++) 
                a[c] = accs[c][d];
            
            Arrays.sort(a, Collections.reverseOrder());
            
            for (int c1 = 0; c1 < accs.length; c1++) {
                for (int c2 = 0; c2 < accs.length; c2++) {
                    if (a[c1] == accs[c2][d]) {
                        ranks[c2][d] = c1+1; //count from one
                    }
                }
            }
        }
        
        return ranks;
    }
    
    public static void debugTest() throws Exception {
        
        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        Classifier[] cs = new Classifier[] { k };
        
        String[] cn = new String[] { "NN" };
        int fold = 5;
        String dset = "balloons";
        
        Instances all=ClassifierTools.loadData("C:/UCI Problems/"+dset+"/"+dset); 
        Instances[] data=InstanceTools.resampleInstances(all, fold, 0.5);
        
        buildAndWriteFullIndividualTrainTestResults(data[0], data[1], "lala/", dset, "asd", fold, cs, cn, null, true, true, false);
        
//        HESCA h = new HESCA(cs, cn);
//        h.setRandSeed(fold);
//        h.buildClassifier(data[0]);
    }
    
    public String buildEnsembleReport(boolean printPreds, boolean builtFromFile) {
        StringBuilder sb = new StringBuilder();
        
        sb.append("HESCA REPORT");
        sb.append("\nname: ").append(ensembleIdentifier);
        sb.append("\nmodules: ").append(classifierNames[0]);
        for (int i = 1; i < classifierNames.length; i++) 
            sb.append(",").append(classifierNames[i]);
        sb.append("\nweight scheme: ").append(weightingScheme);
        sb.append("\nvote scheme: ").append(votingScheme);
        sb.append("\ndataset: ").append(datasetName);
        sb.append("\nfold: ").append(resampleIdentifier);
        sb.append("\ntrain acc: ").append(ensembleTrainResults.acc);
        sb.append("\ntest acc: ").append(builtFromFile ? ensembleTestResults.acc : "NA");
        
        int precision = 4;
        int numWidth = precision+2;
        int trainAccColWidth = 8;
        int priorWeightColWidth = 12;
        int postWeightColWidth = 12;
        
        String moduleHeaderFormatString = "\n\n%20s | %"+(Math.max(trainAccColWidth, numWidth))+"s | %"+(Math.max(priorWeightColWidth, numWidth))+"s | %"+(Math.max(postWeightColWidth, this.numClasses*(numWidth+2)))+"s";
        String moduleRowHeaderFormatString = "\n%20s | %"+trainAccColWidth+"."+precision+"f | %"+priorWeightColWidth+"."+precision+"f | %"+(Math.max(postWeightColWidth, this.numClasses*(precision+2)))+"s";
        
        sb.append(String.format(moduleHeaderFormatString, "modules", "trainacc", "priorweights", "postweights"));
        for (EnsembleModule module : modules) {
            String postweights = String.format("  %."+precision+"f", module.posteriorWeights[0]);
            for (int c = 1; c < this.numClasses; c++) 
                postweights += String.format(", %."+precision+"f", module.posteriorWeights[c]);
            
            sb.append(String.format(moduleRowHeaderFormatString, module.getModuleName(), module.trainResults.acc, module.priorWeight, postweights));
        }
        
        
        if (printPreds) {
            sb.append("\n\nensemble train preds: ");
            sb.append("\ntrain acc: ").append(ensembleTrainResults.acc);
            sb.append("\n");
            for(int i = 0; i < ensembleTrainResults.numInstances();i++)
                sb.append(buildEnsemblePredsLine(true, i)).append("\n");

            sb.append("\n\nensemble test preds: ");
            sb.append("\ntest acc: ").append(builtFromFile ? ensembleTestResults.acc : "NA");
            sb.append("\n");
            for(int i = 0; i < ensembleTestResults.numInstances();i++)
                sb.append(buildEnsemblePredsLine(false, i)).append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * trueClassVal,predClassVal,[empty],dist1,...,distC,#indpreddist1,...,indpreddistC,#module1pred,...,moduleMpred
     * split on "#"
     * [0] = normal results file format (true class, pred class, distforinst)
     * [1] = number of individual unweighted votes per class
     * [2] = the unweighted prediction of each module
     */
    private String buildEnsemblePredsLine(boolean train, int index) {
        StringBuilder sb = new StringBuilder();
        
        if (train) //pred
            sb.append(modules[0].trainResults.actualClassValues.get(index)).append(",").append(ensembleTrainResults.predictedClassValues.get(index)).append(","); 
        else
            sb.append(modules[0].testResults.actualClassValues.get(index)).append(",").append(ensembleTestResults.predictedClassValues.get(index)).append(","); 
        
        if (train){ //dist
            double[] pred=ensembleTrainResults.getDistributionForInstance(index);
            for (int j = 0; j < pred.length; j++) 
                sb.append(",").append(pred[j]);
        }
        else{
            double[] pred=ensembleTestResults.getDistributionForInstance(index);
            for (int j = 0; j < pred.length; j++) 
                sb.append(",").append(pred[j]);
        }
        sb.append(",");
        
        
        double[] predDist = new double[numClasses]; //indpreddist
        for (int m = 0; m < modules.length; m++) {
            if (train) 
                ++predDist[(int)modules[m].trainResults.getPredClassValue(index)];
            else 
                ++predDist[(int)modules[m].testResults.getPredClassValue(index)];
        }
        for (int c = 0; c < numClasses; c++) 
            sb.append(",").append(predDist[c]);
        sb.append(",");
                
        for (int m = 0; m < modules.length; m++) {
            if (train) 
                sb.append(",").append(modules[m].trainResults.getPredClassValue(index));
            else 
                sb.append(",").append(modules[m].testResults.getPredClassValue(index));
        }   
        
        return sb.toString();
    }

}
