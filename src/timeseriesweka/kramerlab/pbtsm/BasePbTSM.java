package timeseriesweka.kramerlab.pbtsm;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import com.google.common.collect.Tables;

import java.io.File;
import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;

import java.util.function.Consumer;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import timeseriesweka.filters.SAX;

import weka.classifiers.AbstractClassifier;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.RandomCommittee;

import weka.classifiers.trees.ExtraTree;
import weka.classifiers.trees.RandomForest;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import weka.core.converters.ArffSaver;
import weka.core.converters.LibSVMSaver;

public abstract class BasePbTSM {
    
    private static Logger LOGGER = LoggerFactory.getLogger(BasePbTSM.class);

    protected final int EXECUTION_SEED;
    
    protected final int TOTAL_INSTANCES;
    
    protected PatternIndependenceTest independenceTest;
    
    protected PatternApproximationMethod approximationMethod;
    
    /**
     * Name of data set being processed.
     */
    private final String DATASET_NAME;

    /**
     * Number of trees per ensemble.
     */
    private final int NUM_TREES = 1000;
    
    /**
     * Number of execution threads.
     */
    private final int EXECUTION_THREADS;

    /**
     * Set of {@link timeseriesweka.kramerlab.FeatureSetType} values.
     */
    private final Set<FeatureSetType> FEATURE_SET_TYPES;

    /**
     * Map the "Class Values" in the data set to the corresponding Instance counts.
     */
    private final Map<Double, Integer> CLASS_VALUES_TO_INSTANCE_COUNTS = new TreeMap<>();

    /**
     * Map the {@link SplitType} to real valued {@link Instances} objects.
     */
    private final Map<SplitType, Instances> SPLITTYPE_TO_REAL_VALUED_INSTANCES = new EnumMap<>(SplitType.class);

    /**
     * Map of {@link timeseriesweka.kramerlab.SplitType} to Instance ClassLabels
     * (used for final feature set creation).
     */
    private final Map<SplitType, Instances> SPLITTYPE_TO_INST_CLASS_LABELS = new EnumMap<>(SplitType.class);

    /**
     * Map of {@link SplitType} to {@link Set} of classifications obtained from generated model.
     * 
     * Initialize a Map to save the Set of correctly classified instances
     * Used for parameter optimization and accuracy calculation
     */
    private final Map<SplitType, Set<Integer>> SPLITTYPE_TO_OBTAINED_CLASSIFICATIONS = new EnumMap<>(SplitType.class);
    
    /**
     * Map of {@link SplitType} to {@link com.google.common.collect.Table} of Transformation Time
     * values.
     * 
     * The Table object is keyed with Alpha (rows) and Window (column) values.
     */
    private final Map<SplitType, Table<Integer, Integer, Long>> SPLITTYPE_TO_DATA_TRANSFORMATION_TIME = new EnumMap<>(SplitType.class);

    /**
     * Map of {@link SplitType} to {@link com.google.common.collect.Table} of Stringified Instances.
     * The Table object is keyed with Alpha (rows) and Window (column) values.
     */
    private final Map<SplitType, Table<Integer, Integer, Instances>> SPLITTYPE_TO_TABLE_OF_SYM_INSTANCES = new EnumMap<>(SplitType.class);

    /**
     * {@link com.google.common.collect.Table} of feature set creation times.
     * The Table object is keyed with {@link timeseriesweka.kramerlab.SplitType} (rows)
     * and {@link timeseriesweka.kramerlab.FeatureSetType} (columns).
     */
    private final Table<SplitType, FeatureSetType, Long> TBL_FEATURE_SET_CREATION_TIMES = Tables.synchronizedTable(HashBasedTable.create());
    
    /**
     * A {@link com.google.common.collect.Table} of model accuracies keyed
     * with {@link timeseriesweka.kramerlab.SplitType} (rows) and 
     * {@link timeseriesweka.kramerlab.FeatureSetType} (columns).
     */
    private final Table<SplitType, FeatureSetType, Double> TBL_MODEL_ACCURACY_VALUES;
    
    /**
     * A {@link com.google.common.collect.Table} of final feature sets keyed
     * with {@link timeseriesweka.kramerlab.SplitType} (rows) and 
     * {@link timeseriesweka.kramerlab.FeatureSetType} (columns).
     */
    private final Table<SplitType, FeatureSetType, Instances> TBL_OF_FEATURE_SETS;
    
    /**
     * Table of training/testing times keyed with SplitType and FeatureSetType values
     * corresponding to each ClassifierType (in map).
     *
     * Initialize a Table object to save the training/testing times for each SplitType and FeatureSetType
     */
    private final Table<SplitType, FeatureSetType, Map<ClassifierType, Long>> TBL_OF_TRAIN_TEST_TIMES = HashBasedTable.create();
    
    /**
     * A {@link com.google.common.collect.Table} keyed with 
     * {@link timeseriesweka.kramerlab.SplitType} (rows) and 
     * {@link timeseriesweka.kramerlab.FeatureSetType} (columns).
     * 
     * Each cell of the {@link com.google.common.collect.Table} contains a
     * feature set for specific Alpha and Window values (used as keys for the
     * nested Table).
     * 
     * Initialize Table object for saving the Feature Set Instances objects for each
     * SplitType, FeatureSetType, Alpha and Window
     */
    private final Table<SplitType, FeatureSetType, Table<Integer, Integer, Instances>> TBL_OF_SPLITTYPE_FSTYPE_TO_TBL_FEATURESETS = Tables.synchronizedTable(HashBasedTable.create());

    /**
     * List of {@link AlphaOmegaPair} objects. Used to create discrete versions of the data.
     */
    private List<AlphaOmegaPair> listOfAlphaOmegaPairs;

    protected BasePbTSM(RealValuedDataset dataset, Set<FeatureSetType> fsTypes, int threads, int seed) {

        DATASET_NAME = dataset.getDatasetName();
        
        // Initialize FeatureSetTypes
        FEATURE_SET_TYPES = fsTypes;
        
        // Initialize thread count variable for controlling level of parallelization
        EXECUTION_THREADS = threads;

        EXECUTION_SEED = seed;
        
        this.setDataSplits(dataset.getShuffledDatasetSplits(this.getExecutionSeed()));

        TOTAL_INSTANCES = this.getClassLabelsToInstanceCounts().values().stream().reduce(0, Integer::sum);
        
        // Initialize a Table object to hold the TRANSFORMED feature set Instances objects for each 
        // SplitType and FeatureSetType
        TBL_OF_FEATURE_SETS = HashBasedTable.create();
        
        for (SplitType splitType : SplitType.values()) {
            // Create a single Attribute Instances object containing ClassLabels for
            // the corresponding split instances. Used for Feature Set creation
            SPLITTYPE_TO_INST_CLASS_LABELS.put(splitType,
                                               createOneColumnInstancesObjectWithClassLabelAttribute(splitType));
            
            SPLITTYPE_TO_OBTAINED_CLASSIFICATIONS.put(splitType, Sets.newTreeSet());
            
            // Place a Table Object for each SplitType
            // The table saved the transformation times for each Alpha/Window value pair
            SPLITTYPE_TO_DATA_TRANSFORMATION_TIME.put(splitType,
                                                      Tables.synchronizedTable(HashBasedTable.create()));
            
            // Place a Table object for each SplitType
            SPLITTYPE_TO_TABLE_OF_SYM_INSTANCES.put(splitType,
                                                    Tables.synchronizedTable(HashBasedTable.create()));
            
            this.getFeatureSetTypes().forEach(fsType ->
                    TBL_OF_TRAIN_TEST_TIMES.put(splitType, fsType, new EnumMap<>(ClassifierType.class)));

            this.getFeatureSetTypes().forEach(fsType -> 
                    TBL_OF_SPLITTYPE_FSTYPE_TO_TBL_FEATURESETS.put(splitType, fsType,
                                                                   Tables.synchronizedTable(HashBasedTable.create())));
        }
        
        TBL_MODEL_ACCURACY_VALUES = HashBasedTable.create();
    }

    public void combineFeatureSets(FeatureSetType featureSetType, List<AlphaOmegaPair> pairCombination) {
        Instances currSplitFeatureSet;
        for (SplitType splitType : SplitType.values()) {
            currSplitFeatureSet = null;
            for (AlphaOmegaPair pair : pairCombination) {
                if (currSplitFeatureSet == null) {
                    // currSplitFeatureSet = new Instances(this.getFeatureSet(splitType, featureSetType,alpha, window));
                    // SAFE: No need to create clone as above, since merging will create new instance objects
                    currSplitFeatureSet = this.getFeatureSet(splitType, featureSetType, pair);
                } else {
                    Instances currPairFeatureSet = this.getFeatureSet(splitType, featureSetType, pair);
                    currSplitFeatureSet = Instances.mergeInstances(currSplitFeatureSet, currPairFeatureSet);
                }
            }
            
            // To create SVM format data correctly add a dummy all-ones attribute before the folowing statement
            Attribute dummy = new Attribute("dummy");
            ArrayList<Attribute> listOfAttributes = new ArrayList<>();
            listOfAttributes.add(dummy);

            Instances dummyAttColumn = new Instances("", listOfAttributes, this.getInstanceClassLabelsOf(splitType).size());
            Instance newDestInst = new DenseInstance(dummyAttColumn.numAttributes());
            newDestInst.setDataset(dummyAttColumn);
            newDestInst.setValue(0, 1);
            
            this.getInstanceClassLabelsOf(splitType).forEach( i -> dummyAttColumn.add(newDestInst) );
            
            if (currSplitFeatureSet == null) {
                currSplitFeatureSet = dummyAttColumn;    // SAFE since merging will create new instance objects
            } else {
                currSplitFeatureSet = Instances.mergeInstances(currSplitFeatureSet, dummyAttColumn);
            }
            
            currSplitFeatureSet = Instances.mergeInstances(currSplitFeatureSet, this.getInstanceClassLabelsOf(splitType));
            currSplitFeatureSet.setClassIndex(currSplitFeatureSet.numAttributes()-1);
            currSplitFeatureSet.setRelationName(this.getDatasetName() + "_" + splitType.name() + "_" + featureSetType.name());
            this.setFeatureSet(splitType, featureSetType, currSplitFeatureSet);
        }
    }
    
    public void saveFinalFeatureSetData(ParameterSelectionMethod psm, PatternIndependenceTest pit, FeatureSetType f, int ppc) throws IOException {
        try {
            Path path = Paths.get(System.getProperty("user.dir"), "instances", this.getDatasetName());
            Files.createDirectories(path);
            ArffSaver saver = new ArffSaver();
            for (SplitType splitType : SplitType.values()) {
                saver.setInstances(this.getFeatureSet(splitType, f));

                saver.setFile(Paths.get(path.toString(),
                                        this.getDatasetName() + "_" + 
                                        splitType.name() + "_" + 
                                        psm.name() + "_" + 
                                        pit.name() + "_" + 
                                        f.name() + "_" + ppc + ".arff").toFile());
                saver.writeBatch();
            }
        } catch (IOException e) {
            LOGGER.error("Couldn't write feature set instances.\n{}", e.getMessage());
            throw e;
        }
    }

    public void trainClassifier(ClassifierType classifierType, FeatureSetType featureSetType) throws XGBoostError, Exception {
        this.trainClassifier(classifierType, featureSetType, -1);
    }
    
    public void trainClassifier(ClassifierType classifierType, FeatureSetType featureSetType, int numTrees) throws IOException, XGBoostError, Exception {
        for (SplitType splitType : SplitType.values()) {
            this.getClassificationsFor(splitType).clear();
            TBL_MODEL_ACCURACY_VALUES.put(splitType, featureSetType, 0.0);
            this.setTrainTestTimeInMilisecs(splitType, featureSetType, classifierType, 0L);
        }
        
        if (this.getFeatureSet(SplitType.TRAIN, featureSetType).numAttributes() <= 2) {
            return;
        }
        
        int numTreesToUse = (numTrees == -1) ? this.getNumOfTrees() : numTrees;
        
        switch (classifierType) {
            case XG:
                this.trainXGBoostClassifier(classifierType, featureSetType, numTreesToUse);
                break;
            case ET:
            case RF:
            case AB:
            default:
                this.trainWekaClassifier(classifierType, featureSetType, numTreesToUse);
                break;
        }
    }
    
    private void trainWekaClassifier(ClassifierType classifierType, FeatureSetType featureSetType, int numTrees) throws Exception {
        Instances trainSplit, testSplit;
        trainSplit = this.getFeatureSet(SplitType.TRAIN, featureSetType);
        testSplit = this.getFeatureSet(SplitType.TEST, featureSetType);
        
        AbstractClassifier classifier;
        switch(classifierType) {
            case AB:
                AdaBoostM1 adb = new AdaBoostM1();
                AbstractClassifier base = new RandomForest();
                ((RandomForest)base).setNumIterations(10);
                ((RandomForest)base).setNumExecutionSlots(this.getNumOfExecutionThreads());
                adb.setClassifier(base);
                adb.setNumIterations(numTrees);  //(trainSplit.numAttributes()-1)/(int)Math.log(trainSplit.numInstances())
                ((RandomForest)base).setSeed(this.getExecutionSeed());
                adb.setSeed(this.getExecutionSeed());
                classifier = adb;
                break;
            case RF:
                RandomForest rf = new RandomForest();
                rf.setNumIterations(numTrees);
                rf.setNumExecutionSlots(this.getNumOfExecutionThreads());
                rf.setSeed(this.getExecutionSeed());
                classifier = rf;
                break;
            case ET:
            default:
                RandomCommittee committee = new RandomCommittee();
                committee.setNumIterations(numTrees);
                committee.setNumExecutionSlots(this.getNumOfExecutionThreads());
                ExtraTree ert = new ExtraTree();
                committee.setSeed(this.getExecutionSeed());
                ert.setSeed(this.getExecutionSeed());
                committee.setClassifier(ert);
                classifier = committee;
                break;
        }
        
        try {
            long trainingTime, testingTime;
            double trainAcc, testAcc;
            trainingTime = System.currentTimeMillis();
            classifier.buildClassifier(trainSplit);
            trainingTime = System.currentTimeMillis() - trainingTime;
            
            trainAcc = this.classifySplit(classifier, trainSplit, SplitType.TRAIN);
            
            TBL_MODEL_ACCURACY_VALUES.put(SplitType.TRAIN, featureSetType, trainAcc);
            this.setTrainTestTimeInMilisecs(SplitType.TRAIN, featureSetType, classifierType, trainingTime);
//            this.getRuntimeStatsMap(SplitType.TRAIN, featureSetType).put(classifierType, trainingTime);
            
            testingTime = System.currentTimeMillis();
            testAcc = this.classifySplit(classifier, testSplit, SplitType.TEST);
            testingTime = System.currentTimeMillis() - testingTime;
            
            TBL_MODEL_ACCURACY_VALUES.put(SplitType.TEST, featureSetType, testAcc);
            this.setTrainTestTimeInMilisecs(SplitType.TEST, featureSetType, classifierType, testingTime);
//            this.getRuntimeStatsMap(SplitType.TEST, featureSetType).put(classifierType, testingTime);
        } catch (Exception e) {
            throw e;
        }
    }
    
    private void trainXGBoostClassifier(ClassifierType classifierType, FeatureSetType featureSetType, int numTrees) throws IOException, XGBoostError {
        Instances trainSplit, testSplit;
        trainSplit = this.getFeatureSet(SplitType.TRAIN, featureSetType);
        testSplit = this.getFeatureSet(SplitType.TEST, featureSetType);
        String TEMP_DIR = System.getProperty("user.dir");
        LibSVMSaver svmSaver = new LibSVMSaver();
        String trainSetSVMFile = (new File(TEMP_DIR, "trainset.svm.txt")).getAbsolutePath(),
               testSetSVMFile = (new File(TEMP_DIR, "testset.svm.txt")).getAbsolutePath();
        try {
            svmSaver.setInstances(trainSplit);
            svmSaver.setFile(new File(trainSetSVMFile));
            svmSaver.writeBatch();
            svmSaver.setInstances(testSplit);
            svmSaver.setFile(new File(testSetSVMFile));
            svmSaver.writeBatch();
        } catch (IOException e) {
            throw new IOException("Error writing libSVM versions of Feature Datasets", e);
        }
        
        HashMap<String, Object> params = new HashMap<>();
//        params.put("eta", 0.1);
//        params.put("max_depth", 2);
//        params.put("subsample", 0.7);
//        params.put("colsample_bylevel", 0.7);
        params.put("silent", 1);
        params.put("nthread", this.getNumOfExecutionThreads());
        if (this.getClassLabelsToInstanceCounts().size() == 2) {
            params.put("objective", "binary:logistic");
            params.put("eval_metric", "error");
        } else {
            params.put("objective", "multi:softmax");
            params.put("num_class", Integer.toString(this.getClassLabelsToInstanceCounts().size()));
            params.put("eval_metric", "merror");
        }
        
        DMatrix trainMat = null, testMat = null;
        try {
            trainMat = new DMatrix(trainSetSVMFile);
            testMat = new DMatrix(testSetSVMFile);
        } catch (XGBoostError e) {
            System.err.println(e);
        }
        
        HashMap<String, DMatrix> watches = new HashMap<>();
        watches.put("train", trainMat);
        watches.put("test", testMat);
        
        try {
            long trainingTime, testingTime;
            double trainAcc, testAcc;
            Booster booster;
            trainingTime = System.currentTimeMillis();
            booster = XGBoost.train(trainMat, params, numTrees, watches, null, null, null, 5);
            trainingTime = System.currentTimeMillis() - trainingTime;
            
            trainAcc = classifySplit(booster, trainMat, SplitType.TRAIN);
            
            TBL_MODEL_ACCURACY_VALUES.put(SplitType.TRAIN, featureSetType, trainAcc);
            this.setTrainTestTimeInMilisecs(SplitType.TRAIN, featureSetType, classifierType, trainingTime);
//            this.getRuntimeStatsMap(SplitType.TRAIN, featureSetType).put(classifierType, trainingTime);
            
            testingTime = System.currentTimeMillis();
            testAcc = classifySplit(booster, testMat, SplitType.TEST);
            testingTime = System.currentTimeMillis() - testingTime;
            
            TBL_MODEL_ACCURACY_VALUES.put(SplitType.TEST, featureSetType, testAcc);
            this.setTrainTestTimeInMilisecs(SplitType.TEST, featureSetType, classifierType, testingTime);
//            this.getRuntimeStatsMap(SplitType.TEST, featureSetType).put(classifierType, testingTime);    
        } catch (XGBoostError e) {
            throw e;
        }
        
        Arrays.stream(Paths.get(TEMP_DIR).toFile().listFiles()).forEach(File::delete);
    }
    
    private double classifySplit(AbstractClassifier classifier, Instances featureSetSplit, SplitType splitType) throws Exception {
        Set<Integer> classifications = this.getClassificationsFor(splitType);
        int classInd = featureSetSplit.numAttributes()-1;
        try {
            double result;
            for (int ind = 0; ind < featureSetSplit.numInstances(); ind++) {
                result = classifier.classifyInstance(featureSetSplit.get(ind));
                if (this.nearlyEqual(result, featureSetSplit.get(ind).value(classInd), 1e-6)) {
                    classifications.add(ind);
                }
            }
        } catch (Exception e) {
            LOGGER.error("Exception occurred while classifying feature set instance.\n{}", e.getMessage());
            throw e;
        }
        return 100.0 * classifications.size() / featureSetSplit.numInstances();
    }
    
    private double classifySplit(Booster booster, DMatrix fbSplit, SplitType splitType) throws XGBoostError {
        Set<Integer> classifications = this.getClassificationsFor(splitType);
        try {
            float[][] predictions = booster.predict(fbSplit);
            float[] labels = fbSplit.getLabel();
            for (int i = 0; i < labels.length; i++) {
                if (this.getClassLabelsToInstanceCounts().size() == 2) {
                    if ( predictions[i][0] < 0.5 ) {
                        if ( this.nearlyEqual(labels[i], 0.0, 1e-6) ) {
                            classifications.add(i);
                        }
                    } else {
                        if ( this.nearlyEqual(labels[i], 1.0, 1e-6) ) {
                            classifications.add(i);
                        }
                    }
                } else {
                    if (this.nearlyEqual(labels[i], predictions[i][0], 1e-6)) {
                        classifications.add(i);
                    }
                }
            }
            
            return 100.0*classifications.size()/labels.length;
        } catch (XGBoostError e) {
            LOGGER.error("Exception occurred while classifying feature set instance.\n{}", e.getMessage());
            throw e;
        }
    }
    
    public double getSplitAccuracy(SplitType splitType, FeatureSetType featureSetType) {
        return TBL_MODEL_ACCURACY_VALUES.get(splitType, featureSetType);
    }
    
    protected final void setAlphaOmegaPairs(List<AlphaOmegaPair> pairs) {
        listOfAlphaOmegaPairs = pairs;
    }

    protected final String getDatasetName() {
        return DATASET_NAME;
    }

    protected final int getNumOfExecutionThreads() {
        return EXECUTION_THREADS;
    }
    
    protected final int getExecutionSeed() {
        return EXECUTION_SEED;
    }

    protected final int getNumOfTrees() {
        return NUM_TREES;
    }
    
    protected final void setFeatureSet(SplitType splitType, FeatureSetType fsType, AlphaOmegaPair pair, Instances featureSet) {
        TBL_OF_SPLITTYPE_FSTYPE_TO_TBL_FEATURESETS.get(splitType, fsType)
                                                  .put(pair.getAlpha(), pair.getOmega(), featureSet);
    }
    
    protected final Instances getFeatureSet(SplitType splitType, FeatureSetType fsType, AlphaOmegaPair pair) {
        return TBL_OF_SPLITTYPE_FSTYPE_TO_TBL_FEATURESETS.get(splitType, fsType)
                                                         .get(pair.getAlpha(), pair.getOmega());
    }
    
    protected final Instances getInstanceClassLabelsOf(SplitType splitType) {
        return SPLITTYPE_TO_INST_CLASS_LABELS.get(splitType);
    }
    
    protected final Set<Integer> getClassificationsFor(SplitType splitType) {
        return SPLITTYPE_TO_OBTAINED_CLASSIFICATIONS.get(splitType);
    }
    
    protected final long getTrainTestTimeInMilisecs(SplitType splitType, FeatureSetType featureSetType, ClassifierType clsType) {
        return TBL_OF_TRAIN_TEST_TIMES.get(splitType, featureSetType).get(clsType);
    }
    
    protected final void setTrainTestTimeInMilisecs(SplitType splitType, FeatureSetType fsType, ClassifierType clsType, long runtime) {
        TBL_OF_TRAIN_TEST_TIMES.get(splitType, fsType).put(clsType, runtime);
    }
    
    protected final long getFeatureSetCreationTimeInMilisecs(SplitType splitType, FeatureSetType fsType) {
        return TBL_FEATURE_SET_CREATION_TIMES.get(splitType, fsType);
    }
    
    protected final void setFeatureSetCreationTimeInMilisecs(SplitType splitType, FeatureSetType fsType, long milisecs) {
        TBL_FEATURE_SET_CREATION_TIMES.put(splitType, fsType, milisecs);
    }
    
    protected final void resetFeatureSetCreationTimes() {
        for (SplitType splitType : SplitType.values()) {
            this.getFeatureSetTypes().forEach(fsType ->
                    TBL_FEATURE_SET_CREATION_TIMES.put(splitType, fsType, 0L));
        }
    }
    
    protected final int lengthOfRealValTSInst() {
        return (this.getRealValuedInstsOf(SplitType.TRAIN).numAttributes() - 1);
    }

    protected final Instances getRealValuedInstsOf(SplitType splitType) {
        return SPLITTYPE_TO_REAL_VALUED_INSTANCES.get(splitType);
    }

    protected final Instances getFeatureSet(SplitType splitType, FeatureSetType fsType) {
        return TBL_OF_FEATURE_SETS.get(splitType, fsType);
    }
    
    protected final void setFeatureSet(SplitType splitType, FeatureSetType fsType, Instances featureSet) {
        TBL_OF_FEATURE_SETS.put(splitType, fsType, featureSet);
    }
    
    protected final List<AlphaOmegaPair> getAlphaOmegaPairs() {
        return listOfAlphaOmegaPairs;
    }

    protected final Map<Double, Integer> getClassLabelsToInstanceCounts() {
        return CLASS_VALUES_TO_INSTANCE_COUNTS;
    }
    
    protected final Instances getSymbolizedInsts(SplitType splitType, AlphaOmegaPair awPair) {
        return SPLITTYPE_TO_TABLE_OF_SYM_INSTANCES.get(splitType)
                                                  .get(awPair.getAlpha(), awPair.getOmega());
    }
    
    protected final void setSymbolizedInsts(SplitType splitType, AlphaOmegaPair awPair, Instances symSplit) {
        SPLITTYPE_TO_TABLE_OF_SYM_INSTANCES.get(splitType)
                                           .put(awPair.getAlpha(), awPair.getOmega(), symSplit);
    }
    
    public long getTotalDataTransformationTime(SplitType st) {
        return SPLITTYPE_TO_DATA_TRANSFORMATION_TIME.get(st).cellSet()
                                                    .stream().mapToLong(c -> c.getValue()).sum();
    }
    
    protected final long getDataTransformationTime(SplitType st, AlphaOmegaPair p) {
        return SPLITTYPE_TO_DATA_TRANSFORMATION_TIME.get(st).get(p.getAlpha(), p.getOmega());
    }

    protected final void setDataTransformationTime(SplitType st, AlphaOmegaPair p, long milisecs) {
        SPLITTYPE_TO_DATA_TRANSFORMATION_TIME.get(st).put(p.getAlpha(), p.getOmega(), milisecs);
    }

    protected final void transformData(AlphaOmegaPair pair) throws Exception {
        SAX saxer = new SAX();
        saxer.useRealValuedAttributes(false);
        saxer.setAlphabetSize(pair.getAlpha());
        saxer.setNumIntervals(this.lengthOfRealValTSInst() / pair.getOmega());

        Instances realValuedSplit, saxedSplit, symbolizedSplit;

        long timer;

        try {
            for (SplitType splitType : SplitType.values()) {
                timer = System.currentTimeMillis();

                realValuedSplit = this.getRealValuedInstsOf(splitType);

                saxedSplit = saxer.process(realValuedSplit);
                
                saxedSplit.setRelationName(realValuedSplit.relationName()
                                           + " - A=" + pair.getAlpha() + ", W=" + pair.getOmega());
                
                symbolizedSplit = this.getSymbolizedForm(saxedSplit);
                
                timer = System.currentTimeMillis() - timer;
                
                this.setSymbolizedInsts(splitType, pair, symbolizedSplit);

                this.setDataTransformationTime(splitType, pair, timer);
            }
        } catch (Exception e) {
            LOGGER.error("Exception occured while transforming data to SAX"
                         + "Alpha: {}, Step: {}.\n{}", pair.getAlpha(), pair.getOmega(), e.getMessage());
            throw e;
        }
    }
    
    protected final boolean nearlyEqual(double a, double b, double epsilon) {
        if (a == b) { // shortcut, handles infinities
            return true;
        }
        
        final double diff = Math.abs(a - b);

        if (a == 0 || b == 0 || diff < Double.MIN_NORMAL) {
            // a or b is zero or both are extremely close to it
            // relative error is less meaningful here
            return diff < (epsilon * Double.MIN_NORMAL);
        } else { // use relative error
            final double absA = Math.abs(a);
            final double absB = Math.abs(b);
            return diff / Math.min((absA + absB), Double.MAX_VALUE) < epsilon;
        }
    }
    
    protected final double[] getBreakpoints(AlphaOmegaPair pair) throws Exception {
        double[] breakpoints = null;
        try {
            // Create SAX object
            SAX saxer = new SAX();
            saxer.useRealValuedAttributes(false);
            saxer.setAlphabetSize(pair.getAlpha());
            saxer.setNumIntervals(this.lengthOfRealValTSInst() / pair.getOmega());
            breakpoints = saxer.generateBreakpoints(pair.getAlpha());
        } catch (Exception e) {
            throw new Exception("Error getting breakpoints.", e);
        }
        return breakpoints;
    }
    
    protected final double[] getRealValInstanceAsArray(int ind, SplitType splitType) {
        Instance inst = this.getRealValuedInstsOf(splitType).get(ind);
        double[] temp = new double[inst.numAttributes()-1];
        for (int attIndex = 0; attIndex < inst.numAttributes()-1; attIndex++) {
            temp[attIndex] = inst.value(attIndex);
        }
        return temp;
    }
    
    protected final double getMinimumDistance(double[] inst, Double[] shapelet) {
        double shortestDistance = Double.MAX_VALUE, currentDistance, difference;
        boolean computationNotAbandonedEarly;
        for (int i = 0; i < inst.length - shapelet.length + 1; i++) {
            currentDistance = 0;
            computationNotAbandonedEarly = true;
            for (int j = 0; j < shapelet.length; j++) {
                difference = shapelet[j] - inst[i+j];
                currentDistance += difference*difference;
                if (currentDistance > shortestDistance) {
                    computationNotAbandonedEarly = false;
                    break;
                }
            }
            if (computationNotAbandonedEarly) {
                shortestDistance = currentDistance;
            }
        }
        return shortestDistance;
    }
    
    protected final Double[] getBestApproximateShapelet(AlphaOmegaPair pair, String symForm, double[] breakpoints, Instances symTrainSet) {
        double[] realValuedPattern = new double[symForm.length()];
        int ind;
        for (int i = 0; i < symForm.length(); i++) {
            ind = symForm.charAt(i)-'a';
            if (ind == 0) {
                realValuedPattern[i] = breakpoints[0];
            } else if (ind == breakpoints.length-1) {
                realValuedPattern[i] = breakpoints[breakpoints.length-2];
            } else {
                realValuedPattern[i] = (breakpoints[ind]+breakpoints[ind-1]) / 2;
            }
        }
        
        ArrayList<Double> interpolatedPattern = new ArrayList<>();
        switch (approximationMethod) {
            case S:     // SAFE and just a single time point shorter than the original
                for (int i = 0; i < symForm.length(); i++) {
                    for (int j = 0; j < pair.getOmega(); j++) {
                        interpolatedPattern.add(realValuedPattern[i]);
                    }
                }
                break;
            case I:
                double stepSize = ( this.lengthOfRealValTSInst()/pair.getOmega()) / (double)this.lengthOfRealValTSInst();
                int xi;
                double x, p0, p1, p2, p3, temp;
                for (double idx = 0.0; idx < symForm.length(); idx += stepSize) {
                    x = idx;
                    xi = (int) x;
                    x -= xi;
                    p0 = realValuedPattern[Math.max(0, xi - 1)];
                    p1 = realValuedPattern[xi];
                    p2 = realValuedPattern[Math.min(realValuedPattern.length - 1, xi + 1)];
                    p3 = realValuedPattern[Math.min(realValuedPattern.length - 1, xi + 2)];
                    temp = p1 + 0.5 * x
                                * (p2 - p0 + x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                                                  + x * (3.0 * (p1 - p2) + p3 - p0)));
                    interpolatedPattern.add(temp);
                }
                break;
        }
        
        Double[] appShapelet = interpolatedPattern.toArray(new Double[interpolatedPattern.size()]);
        Double[] bestApproxShapelet = new Double[appShapelet.length];
        double[] inst, subsequence = new double[appShapelet.length];
        double dist = Double.POSITIVE_INFINITY, tempDist;
        int bestTimepoint = 0;
//        int bestInst = 0;
        boolean copied = true;
        int indexOfFreqPatt;
        for (int i = 0; i < symTrainSet.numInstances(); i++) {
            indexOfFreqPatt = symTrainSet.get(i).stringValue(0).indexOf(symForm);
            if ( indexOfFreqPatt >= 0 ) {
//                // 2018-01-18
//                System.out.println("Instance num: " + i + ", Pattern Index: " + indexOfFreqPatt);
                inst = this.getRealValInstanceAsArray(i, SplitType.TRAIN);
                // To reduce the searching cost even further
                // Instead of searching in the entire length of the real-valued instance,
                // could use the approximate location at the start of the match
                // Would require to change the if condition to (symTrainSet[i].indexOf(symForm) > 0)
                // instead of contains(...)
                // Then the start of the search in the following loop would be from j = indexOf(...)*pair.getOmega()
                for (int j = 0; j < inst.length - appShapelet.length + 1; j++) {
                    // Using following loop instead of Arrays.copyOfRange allows to reuse array
                    for (int k = 0; k < subsequence.length; k++) {
                        subsequence[k] = inst[j+k];
                    }
                    tempDist = this.getMinimumDistance(subsequence, appShapelet);
                    if (tempDist < dist) {
                        dist = tempDist;
                        bestTimepoint = j;
                        copied = false;
                    }
                }
                if (!copied) {
//                    bestInst = i;
                    for (int k = 0; k < appShapelet.length; k++) {
                        bestApproxShapelet[k] = inst[bestTimepoint + k];
                    }
                    copied = true;
                }
            }
        }
//        // 2018-01-18
//        System.out.print("Best found - Instance: " + bestInst + ", subsequence index: " + bestTimepoint + ", Seq: ");
//        Arrays.asList(bestApproxShapelet).stream().forEachOrdered((e) -> System.out.format("%.3f, ", e));
//        System.out.println();
        
        return bestApproxShapelet;
    }
    
    protected final void createAllFeatureSets(PatternApproximationMethod approxMethod, PatternIndependenceTest indTest) {
        independenceTest = indTest;
        
        approximationMethod = approxMethod;
        
        this.resetFeatureSetCreationTimes();
        
        ForkJoinPool threads = new ForkJoinPool(this.getNumOfExecutionThreads());
        
        try {
            threads.submit(() -> this.getAlphaOmegaPairs()
                                     .parallelStream()
                                     .forEach( throwingConsumerWrapper( pair -> createIndividualFeatureSet(pair) ) )
                          )
                   .get();
        } catch (InterruptedException|ExecutionException e) {
            LOGGER.error("Interrupted or Execution exception.\n{}", e.getMessage());
        } finally {
            threads.shutdown();
        }
    }

    protected final void createIndividualFeatureSet(AlphaOmegaPair pair) throws Exception {
        ArrayList<Attribute> attributes = this.getTopFrequentPatternAttributes(pair);
        
        Instances featureDataTemplate = new Instances("", attributes, 0);
        
        for (SplitType splitType : SplitType.values()) {
            this.getFeatureSetTypes().forEach(fsType ->
            {
                Instances blankFeatureSet = new Instances(featureDataTemplate);
                
                for (int instCount = 0 ; instCount < this.getRealValuedInstsOf(splitType).numInstances(); instCount++) {
                    Instance inst = new DenseInstance(blankFeatureSet.numAttributes());
                    inst.setDataset(blankFeatureSet);
                    blankFeatureSet.add(inst);
                }
                this.setFeatureSet(splitType, fsType, pair, blankFeatureSet);
            });
        }
        
        double[] breakpoints = this.getBreakpoints(pair);
        Double[] shapelet;
        Instances symbolizedSet;
        
        String freqString, attributeName;
        double featureValue;
        
        Table<SplitType, FeatureSetType, Long> timeTable = HashBasedTable.create();
        
        for (SplitType splitType: SplitType.values()) {
            this.getFeatureSetTypes().forEach(fsType -> timeTable.put(splitType, fsType, 0L) );
        }
        
        long timer;
        
//        // 2018-01-18
//        System.out.println("Time series length: " + this.lengthOfRealValTSInst() + ", Alpha: " + pair.getAlpha() + ", Window: " + pair.getOmega());
        
        for (int attInd = 0; attInd < attributes.size(); attInd++) {
            attributeName = attributes.get(attInd).name();
            freqString = attributeName.substring(attributeName.lastIndexOf("_")+1); //, 20
            
            symbolizedSet = this.getSymbolizedInsts(SplitType.TRAIN, pair);
            shapelet = this.getBestApproximateShapelet(pair, freqString, breakpoints, symbolizedSet);
            
            for (SplitType splitType : SplitType.values()) {
                symbolizedSet = this.getSymbolizedInsts(splitType, pair);
                for (FeatureSetType fsType : this.getFeatureSetTypes()) {
                    timer = System.currentTimeMillis();
                    
                    Instances currFeatureSet = this.getFeatureSet(splitType, fsType, pair);
                    
                    for (int instInd = 0; instInd < currFeatureSet.numInstances(); instInd++) {
                        
                        switch (fsType) {
                            case NUM:
                                featureValue = this.getMinimumDistance(this.getRealValInstanceAsArray(instInd, splitType), shapelet);
                                break;
                            case BIN:
                                featureValue = symbolizedSet.get(instInd).stringValue(0).contains(freqString) ? 1 : 0;
                                break;
                            default:
                                featureValue = 0.0;
                                break;
                        }
                        currFeatureSet.get(instInd).setValue(attInd, featureValue);
                    }
                    timer = (System.currentTimeMillis() - timer) + timeTable.get(splitType, fsType);
                    timeTable.put(splitType, fsType, timer);
                }
            }
        }
        
        for (SplitType splitType: SplitType.values()) {
            this.getFeatureSetTypes().forEach(fsType ->
                    this.setFeatureSetCreationTimeInMilisecs(splitType, fsType,
                                                   timeTable.get(splitType, fsType)
                                                   + this.getFeatureSetCreationTimeInMilisecs(splitType, fsType)));
        }
    }

    protected long getTotalTime(SplitType splitType, FeatureSetType featureSetType, ClassifierType classifierType) {
        long total = this.getTotalDataTransformationTime(splitType)
                     + this.getFeatureSetCreationTimeInMilisecs(splitType, featureSetType)
                     + this.getTrainTestTimeInMilisecs(splitType, featureSetType, classifierType);
        return total;
    }
    
    protected abstract void performPreprocessing() throws IOException;
    
    protected abstract ArrayList<Attribute> getTopFrequentPatternAttributes(AlphaOmegaPair pair);
    
    @FunctionalInterface
    protected interface ThrowingConsumer<T, E extends Exception> {

        void accept(T t) throws E;
    }

    protected <T> Consumer<T> throwingConsumerWrapper(ThrowingConsumer<T, Exception> throwingConsumer) {
        return i -> {
            try {
                throwingConsumer.accept(i);
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        };
    }

    private void setDataSplits(Instances[] instancesArray) {
        SPLITTYPE_TO_REAL_VALUED_INSTANCES.put(SplitType.TRAIN, instancesArray[0]);
        SPLITTYPE_TO_REAL_VALUED_INSTANCES.put(SplitType.TEST, instancesArray[1]);

        Map<Double, Integer> destMap = this.getClassLabelsToInstanceCounts();
        destMap.clear();

        Instances trainSplit = this.getRealValuedInstsOf(SplitType.TRAIN);

        // Enumerate all training set instances and their counts
        trainSplit.forEach(in -> destMap.compute(in.classValue(), (k, v) -> v == null ? 1 : v + 1));
    }

    private Set<FeatureSetType> getFeatureSetTypes() {
        return FEATURE_SET_TYPES;
    }
    
    /**
     * Creates a single column Instances object for the provided SplitType
     * which is subsequently used when the feature sets are combined/merged
     * 
     * @param splitType
     * @return 
     */
    private Instances createOneColumnInstancesObjectWithClassLabelAttribute(SplitType splitType) {
        Instances srcSplit = this.getRealValuedInstsOf(splitType);
        
        Attribute targetAttribute = this.getRealValuedInstsOf(splitType).classAttribute();
        
        ArrayList<Attribute> listOfAttributes = new ArrayList<>()
        {{
            add(targetAttribute);
        }};
        
        
        Instances template = new Instances("", listOfAttributes, 0);
        template.setClassIndex(0);
        
        Instances destSplit = new Instances(template, srcSplit.numInstances());
        
        Instance newDestInst;
        
        for (Instance currSrcInst : srcSplit) {
            newDestInst = new DenseInstance(destSplit.numAttributes());
            newDestInst.setDataset(destSplit);
            newDestInst.setClassValue(currSrcInst.classValue());
            
            destSplit.add(newDestInst);
        }
        
        return destSplit;
    }
    
    private Instances getSymbolizedForm(Instances saxedSplit) {
        int totalInstances = saxedSplit.numInstances();
        int totalAttributes = saxedSplit.numAttributes() - 1;

        Attribute stringAttribute = new Attribute("symbolic-instance", true);
        Attribute targetAttribute = this.getRealValuedInstsOf(SplitType.TRAIN).classAttribute();

        ArrayList<Attribute> listOfAttributes = new ArrayList<Attribute>() {
            {
                add(stringAttribute);
                add(targetAttribute);
            }
        };

        Instances symbolicDataset = new Instances(saxedSplit.relationName(), listOfAttributes, 0);
        symbolicDataset.setClassIndex(listOfAttributes.size() - 1);

        StringBuilder currSymbolizedInstance = new StringBuilder(totalAttributes * 2);
        Instance currSAXedInstance;
        Instance newDestInst;
        for (int instanceIndex = 0; instanceIndex < totalInstances; instanceIndex++) {
            currSAXedInstance = saxedSplit.get(instanceIndex);
            currSymbolizedInstance.setLength(0);

            for (int attributeIndex = 0; attributeIndex < totalAttributes; attributeIndex++) {
                currSymbolizedInstance.append(currSAXedInstance.stringValue(attributeIndex));
            }

            newDestInst = new DenseInstance(symbolicDataset.numAttributes());
            newDestInst.setDataset(symbolicDataset);
            newDestInst.setValue(0, currSymbolizedInstance.toString());
            newDestInst.setClassValue(currSAXedInstance.classValue());
            symbolicDataset.add(newDestInst);
        }
        return symbolicDataset;
    }
}
