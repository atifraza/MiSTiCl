package timeseriesweka.kramerlab.pbtsm.misticl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import com.google.common.collect.Tables;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import org.apache.commons.math3.stat.inference.ChiSquareTest;

import timeseriesweka.filters.SAX;

import timeseriesweka.kramerlab.pbtsm.AlphaWindowPair;
import timeseriesweka.kramerlab.pbtsm.BasePbTSM;
import timeseriesweka.kramerlab.pbtsm.RealValuedDataset;
import timeseriesweka.kramerlab.pbtsm.SplitType;

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

/**
 *
 * @author atif
 */
public class MiSTiCl extends BasePbTSM {
    
    /**
     * Math.log(2) Constant. Used to convert LOG_e(x) to LOG_2(x) for entropy calculation
     */
    private final double LOG2 = Math.log(2);
    
    /**
     * Number of trees per ensemble.
     */
    private final int NUM_TREES = 1000;
    
    /**
     * Name of string miner executable.
     */
    private final String STRING_MINER;
    
    /**
     * Log of V value for string miner. Should be an integer.
     */
    private final String LOG_V;
    
    /**
     * Block size value for string miner. Should be an integer.
     */
    private final String BLOCK_SIZE;
    
    /**
     * Minimum positive frequency for patterns in the positive class split.
     */
    private final String MIN_POS_FREQ;
    
    /**
     * Maximum negative frequency for patterns in the negative class split.
     */
    private final String MAX_NEG_FREQ;

    /**
     * Path to temporary directory.
     */
    private final String TEMP_DIR;
    
    /**
     * Time out value for pattern extraction process.
     */
    private final long TIMEOUT_VAL;

    /**
     * Unit of time for Time out value.
     */
    private final TimeUnit TIMEOUT_UNIT;
    
    /**
     * Set of {@link timeseriesweka.kramerlab.FeatureSetType} values.
     */
    private final Set<FeatureSetType> FEATURE_SET_TYPES;

    /**
     * Map of {@link timeseriesweka.kramerlab.SplitType} to Instance ClassLabels
     * (used for final feature set creation).
     */
    private final Map<SplitType, Instances> map_SplitType_InstanceClassLabels;

    /**
     * {@link com.google.common.collect.Table} of Frequent pattern extraction
     * times.
     * The Table object is keyed with Alpha (rows) and Window (column) values.
     */
    private final Table<Integer, Integer, Long> tbl_FreqPattExtractionTimes;

    /**
     * {@link java.util.Map} of
     * {@link timeseriesweka.kramerlab.PatternIndependenceTest}
     * to {@link com.google.common.collect.Table} of frequent patterns.
     * The Table object is keyed with Alpha (rows) and Window (column) values.
     * Each {@link com.google.common.collect.Table} entry is a
     * {@link java.util.Map} of
     * ClassValues to {@link java.util.Map} of independence test values to a
     * {@link java.util.List} of Strings
     */
    private final Map<PatternIndependenceTest, Table<Integer, Integer, Map<Double, Map<Double, List<String>>>>> map_PattIndTest_Tbl_FreqPatts;
    
    /**
     * {@link java.util.Map} of
     * {@link timeseriesweka.kramerlab.PatternIndependenceTest}
     * to pattern filtering time
     */
    private final Map<PatternIndependenceTest, Long> map_PattIndTest_PatternFilterringTime;
    
    /**
     * {@link com.google.common.collect.Table} of feature set creation times.
     * The Table object is keyed with {@link timeseriesweka.kramerlab.SplitType} (rows)
     * and {@link timeseriesweka.kramerlab.FeatureSetType} (columns).
     */
    private final Table<SplitType, FeatureSetType, Long> tbl_FeatureSetCreationTimes;
    
    /**
     * A {@link com.google.common.collect.Table} keyed with 
     * {@link timeseriesweka.kramerlab.SplitType} (rows) and 
     * {@link timeseriesweka.kramerlab.FeatureSetType} (columns).
     * 
     * Each cell of the {@link com.google.common.collect.Table} contains a
     * feature set for specific Alpha and Window values (used as keys for the
     * nested Table).
     */
    private final Table<SplitType, FeatureSetType, Table<Integer, Integer, Instances>> map_SplitType_FeatureSetType_Tbl_Instances;
    
    /**
     * A {@link com.google.common.collect.Table} of final feature sets keyed
     * with {@link timeseriesweka.kramerlab.SplitType} (rows) and 
     * {@link timeseriesweka.kramerlab.FeatureSetType} (columns).
     */
    private final Table<SplitType, FeatureSetType, Instances> tbl_FeatureSets;
    
    /**
     * A {@link com.google.common.collect.Table} of model accuracies keyed
     * with {@link timeseriesweka.kramerlab.SplitType} (rows) and 
     * {@link timeseriesweka.kramerlab.FeatureSetType} (columns).
     */
    private final Table<SplitType, FeatureSetType, Double> tbl_ModelAccuracyVals;
    
    // Table of training/testing times keyed with SplitType and FeatureSetType values
    // corresponding to each ClassifierType (in map)

    /**
     *
     */
    private final Table<SplitType, FeatureSetType, Map<ClassifierType, Long>> tbl_TrainTestTime;
    
    // Map of correct classification sets for SplitType values
    // used for parameter optimization and model accuracy calculations

    /**
     *
     */
    private final Map<SplitType, Set<Integer>> map_SplitClassifications;
    
    private PatternIndependenceTest independenceTest;
    
    private PatternApproximationMethod approximationMethod;
    
    private int maxPatternsPerClass;
    
    private final int seed;
    
    public MiSTiCl(RealValuedDataset rvDataset, List<AlphaWindowPair> awPairs, int numThreads, int seed, String tempDir,
                   String stringMinerFileName, int log_v, int blockSize, double min_pos_freq, double max_neg_freq,
                   Set<PatternIndependenceTest> pattIndTests, Set<FeatureSetType> fsTypes) throws RuntimeException, IOException {
        
        super(rvDataset.getDatasetName(), rvDataset.getShuffledDataset(seed), numThreads);
        
        // Initialize map for saving the class labels attributes for merging with feature sets
        this.map_SplitType_InstanceClassLabels = new EnumMap<>(SplitType.class);
        
        // Initialize a Map to save the Set of correctly classified instances
        // Used for parameter optimization and accuracy calculation
        this.map_SplitClassifications = new EnumMap<>(SplitType.class);
        
        // Initialize FeatureSetTypes
        this.FEATURE_SET_TYPES = fsTypes;
        
        // Initialize Table object for saving the Feature Set Instances objects for each 
        // SplitType, FeatureSetType, Alpha and Window
        this.map_SplitType_FeatureSetType_Tbl_Instances = Tables.synchronizedTable(HashBasedTable.create());      // Synchronized ??
        
        // Initialize a Table object to save the training/testing times for each SplitType and FeatureSetType
        this.tbl_TrainTestTime = HashBasedTable.create();
        
        for (SplitType splitType : SplitType.values()) {
            // Create a single Attribute Instances object containing ClassLabels for
            // the corresponding split instances. Used for Feature Set creation
            this.map_SplitType_InstanceClassLabels.put(splitType, this.createOneColumnInstancesObjectWithClassLabelAttribute(splitType));
            
            this.map_SplitClassifications.put(splitType, Sets.newTreeSet());
            
            this.FEATURE_SET_TYPES.stream().forEach( (featureSetType) -> {
                this.map_SplitType_FeatureSetType_Tbl_Instances.put(splitType, featureSetType,
                                                                    Tables.synchronizedTable(HashBasedTable.create()));
                this.tbl_TrainTestTime.put(splitType, featureSetType, new HashMap<>());
            });
        }
        
        // Initialize temp directory path
        this.TEMP_DIR = Paths.get(tempDir, this.getDatasetName()).toString();
        
        // Initialize the AlphaWindowPair list with all initial pairs
        this.setAlphaWindowPairs(awPairs);
        
        this.seed = seed;
        
        // String mining executable name
        this.STRING_MINER = Paths.get(System.getProperty("user.dir"), stringMinerFileName).toString();
        
        // String miner parameters
        this.LOG_V = Integer.toString(log_v);
        this.BLOCK_SIZE = Integer.toString(blockSize);
        
        // Minimum frequency for positive class patterns
        this.MIN_POS_FREQ = Double.toString(min_pos_freq);
        
        // Maximum frequency for negative class patterns
        this.MAX_NEG_FREQ = Double.toString(max_neg_freq);
        
        // String mining process timeout parameters
        this.TIMEOUT_VAL = 30;
        this.TIMEOUT_UNIT = TimeUnit.MINUTES;
        
        // Initialize Table object for saving frequent pattern extraction times for each Alpha/Window pair
        this.tbl_FreqPattExtractionTimes = Tables.synchronizedTable(HashBasedTable.create());
        
        // Initialize a Map for each PatternIndependenceTest value to save a Table of each Alpha, Window pair's (filtered) FreqPatternStrings
        this.map_PattIndTest_Tbl_FreqPatts = new HashMap<>();   // Not using ConcurrentHashMap since
        // the HashMap is filled without any concurrency but the Tables are synchronized since writing
        // to Tables can be concurrently done
        
        // Initialize a Map for each PatternIndependenceTest value to save the Pattern filtering time
        this.map_PattIndTest_PatternFilterringTime = new EnumMap<>(PatternIndependenceTest.class);
        
        pattIndTests.stream().forEach( (indTest) -> {
            this.map_PattIndTest_Tbl_FreqPatts.put(indTest, Tables.synchronizedTable(HashBasedTable.create()));
            this.map_PattIndTest_PatternFilterringTime.put(indTest, 0L);
        });
        
        // Initialize Table object for saving feature set creation times
        this.tbl_FeatureSetCreationTimes = Tables.synchronizedTable(HashBasedTable.create());
        
        // Initialize a Table object to hold the TRANSFORMED feature set Instances objects for each 
        // SplitType and FeatureSetType
        this.tbl_FeatureSets = HashBasedTable.create();
        
        this.tbl_ModelAccuracyVals = HashBasedTable.create();
        
        // Provide the shuffled dataset splits to the MiSTiCl object, which creates the required symbolic versions and extracts the frequent patterns
        this.performPreprocessing();
    }
    
    /**
     *
     * @throws java.io.IOException
     */
    @Override
    protected final void performPreprocessing() throws IOException {
        ForkJoinPool threads = new ForkJoinPool(this.getNumExecutionThreads());
        
        try {
            Files.createDirectories(Paths.get(this.TEMP_DIR));
        } catch (IOException e) {
            throw new IOException("Error creating temporary files or directory", e);
        }
            
        try {
            // Transform the train/test splits into SAX/Symbolic versions
            threads.submit(() -> this.getAlphaWindowPairs()
                                     .parallelStream()
                                     .forEach(throwingConsumerWrapper(pair -> this.transformData(pair))))
                    .get();
            
            // extract the frequent patterns (write to files, extract patterns, delete files, add to ordered maps)
            threads.submit(() -> this.getAlphaWindowPairs()
                                     .parallelStream()
                                     .forEach(throwingConsumerWrapper(pair -> this.findFrequentPatterns(pair))))
                    .get();
            
            // delete the temp directory
            Paths.get(TEMP_DIR).toFile().delete();
            
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("The dataset transformation/frequent pattern extraction process was interrupted/halted.", e);
        } catch (RuntimeException e) {
            throw e;
        } finally {
            threads.shutdown();
        }
    }
    
    /**
     * Creates a single column Instances object for the provided SplitType
     * which is subsequently used when the feature sets are combined/merged
     * 
     * @param splitType
     * @return 
     */
    private Instances createOneColumnInstancesObjectWithClassLabelAttribute(SplitType splitType) {
        Attribute targetAttribute = this.getSplitTypeToRVInstsMap().get(splitType).classAttribute();
        
        ArrayList<Attribute> listOfAttributes = new ArrayList<>();
        listOfAttributes.add(targetAttribute);
        
        Instances template = new Instances("", listOfAttributes, 0);
        template.setClassIndex(0);
        
        Instances srcSplit = this.getSplitTypeToRVInstsMap().get(splitType);
        
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
    
    /**
     *
     * @param pair
     * @throws java.io.IOException
     */
    protected void findFrequentPatterns(AlphaWindowPair pair) throws IOException {
        String fileNamePrefix = this.getCurrentFilePrefix(pair);
        
        int alpha = pair.getNumAlphabets();
        int window = pair.getWindowSize();
        
        // Extract the frequent patterns for the current Alpha, Window pair symbolic form
        ProcessBuilder processBuilder = new ProcessBuilder();
        Process process;
        
        Map<Double, Map<Double, Map<Double, List<String>>>> map_ClassValues_UnfilteredFreqStrings = new HashMap<>();

        // Write the symbolized training instances for current Alpha, Window pair to files
        // REQUIRED for string miner
        this.writeSymbolicTrainingSplitToFiles(fileNamePrefix, pair);

        long timer = System.currentTimeMillis();
        
        for (Double classValue : this.getClsValsToInstCountsMap().keySet()) {
            processBuilder.command(this.STRING_MINER, LOG_V, BLOCK_SIZE, "2",
                                   Paths.get(TEMP_DIR, fileNamePrefix + classValue.intValue() + "vsALL.fasta")
                                        .toString(),
                                   Paths.get(TEMP_DIR, fileNamePrefix + "ALLvs" + classValue.intValue() + ".fasta")
                                        .toString(),
                                   this.MIN_POS_FREQ,
                                   this.MAX_NEG_FREQ,
                                   "F");
            try {
                process = processBuilder.start();
                Threaded_IOHandler ioHandler = new Threaded_IOHandler(process.getInputStream());
                ioHandler.start();
                process.waitFor(this.TIMEOUT_VAL, this.TIMEOUT_UNIT);
                ioHandler.join();

                Map<Double, Map<Double, List<String>>> map_CurrClass_FreqStrings = ioHandler.getParsedFrequentStrings();
                
                map_ClassValues_UnfilteredFreqStrings.put(classValue, map_CurrClass_FreqStrings);
            } catch (IOException e) {
                throw new IOException("Error occurred while initiating string mining task.", e);
            } catch (InterruptedException e) {
                throw new RuntimeException("The string mining process did not complete within the set limit of " +
                                           this.TIMEOUT_VAL + this.TIMEOUT_UNIT.toString(), e);
            }
        }
        
        timer = System.currentTimeMillis() - timer;
        this.tbl_FreqPattExtractionTimes.put(alpha, window, timer);
        
        // Delete the files created earlier, these files will not be needed any longer
        Arrays.stream(Paths.get(TEMP_DIR).toFile().listFiles((d, name) -> name.startsWith(fileNamePrefix))).forEach(File::delete);
        
        // Sort and filter the extracted patterns based on independence tests
        int numPositiveInsts, numNegativeInsts, totalInstances;
        totalInstances = this.getClsValsToInstCountsMap()
                             .values()
                             .stream()
                             .map(i -> i)
                             .reduce(0, Integer::sum);
        double srcEntropy, independenceVal;
        ChiSquareTest chiSqTest = new ChiSquareTest();
        long[][] counts = new long[2][2];
        List<String> list;
        
        Map<Double, Map<Double, List<String>>> map_ClassValues_FilteredFreqStrings;
        
        for (PatternIndependenceTest indTest : this.map_PattIndTest_Tbl_FreqPatts.keySet()) {
            
            map_ClassValues_FilteredFreqStrings = new HashMap<>();
            
            for (Double classValue : map_ClassValues_UnfilteredFreqStrings.keySet()) {
                map_ClassValues_FilteredFreqStrings.put(classValue, new TreeMap<>(Collections.reverseOrder()));
            }
            
            this.map_PattIndTest_Tbl_FreqPatts.get(indTest).put(alpha, window, map_ClassValues_FilteredFreqStrings);
        }
        
        Map<Double, List<String>> map_IndVal_FilteredFreqStrings;
        
        for (Double classValue : map_ClassValues_UnfilteredFreqStrings.keySet()) {
            numPositiveInsts = this.getClsValsToInstCountsMap().get(classValue);
            numNegativeInsts = totalInstances - numPositiveInsts;
            srcEntropy = this.getEntropy(numPositiveInsts, 1.0, numNegativeInsts, 1.0);
            
            for (Double positiveFreq : map_ClassValues_UnfilteredFreqStrings.get(classValue).keySet()) {

                for (Double negativeFreq : map_ClassValues_UnfilteredFreqStrings.get(classValue).get(positiveFreq).keySet()) {
                    
                    for (PatternIndependenceTest indTest : this.map_PattIndTest_Tbl_FreqPatts.keySet()) {
                        timer = System.currentTimeMillis();
                        switch (indTest) {
                            case CS:
                                // True Positive
                                counts[0][0] = Math.round(positiveFreq * numPositiveInsts);
                                // False Positive
                                counts[1][0] = Math.round(negativeFreq * numNegativeInsts);

                                // False Negative
                                counts[0][1] = numPositiveInsts - counts[0][0];
                                // True Negative
                                counts[1][1] = numNegativeInsts - counts[1][0];
                                // chiSqTest.chiSquareTest(counts, 0.001)
                                independenceVal = chiSqTest.chiSquare(counts) / totalInstances;  // /totalInstances ?? normalizes the independence value [0, 1]
                                independenceVal = Math.round(independenceVal*1e3)/1e3;    // Convert to an integer then divide again to reduce floating point accuracy
                                break;
                                
                            case IG:
                                independenceVal = getInformationGain(totalInstances,
                                                                     srcEntropy,
                                                                     numPositiveInsts,
                                                                     positiveFreq,
                                                                     numNegativeInsts,
                                                                     negativeFreq);
                                independenceVal = Math.round(independenceVal*1e3)/1e3;    // Convert to an integer then divide again to reduce floating point accuracy
                                break;
                                
                            case NO:
                            case NF:
                            default:
                                independenceVal = 1.0;
                                break;
                        }
                        
                        map_ClassValues_FilteredFreqStrings = this.map_PattIndTest_Tbl_FreqPatts.get(indTest)
                                                                                                .get(alpha, window);
                        map_IndVal_FilteredFreqStrings = map_ClassValues_FilteredFreqStrings.get(classValue);
                        
                        for (String freqString : map_ClassValues_UnfilteredFreqStrings.get(classValue).get(positiveFreq).get(negativeFreq)) {
                            
                            list = map_IndVal_FilteredFreqStrings.computeIfAbsent(independenceVal, l -> new ArrayList<>());
                            
                            if (indTest == PatternIndependenceTest.NO) {
                                list.add(freqString);
                            } else {
                                alreadyAdded: {
                                    for (String added : list) {
                                        if (freqString.contains(added)) {
                                            break alreadyAdded;
                                        }
                                    }
                                    list.add(freqString);
                                }
                            }
                        }
                        timer = System.currentTimeMillis() - timer;
                        this.map_PattIndTest_PatternFilterringTime.put(indTest, timer + this.map_PattIndTest_PatternFilterringTime.get(indTest));
                    }
                }
            }
        }
    }
    
    private String getCurrentFilePrefix(AlphaWindowPair pair) {
        StringBuilder strb = new StringBuilder();
        return strb.append("Alpha_")
                   .append(pair.getNumAlphabets())
                   .append("_Step_")
                   .append(pair.getWindowSize())
                   .append("_")
                   .toString();
    }
    
    private void writeSymbolicTrainingSplitToFiles(String fileNamePrefix, AlphaWindowPair pair) throws IOException {
        long timer = System.currentTimeMillis();
        Map<Double, BufferedWriter[]> map_ClassValues_BufferedWriters = new HashMap<>();

        for (Double classValue : this.getClsValsToInstCountsMap().keySet()) {
            this.createBufferedWriters(fileNamePrefix,
                                       classValue,
                                       map_ClassValues_BufferedWriters);
        }

        Instances saxedInstances = this.getSplitTypeToSymInstsMap().get(SplitType.TRAIN)
                                                                   .get(pair.getNumAlphabets(),
                                                                        pair.getWindowSize());

        for (int instanceIndex = 0; instanceIndex < saxedInstances.numInstances(); instanceIndex++) {
            this.writeData(instanceIndex, saxedInstances.get(instanceIndex), map_ClassValues_BufferedWriters);
        }

        this.closeBufferedWriters(map_ClassValues_BufferedWriters);
        
        Table<Integer, Integer, Long> trainingSplitTransformationTimes
                                      = this.getSplitTypeToTransformationTimesTable().get(SplitType.TRAIN);
        
        timer = (System.currentTimeMillis() - timer) +
                trainingSplitTransformationTimes.get(pair.getNumAlphabets(), 
                                                     pair.getWindowSize());
        
        trainingSplitTransformationTimes.put(pair.getNumAlphabets(),
                                             pair.getWindowSize(),
                                             timer);
    }

    private void createBufferedWriters(String fileNamePrefix, Double classValue, Map<Double, BufferedWriter[]> map_ClassValues_BufferedWriters) throws IOException {
        BufferedWriter[] bwArray = new BufferedWriter[2];
        try {
            bwArray[0]  = Files.newBufferedWriter(Paths.get(TEMP_DIR,
                                                            fileNamePrefix
                                                            + classValue.intValue()
                                                            + "vsALL.fasta"),
                                                  StandardOpenOption.CREATE,
                                                  StandardOpenOption.TRUNCATE_EXISTING);
            bwArray[1]  = Files.newBufferedWriter(Paths.get(TEMP_DIR,
                                                            fileNamePrefix
                                                            + "ALLvs"
                                                            + classValue.intValue()
                                                            + ".fasta"),
                                                  StandardOpenOption.CREATE,
                                                  StandardOpenOption.TRUNCATE_EXISTING);
            
            map_ClassValues_BufferedWriters.put(classValue, bwArray);
        } catch (IOException e) {
            throw new IOException("Error occurred while creating buffered writers for writing binary class dataset splits.", e);
        }
    }
    
    private void writeData(int instInd, Instance symbolicFormInst, Map<Double, BufferedWriter[]> map_ClassValues_BufferedWriters) throws IOException {
        BufferedWriter[] bwArray;
        String currInstClassLabel = symbolicFormInst.stringValue(symbolicFormInst.classIndex());
        String symbolicForm = symbolicFormInst.stringValue(0);
        try {
            for (Double destFileClassValue : map_ClassValues_BufferedWriters.keySet()) {
                bwArray = map_ClassValues_BufferedWriters.get(destFileClassValue);
                if (this.nearlyEqual(symbolicFormInst.classValue(), destFileClassValue, 1e-3)) {
                    bwArray[0].write("> label:" + currInstClassLabel + " - id:" + instInd +
                                     "\n" + symbolicForm + "\n");
                } else {
                    bwArray[1].write("> label:" + currInstClassLabel + " - id:" + instInd +
                                     "\n" + symbolicForm + "\n");
                }
            }
        } catch (IOException e) {
            throw new IOException("Error occurred while writing to the binary class dataset splits", e);
        }
    }
    
    private void closeBufferedWriters(Map<Double, BufferedWriter[]> map_ClassValues_BufferedWriters) throws IOException {
        map_ClassValues_BufferedWriters.entrySet()
                                       .stream()
                                       .forEach( throwingConsumerWrapper( classValueToBWsArray ->
                                                                                    {
                                                                                        for (BufferedWriter bw : classValueToBWsArray.getValue()) {
                                                                                                bw.close();
                                                                                         }
                                                                                    })
                                               );
    }
    
    private class Threaded_IOHandler extends Thread {
        private final InputStream inputStream;
        private Map<Double, Map<Double, List<String>>> frequentStringsMap;
        Comparator<String> strComp = (String s1, String s2) -> {
            // // Alphabetic order
            // return s1.compareTo(s2);
            // // Reverse alphabetic order
            // return s2.compareTo(s1);
            
            // Length based
            if(s1.length() < s2.length()) {
                return -1;
            } else if (s1.length() > s2.length()) {
                return 1;
            } else {
                return 0;
            }
        };
        
        public Threaded_IOHandler(InputStream inputStream) {
            this.inputStream = inputStream;
        }
        
        @Override
        public void run() {
            String line;
            String[] freqPattArray;
            this.frequentStringsMap = new TreeMap<>(Collections.reverseOrder());
            Map<Double, List<String>> negMap;
            List<String> strList;
            double posFreq, negFreq;
            try (BufferedReader br = new BufferedReader(new InputStreamReader(inputStream))) {
                while ((line = br.readLine()) != null) {
                    freqPattArray = line.split(",");
                    posFreq = Double.parseDouble(freqPattArray[0].trim());
                    negFreq = Double.parseDouble(freqPattArray[1].trim());
                    
                    negMap = frequentStringsMap.computeIfAbsent(posFreq, m -> new TreeMap<>());
                    
                    strList = negMap.computeIfAbsent(negFreq, l -> new ArrayList<>());
                    
                    strList.add(freqPattArray[freqPattArray.length - 1].trim());
                }
            } catch (Exception e) {
                System.err.println(e);
            }
        }

        public Map<Double, Map<Double, List<String>>> getParsedFrequentStrings() {
            this.frequentStringsMap
                .keySet()
                .stream()
                .forEach( posFreq -> this.frequentStringsMap.get(posFreq)
                                         .keySet()
                                         .stream()
                                         .forEach( negFreq -> Collections.sort(this.frequentStringsMap.get(posFreq)
                                                                                                      .get(negFreq),
                                                                               strComp)
                                                 )
                        );
            
//            this.frequentStringsMap
//                .keySet()
//                .stream()
//                .limit(2)
//                .forEach( posFreq -> this.frequentStringsMap.get(posFreq)
//                                         .keySet()
//                                         .stream()
//                                         .limit(2)
//                                         .forEach( negFreq -> System.out.println(posFreq + " " + negFreq + " " + this.frequentStringsMap.get(posFreq).get(negFreq))
//                                                 )
//                        );
            return this.frequentStringsMap;
        }
    }
    
    /**
     *
     * @param posInsts
     * @param posFreq
     * @param negInsts
     * @param negFreq
     * @return
     */
    protected double getEntropy(int posInsts, double posFreq, int negInsts, double negFreq) {
        double frac = 1.0 / (posFreq*posInsts + negFreq*negInsts);
        double entropy = 0.0;
        double temp;
        temp = posFreq*posInsts*frac;
        entropy -= (temp > 0.0) ? temp*Math.log(temp)/LOG2 : 0.0;
        temp = negFreq*negInsts*frac;
        entropy -= (temp > 0.0) ? temp*Math.log(temp)/LOG2 : 0.0;
        
        return entropy;
    }
    
    /**
     *
     * @param totalInstances
     * @param srcEntropy
     * @param posInsts
     * @param posFreq
     * @param negInsts
     * @param negFreq
     * @return
     */
    protected double getInformationGain(int totalInstances, double srcEntropy, int posInsts, double posFreq, int negInsts, double negFreq) {
        double infoGain = srcEntropy;
        double leftEntropy = this.getEntropy(posInsts, posFreq, negInsts, negFreq);
        double rightEntropy = this.getEntropy(posInsts, 1-posFreq, negInsts, 1-negFreq);
        double frac;
        frac = (posFreq*posInsts+negFreq*negInsts)/totalInstances;
        infoGain -= frac*leftEntropy;
        frac = ((1-posFreq)*posInsts+(1-negFreq)*negInsts)/totalInstances;
        infoGain -= frac * rightEntropy;
        return infoGain;
    }
    
    /**
     *
     * @param independenceTest
     * @param approxMethod
     * @param patternsPerClass
     */
    public void createFeatureSets(PatternApproximationMethod approxMethod, PatternIndependenceTest independenceTest, int patternsPerClass) {
        this.independenceTest = independenceTest;
        
        this.approximationMethod = approxMethod;
        
        this.maxPatternsPerClass = patternsPerClass;
        
        this.resetTimers();
        
        ForkJoinPool threads = new ForkJoinPool(this.getNumExecutionThreads());
        
        try {
            threads.submit(() -> this.getAlphaWindowPairs()
                                     .parallelStream()
                                     .forEach( throwingConsumerWrapper( pair -> createInstances(pair) ) )
                          )
                   .get();
        } catch (InterruptedException|ExecutionException e) {
            System.err.println("Interrupted YADA YADA " + e);
        } finally {
            threads.shutdown();
        }
    }
    
    private void resetTimers() {
        for (SplitType splitType : SplitType.values()) {
            this.FEATURE_SET_TYPES.stream().forEach( (f) -> {
                this.tbl_FeatureSetCreationTimes.put(splitType, f, 0L);
            });
        }
    }
    
    /**
     *
     * @param pair
     * @throws java.lang.Exception
     */
    protected void createInstances(AlphaWindowPair pair) throws Exception {
        ArrayList<Attribute> attributes = this.getTopFrequentPatternAttributes(pair);
        
        int alpha = pair.getNumAlphabets();
        int window = pair.getWindowSize();
        Instances featureDataTemplate = new Instances("", attributes, 0);
        
        for (SplitType splitType : SplitType.values()) {
            this.FEATURE_SET_TYPES.stream().forEach( (fsType) ->
            {
                Instances blankFeatureSet = new Instances(featureDataTemplate);
                
                for (int instCount = 0 ; instCount < this.getSplitTypeToRVInstsMap().get(splitType).numInstances(); instCount++) {
                    Instance inst = new DenseInstance(blankFeatureSet.numAttributes());
                    inst.setDataset(blankFeatureSet);
                    blankFeatureSet.add(inst);
                }
                this.map_SplitType_FeatureSetType_Tbl_Instances.get(splitType, fsType).put(alpha, window, blankFeatureSet);
            });
        }
        
        double[] breakpoints = this.getBreakpoints(pair);
        Double[] shapelet;
        Instances symbolizedSet;
        
        String freqString, attributeName;
        double featureValue;
        
        Table<SplitType, FeatureSetType, Long> timeTable = HashBasedTable.create();
        
        for (SplitType splitType: SplitType.values()) {
            this.FEATURE_SET_TYPES.stream().forEach( (fsType) -> timeTable.put(splitType, fsType, 0L) );
        }
        
        long timer;
        
//        // 2018-01-18
//        System.out.println("Time series length: " + this.getRVInstLength() + ", Alpha: " + pair.getNumAlphabets() + ", Window: " + pair.getWindowSize());
        
        for (int attInd = 0; attInd < attributes.size(); attInd++) {
            attributeName = attributes.get(attInd).name();
            freqString = attributeName.substring(attributeName.lastIndexOf("_")+1); //, 20
            
            symbolizedSet = this.getSplitTypeToSymInstsMap().get(SplitType.TRAIN).get(alpha, window);
            shapelet = this.getNearestBestShapelet(pair, freqString, breakpoints, symbolizedSet);
            
            for (SplitType splitType : SplitType.values()) {
                symbolizedSet = this.getSplitTypeToSymInstsMap().get(splitType).get(alpha, window);
                for (FeatureSetType fsType : this.FEATURE_SET_TYPES) {
                    timer = System.currentTimeMillis();
                    
                    Instances currFeatureSet = this.map_SplitType_FeatureSetType_Tbl_Instances.get(splitType, fsType).get(alpha, window);
                    
                    for (int instInd = 0; instInd < currFeatureSet.numInstances(); instInd++) {
                        
                        switch (fsType) {
                            case NUM:
                                featureValue = this.getMinimumDistance(this.getInstanceAsArray(instInd, splitType), shapelet);
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
            this.FEATURE_SET_TYPES.stream().forEach((fsType) ->
            {
                this.tbl_FeatureSetCreationTimes.put(splitType, fsType, timeTable.get(splitType, fsType) +
                        this.tbl_FeatureSetCreationTimes.get(splitType, fsType));
            });
        }
    }
    
    /**
     *
     * @param pair
     * @return
     */
    protected ArrayList<Attribute> getTopFrequentPatternAttributes(AlphaWindowPair pair) {
        int alpha = pair.getNumAlphabets();
        int window = pair.getWindowSize();
        
        String currAttributeName;
        Attribute currAttribute;
        ArrayList<Attribute> topAttributes = new ArrayList<>();
        
        Map<Double, Map<Double, List<String>>> map_ClassValues_FilteredFreqStrings = this.map_PattIndTest_Tbl_FreqPatts.get(this.independenceTest)
                                                                                                                       .get(alpha, window);
        int currClassAttributeCount;
        Set<String> currParamFreqPatterns = new HashSet<>();
        
        for (Double classValue : map_ClassValues_FilteredFreqStrings.keySet()) {
            
            currClassAttributeCount = 0;
            
            LabelLoop:
            for (Double indTestValue : map_ClassValues_FilteredFreqStrings.get(classValue).keySet()) {
                
                for (String freqString : map_ClassValues_FilteredFreqStrings.get(classValue).get(indTestValue)) {
                    
                    currAttributeName = "Cls_" + classValue.intValue() + "_A_" + alpha + "_W_" + window + "_" + freqString;
                    
                    if (currParamFreqPatterns.add(currAttributeName)) {
                        
                        currAttribute = new Attribute(currAttributeName);
                        topAttributes.add(currAttribute);
                        currClassAttributeCount++;
                        
                        if (currClassAttributeCount >= this.maxPatternsPerClass) {
                            break LabelLoop;
                        }
                    }
                }
            }
        }
        return topAttributes;
    }
    
    /**
     *
     * @param pair
     * @return
     * @throws java.lang.Exception
     */
    protected double[] getBreakpoints(AlphaWindowPair pair) throws Exception {
        double[] breakpoints = null;
        try {
            // Create SAX object
            SAX saxer = new SAX();
            saxer.useRealValuedAttributes(false);
            saxer.setAlphabetSize(pair.getNumAlphabets());
            saxer.setNumIntervals(this.getRVInstLength() / pair.getWindowSize());
            breakpoints = saxer.generateBreakpoints(pair.getNumAlphabets());
        } catch (Exception e) {
            throw new Exception("Error getting breakpoints.", e);
        }
        return breakpoints;
    }
    
    /**
     *
     * @param pair
     * @param symForm
     * @param breakpoints
     * @param symTrainSet
     * @return
     */
    protected Double[] getNearestBestShapelet(AlphaWindowPair pair, String symForm, double[] breakpoints, Instances symTrainSet) {
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
        switch (this.approximationMethod) {
            case S:     // SAFE and just a single time point shorter than the original
                for (int i = 0; i < symForm.length(); i++) {
                    for (int j = 0; j < pair.getWindowSize(); j++) {
                        interpolatedPattern.add(realValuedPattern[i]);
                    }
                }
                break;
            case I:
                double stepSize = ( this.getRVInstLength()/pair.getWindowSize()) / (double)this.getRVInstLength();
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
                inst = this.getInstanceAsArray(i, SplitType.TRAIN);
                // To reduce the searching cost even further
                // Instead of searching in the entire length of the real-valued instance,
                // could use the approximate location at the start of the match
                // Would require to change the if condition to (symTrainSet[i].indexOf(symForm) > 0)
                // instead of contains(...)
                // Then the start of the search in the following loop would be from j = indexOf(...)*pair.getWindowSize()
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
    
    /**
     *
     * @param ind
     * @param splitType
     * @return
     */
    protected double[] getInstanceAsArray(int ind, SplitType splitType) {
        Instance inst = this.getSplitTypeToRVInstsMap().get(splitType).get(ind);
        double[] temp = new double[inst.numAttributes()-1];
        for (int attIndex = 0; attIndex < inst.numAttributes()-1; attIndex++) {
            temp[attIndex] = inst.value(attIndex);
        }
//        return Arrays.copyOf(inst.toDoubleArray(), inst.numAttributes()-1);
        return temp;
    }
    
    /**
     *
     * @param inst
     * @param shapelet
     * @return
     */
    protected double getMinimumDistance(double[] inst, Double[] shapelet) {
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
    
    /**
     *
     * @param featureSetType
     */
    public void combineFeatureSets(FeatureSetType featureSetType) {
        int alpha, window;
        Instances currSplitFeatureSet;
        for (SplitType splitType : SplitType.values()) {
            currSplitFeatureSet = null;
            for (AlphaWindowPair pair : this.getAlphaWindowPairs()) {
                alpha = pair.getNumAlphabets();
                window = pair.getWindowSize();
                if (currSplitFeatureSet == null) {
                    // currSplitFeatureSet = new Instances(this.map_SplitType_FeatureSetType_Tbl_Instances.get(splitType, featureSetType).get(alpha, window));
                    // SAFE: No need to create clone as above, since merging will create new instance objects
                    currSplitFeatureSet = this.map_SplitType_FeatureSetType_Tbl_Instances.get(splitType, featureSetType).get(alpha, window);
                } else {
                    Instances currPairFeatureSet = this.map_SplitType_FeatureSetType_Tbl_Instances.get(splitType, featureSetType).get(alpha, window);
                    currSplitFeatureSet = Instances.mergeInstances(currSplitFeatureSet, currPairFeatureSet);
                }
            }
            
            // To create SVM format data correctly add a dummy all-ones attribute before the folowing statement
            Attribute dummy = new Attribute("dummy");
            ArrayList<Attribute> listOfAttributes = new ArrayList<>();
            listOfAttributes.add(dummy);

            Instances dummyAttColumn = new Instances("", listOfAttributes, this.map_SplitType_InstanceClassLabels.get(splitType).size());
            Instance newDestInst = new DenseInstance(dummyAttColumn.numAttributes());
            newDestInst.setDataset(dummyAttColumn);
            newDestInst.setValue(0, 1);
            
            this.map_SplitType_InstanceClassLabels.get(splitType).forEach( i -> dummyAttColumn.add(newDestInst) );
            
            if (currSplitFeatureSet == null) {
                currSplitFeatureSet = dummyAttColumn;    // SAFE since merging will create new instance objects
            } else {
                currSplitFeatureSet = Instances.mergeInstances(currSplitFeatureSet, dummyAttColumn);
            }
            
            currSplitFeatureSet = Instances.mergeInstances(currSplitFeatureSet, this.map_SplitType_InstanceClassLabels.get(splitType));
            currSplitFeatureSet.setClassIndex(currSplitFeatureSet.numAttributes()-1);
            currSplitFeatureSet.setRelationName(this.getDatasetName() + "_" + splitType.name() + "_" + featureSetType.name());
            this.tbl_FeatureSets.put(splitType, featureSetType, currSplitFeatureSet);
        }
    }
    
    /**
     *
     * @param classifierType
     * @param featureSetType
     * @throws ml.dmlc.xgboost4j.java.XGBoostError
     */
    public void trainClassifier(ClassifierType classifierType, FeatureSetType featureSetType) throws XGBoostError, Exception {
        this.trainClassifier(classifierType, featureSetType, -1);
    }
    
    /**
     *
     * @param classifierType
     * @param featureSetType
     * @param numTrees
     * @throws java.io.IOException
     * @throws ml.dmlc.xgboost4j.java.XGBoostError
     */
    public void trainClassifier(ClassifierType classifierType, FeatureSetType featureSetType, int numTrees) throws IOException, XGBoostError, Exception {
        for (SplitType splitType : SplitType.values()) {
            this.map_SplitClassifications.get(splitType).clear();
            this.tbl_ModelAccuracyVals.put(splitType, featureSetType, 0.0);
            this.tbl_TrainTestTime.get(splitType, featureSetType).put(classifierType, 0L);
        }
        
        if (this.tbl_FeatureSets.get(SplitType.TRAIN, featureSetType).numAttributes() <= 2) {
            return;
        }
        
        int numTreesToUse = (numTrees == -1) ? NUM_TREES : numTrees;
        
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
        trainSplit = this.tbl_FeatureSets.get(SplitType.TRAIN, featureSetType);
        testSplit = this.tbl_FeatureSets.get(SplitType.TEST, featureSetType);
        
        AbstractClassifier classifier;
        switch(classifierType) {
            case AB:
                AdaBoostM1 adb = new AdaBoostM1();
                AbstractClassifier base = new RandomForest();
                ((RandomForest)base).setNumIterations(10);
                ((RandomForest)base).setNumExecutionSlots(this.getNumExecutionThreads());
                adb.setClassifier(base);
                adb.setNumIterations(numTrees);  //(trainSplit.numAttributes()-1)/(int)Math.log(trainSplit.numInstances())
                ((RandomForest)base).setSeed(this.seed);
                adb.setSeed(this.seed);
                classifier = adb;
                break;
            case RF:
                RandomForest rf = new RandomForest();
                rf.setNumIterations(numTrees);
                rf.setNumExecutionSlots(this.getNumExecutionThreads());
                rf.setSeed(this.seed);
                classifier = rf;
                break;
            case ET:
            default:
                RandomCommittee committee = new RandomCommittee();
                committee.setNumIterations(numTrees);
                committee.setNumExecutionSlots(this.getNumExecutionThreads());
                ExtraTree ert = new ExtraTree();
                committee.setSeed(this.seed);
                ert.setSeed(this.seed);
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
            
            trainAcc = classifySplit(classifier, trainSplit, SplitType.TRAIN);
            
            this.tbl_ModelAccuracyVals.put(SplitType.TRAIN, featureSetType, trainAcc);
            this.tbl_TrainTestTime.get(SplitType.TRAIN, featureSetType).put(classifierType, trainingTime);
            
            testingTime = System.currentTimeMillis();
            testAcc = classifySplit(classifier, testSplit, SplitType.TEST);
            testingTime = System.currentTimeMillis() - testingTime;
            
            this.tbl_ModelAccuracyVals.put(SplitType.TEST, featureSetType, testAcc);
            this.tbl_TrainTestTime.get(SplitType.TEST, featureSetType).put(classifierType, testingTime);
        } catch (Exception e) {
            throw e;
        }
    }
    
    private double classifySplit(AbstractClassifier classifier, Instances fbSplit, SplitType splitType) throws Exception {
        Set<Integer> classifications = this.map_SplitClassifications.get(splitType);
        int classInd = fbSplit.numAttributes()-1;
        try {
            double result;
            for (int ind = 0; ind < fbSplit.numInstances(); ind++) {
                result = classifier.classifyInstance(fbSplit.get(ind));
                if (this.nearlyEqual(result, fbSplit.get(ind).value(classInd), 1e-6)) {
                    classifications.add(ind);
                }
            }
        } catch (Exception e) {
            throw e;
        }
        return 100.0*classifications.size()/fbSplit.numInstances();
    }
    
    private void trainXGBoostClassifier(ClassifierType classifierType, FeatureSetType featureSetType, int numTrees) throws IOException, XGBoostError {
        Instances trainSplit, testSplit;
        trainSplit = this.tbl_FeatureSets.get(SplitType.TRAIN, featureSetType);
        testSplit = this.tbl_FeatureSets.get(SplitType.TEST, featureSetType);
        
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
        params.put("nthread", this.getNumExecutionThreads());
        if (this.getClsValsToInstCountsMap().size() == 2) {
            params.put("objective", "binary:logistic");
            params.put("eval_metric", "error");
        } else {
            params.put("objective", "multi:softmax");
            params.put("num_class", Integer.toString(this.getClsValsToInstCountsMap().size()));
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
            
            this.tbl_ModelAccuracyVals.put(SplitType.TRAIN, featureSetType, trainAcc);
            this.tbl_TrainTestTime.get(SplitType.TRAIN, featureSetType).put(classifierType, trainingTime);
            
            testingTime = System.currentTimeMillis();
            testAcc = classifySplit(booster, testMat, SplitType.TEST);
            testingTime = System.currentTimeMillis() - testingTime;
            
            this.tbl_ModelAccuracyVals.put(SplitType.TEST, featureSetType, testAcc);
            this.tbl_TrainTestTime.get(SplitType.TEST, featureSetType).put(classifierType, testingTime);    
        } catch (XGBoostError e) {
            throw e;
        }
        
        Arrays.stream(Paths.get(TEMP_DIR).toFile().listFiles()).forEach(File::delete);
    }
    
    private double classifySplit(Booster booster, DMatrix fbSplit, SplitType splitType) throws XGBoostError {
        Set<Integer> classifications = this.map_SplitClassifications.get(splitType);
        try {
            float[][] predictions = booster.predict(fbSplit);
            float[] labels = fbSplit.getLabel();
            for (int i = 0; i < labels.length; i++) {
                if (this.getClsValsToInstCountsMap().size() == 2) {
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
            throw e;
        }
    }
    
    private boolean nearlyEqual(double a, double b, double epsilon) {
        final double absA = Math.abs(a);
        final double absB = Math.abs(b);
        final double diff = Math.abs(a - b);

        if (a == b) { // shortcut, handles infinities
            return true;
        } else if (a == 0 || b == 0 || diff < Double.MIN_NORMAL) {
            // a or b is zero or both are extremely close to it
            // relative error is less meaningful here
            return diff < (epsilon * Double.MIN_NORMAL);
        } else { // use relative error
            return diff / Math.min((absA + absB), Double.MAX_VALUE) < epsilon;
        }
    }
    
    /**
     *
     * @param splitType
     * @return
     */
    public Set<Integer> getSplitClassifications(SplitType splitType) {
        return this.map_SplitClassifications.get(splitType);
    }
    
    /**
     *
     * @param splitType
     * @param featureSetType
     * @return
     */
    public double getSplitAccuracy(SplitType splitType, FeatureSetType featureSetType) {
        return this.tbl_ModelAccuracyVals.get(splitType, featureSetType);
    }
    
    /**
     *
     * @param splitType
     * @param featureSetType
     * @param classifierType
     * @return
     */
    public long getTotalTime(SplitType splitType, FeatureSetType featureSetType, ClassifierType classifierType) {
        long total = this.getDataTransformationTime(splitType)
                     + this.getFeatureSetCreationTime(splitType, featureSetType)
                     + this.getModelTrainTestTime(splitType, featureSetType, classifierType);
        if (splitType.equals(SplitType.TRAIN)) {
            total += this.getPatternExtractionTime();
        }
        return total;
    }
    
    /**
     *
     * @param splitType
     * @return
     */
    public long getDataTransformationTime(SplitType splitType) {
        Table<Integer, Integer, Long> transformationTimesTable
                                      = this.getSplitTypeToTransformationTimesTable().get(splitType);
        
        long t = transformationTimesTable.cellSet().stream().mapToLong(c -> c.getValue()).sum();
        
        return t;
    }
    
    /**
     *
     * @param splitType
     * @param featureSetType
     * @return
     */
    public long getFeatureSetCreationTime(SplitType splitType, FeatureSetType featureSetType) {
        return this.tbl_FeatureSetCreationTimes.get(splitType, featureSetType);
    }
    
    /**
     *
     * @param splitType
     * @param featureSetType
     * @param classifierType
     * @return
     */
    public long getModelTrainTestTime(SplitType splitType, FeatureSetType featureSetType, ClassifierType classifierType) {
        return this.tbl_TrainTestTime.get(splitType, featureSetType).get(classifierType);
    }
    
    /**
     *
     * @return
     */
    public long getPatternExtractionTime() {
        Table<Integer, Integer, Long> patternExtractionTimes = this.tbl_FreqPattExtractionTimes;
        long t = patternExtractionTimes.cellSet().stream().mapToLong(c -> c.getValue()).sum();
        
        t += this.map_PattIndTest_PatternFilterringTime.get(this.independenceTest);
        return t;
    }
    
    /**
     *
     * @param psm
     * @param pit
     * @param f
     * @param ppc
     * @throws java.io.IOException
     */
    public void saveTransformedData(ParameterSelectionMethod psm, PatternIndependenceTest pit, FeatureSetType f, int ppc) throws IOException {
        try {
            Path path = Paths.get(System.getProperty("user.dir"), "instances", this.getDatasetName());
            Files.createDirectories(path);
            ArffSaver saver = new ArffSaver();
            for (SplitType splitType : SplitType.values()) {
                saver.setInstances(this.tbl_FeatureSets.get(splitType, f));

                saver.setFile(Paths.get(path.toString(),
                                        splitType.name() + "_" + 
                                        this.getDatasetName() + "_" + 
                                        psm.name() + "_" + 
                                        pit.name() + "_" + 
                                        f.name() + "_" + ppc + ".arff").toFile());
                saver.writeBatch();
            }
        } catch (IOException e) {
            throw new IOException("Couldn't write transformed instances. ", e);
        }
    }
    
    /**
     *
     * @param splitType
     * @return
     */
    public int getSplitSize(SplitType splitType) {
        return this.getSplitTypeToRVInstsMap().get(splitType).numInstances();
    }
}
