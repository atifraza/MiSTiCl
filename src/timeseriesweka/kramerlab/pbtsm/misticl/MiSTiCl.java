package timeseriesweka.kramerlab.pbtsm.misticl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.collect.Tables;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import java.nio.file.Files;
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

import org.apache.commons.math3.stat.inference.ChiSquareTest;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import timeseriesweka.kramerlab.pbtsm.AlphaOmegaPair;
import timeseriesweka.kramerlab.pbtsm.BasePbTSM;
import timeseriesweka.kramerlab.pbtsm.ClassifierType;
import timeseriesweka.kramerlab.pbtsm.FeatureSetType;
import timeseriesweka.kramerlab.pbtsm.PatternApproximationMethod;
import timeseriesweka.kramerlab.pbtsm.PatternIndependenceTest;
import timeseriesweka.kramerlab.pbtsm.RealValuedDataset;
import timeseriesweka.kramerlab.pbtsm.SplitType;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;


/**
 *
 * @author atif
 */
public class MiSTiCl extends BasePbTSM {
    
    private static final Logger LOGGER = LoggerFactory.getLogger(MiSTiCl.class);
    
    /**
     * Math.log(2) Constant. Used to convert LOG_e(x) to LOG_2(x) for entropy calculation
     */
    private static final double LOG2 = Math.log(2);
    
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
     * {@link com.google.common.collect.Table} of Frequent pattern extraction
     * times.
     * The Table object is keyed with Alpha (rows) and Window (column) values.
     */
    private final Table<Integer, Integer, Long> TBL_FREQ_PATT_EXTRACT_TIMES;

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
    
    private int maxPatternsPerClass;
    
    public MiSTiCl(RealValuedDataset rvDataset, List<AlphaOmegaPair> awPairs, int numThreads,
                   int seed, String programDir, String stringMinerFileName, String log_v, String blockSize,
                   String min_pos_freq, String max_neg_freq, Set<PatternIndependenceTest> pattIndTests,
                   Set<FeatureSetType> fsTypes, String tempDir) throws RuntimeException, IOException {
        
        super(rvDataset, fsTypes, numThreads, seed);
        
        // Initialize temp directory path
        TEMP_DIR = Paths.get(tempDir, this.getDatasetName()).toString();
        
        // Initialize the AlphaOmegaPair list with all initial pairs
        this.setAlphaOmegaPairs(awPairs);
        
        // String mining executable name
        STRING_MINER = Paths.get(programDir, stringMinerFileName).toString();
        
        // String miner parameters
        LOG_V = log_v;
        BLOCK_SIZE = blockSize;
        
        // Minimum frequency for positive class patterns
        MIN_POS_FREQ = min_pos_freq;
        
        // Maximum frequency for negative class patterns
        MAX_NEG_FREQ = max_neg_freq;
        
        // String mining process timeout parameters
        TIMEOUT_VAL = 30;
        TIMEOUT_UNIT = TimeUnit.MINUTES;
        
        // Initialize Table object for saving frequent pattern extraction times for each Alpha/Window pair
        TBL_FREQ_PATT_EXTRACT_TIMES = Tables.synchronizedTable(HashBasedTable.create());
        
        // Initialize a Map for each PatternIndependenceTest value to save a Table of each Alpha, Window pair's (filtered) FreqPatternStrings
        map_PattIndTest_Tbl_FreqPatts = new HashMap<>();   // Not using ConcurrentHashMap since
        // the HashMap is filled without any concurrency but the Tables are synchronized since writing
        // to Tables can be concurrently done
        
        // Initialize a Map for each PatternIndependenceTest value to save the Pattern filtering time
        map_PattIndTest_PatternFilterringTime = new EnumMap<>(PatternIndependenceTest.class);
        
        pattIndTests.forEach( indTest ->
        {
            map_PattIndTest_Tbl_FreqPatts.put(indTest,
                                                   Tables.synchronizedTable(HashBasedTable.create()));
            map_PattIndTest_PatternFilterringTime.put(indTest, 0L);
        });
        
        // Provide the shuffled dataset splits to the MiSTiCl object, which creates the required symbolic versions and extracts the frequent patterns
        this.performPreprocessing();
    }
    
    public void createAllFeatureSets(PatternApproximationMethod approxMethod, PatternIndependenceTest independenceTest, int patternsPerClass) {
        maxPatternsPerClass = patternsPerClass;
        
        this.createAllFeatureSets(approxMethod, independenceTest);
    }
    
    @Override
    public long getTotalTime(SplitType splitType, FeatureSetType featureSetType, ClassifierType classifierType) {
        long total = super.getTotalTime(splitType, featureSetType, classifierType);
        if (splitType.equals(SplitType.TRAIN)) {
            total += this.getPatternExtractionTime();
        }
        return total;
    }
    
    public long getPatternExtractionTime() {
        Table<Integer, Integer, Long> patternExtractionTimes = TBL_FREQ_PATT_EXTRACT_TIMES;
        long t = patternExtractionTimes.cellSet().stream().mapToLong(c -> c.getValue()).sum();
        
        t += map_PattIndTest_PatternFilterringTime.get(independenceTest);
        return t;
    }
    
    @Override
    protected final void performPreprocessing() throws IOException {
        ForkJoinPool threads = new ForkJoinPool(this.getNumOfExecutionThreads());
        
        try {
            Files.createDirectories(Paths.get(TEMP_DIR));
        } catch (IOException e) {
            LOGGER.error("Error creating temporary files or directory.\n{}", e.getMessage());
            throw e;
        }
            
        try {
            // Transform the train/test splits into SAX/Symbolic versions
            threads.submit(() -> this.getAlphaOmegaPairs()
                                     .parallelStream()
                                     .forEach(throwingConsumerWrapper(pair -> this.transformData(pair))))
                    .get();
            
            // extract the frequent patterns (write to files, extract patterns, delete files, add to ordered maps)
            threads.submit(() -> this.getAlphaOmegaPairs()
                                     .parallelStream()
                                     .forEach(throwingConsumerWrapper(pair -> this.findFrequentPatterns(pair))))
                    .get();
            
            // delete the temp directory
            Paths.get(TEMP_DIR).toFile().delete();
            
        } catch (InterruptedException | ExecutionException e) {
            LOGGER.error("Dataset transformation or frequent pattern extraction process was interrupted.\n{}", e.getMessage());
            throw new RuntimeException();
//        } catch (RuntimeException e) {
//            throw e;
        } finally {
            threads.shutdown();
        }
    }
    
    @Override
    protected ArrayList<Attribute> getTopFrequentPatternAttributes(AlphaOmegaPair pair) {
        int alpha = pair.getAlpha();
        int omega = pair.getOmega();
        
        String currAttributeName;
        Attribute currAttribute;
        ArrayList<Attribute> topAttributes = new ArrayList<>();
        
        Map<Double, Map<Double, List<String>>> map_ClassValues_FilteredFreqStrings = map_PattIndTest_Tbl_FreqPatts.get(independenceTest)
                                                                                                                       .get(alpha, omega);
        int currClassAttributeCount;
        Set<String> currParamFreqPatterns = new HashSet<>();
        
        for (Double classValue : map_ClassValues_FilteredFreqStrings.keySet()) {
            
            currClassAttributeCount = 0;
            
            LabelLoop:
            for (Double indTestValue : map_ClassValues_FilteredFreqStrings.get(classValue).keySet()) {
                
                for (String freqString : map_ClassValues_FilteredFreqStrings.get(classValue).get(indTestValue)) {
                    
                    currAttributeName = "Cls_" + classValue.intValue() + "_A_" + alpha + "_W_" + omega + "_" + freqString;
                    
                    if (currParamFreqPatterns.add(currAttributeName)) {
                        
                        currAttribute = new Attribute(currAttributeName);
                        topAttributes.add(currAttribute);
                        currClassAttributeCount++;
                        
                        if (currClassAttributeCount >= maxPatternsPerClass) {  // should be ==
                            break LabelLoop;
                        }
                    }
                }
            }
        }
        return topAttributes;
    }
    
    private void findFrequentPatterns(AlphaOmegaPair pair) throws IOException {
        String fileNamePrefix = this.getCurrentFilePrefix(pair);
        
        int alpha = pair.getAlpha();
        int omega = pair.getOmega();
        
        // Extract the frequent patterns for the current Alpha, Window pair symbolic form
        ProcessBuilder processBuilder = new ProcessBuilder();
        Process process;
        
        Map<Double, Map<Double, Map<Double, List<String>>>> map_ClassValues_UnfilteredFreqStrings = new HashMap<>();

        // Write the symbolized training instances for current Alpha, Window pair to files
        // REQUIRED for string miner
        this.writeSymbolicTrainingSplitToFiles(fileNamePrefix, pair);

        long timer = System.currentTimeMillis();
        
        for (Double classValue : this.getClassLabelsToInstanceCounts().keySet()) {
            processBuilder.command(STRING_MINER, LOG_V, BLOCK_SIZE, "2",
                                   Paths.get(TEMP_DIR, fileNamePrefix + classValue + "vsALL.fasta")
                                        .toString(),
                                   Paths.get(TEMP_DIR, fileNamePrefix + "ALLvs" + classValue + ".fasta")
                                        .toString(),
                                   MIN_POS_FREQ,
                                   MAX_NEG_FREQ,
                                   "F");
            try {
                process = processBuilder.start();
                Threaded_IOHandler ioHandler = new Threaded_IOHandler(process.getInputStream());
                ioHandler.start();
                process.waitFor(TIMEOUT_VAL, TIMEOUT_UNIT);
                ioHandler.join();

                Map<Double, Map<Double, List<String>>> map_CurrClass_FreqStrings = ioHandler.getParsedFrequentStrings();
                
                map_ClassValues_UnfilteredFreqStrings.put(classValue, map_CurrClass_FreqStrings);
            } catch (IOException e) {
                throw new IOException("Error occurred while initiating string mining task.", e);
            } catch (InterruptedException e) {
                throw new RuntimeException("The string mining process did not complete within the set limit of " +
                                           TIMEOUT_VAL + TIMEOUT_UNIT.toString(), e);
            }
        }
        
        timer = System.currentTimeMillis() - timer;
        TBL_FREQ_PATT_EXTRACT_TIMES.put(alpha, omega, timer);
        
        // Delete the files created earlier, these files will not be needed any longer
        Arrays.stream(Paths.get(TEMP_DIR).toFile()
                                         .listFiles((d, name) -> name.startsWith(fileNamePrefix)))
              .forEach(File::delete);
        
        // Sort and filter the extracted patterns based on independence tests
        int numPositiveInsts, numNegativeInsts;
        
        double srcEntropy, independenceVal;
        ChiSquareTest chiSqTest = new ChiSquareTest();
        long[][] counts = new long[2][2];
        List<String> list;
        
        Map<Double, Map<Double, List<String>>> map_ClassValues_FilteredFreqStrings;
        
        for (PatternIndependenceTest indTest : map_PattIndTest_Tbl_FreqPatts.keySet()) {
            
            map_ClassValues_FilteredFreqStrings = new HashMap<>();
            
            for (Double classValue : map_ClassValues_UnfilteredFreqStrings.keySet()) {
                map_ClassValues_FilteredFreqStrings.put(classValue, new TreeMap<>(Collections.reverseOrder()));
            }
            
            map_PattIndTest_Tbl_FreqPatts.get(indTest).put(alpha, omega, map_ClassValues_FilteredFreqStrings);
        }
        
        Map<Double, List<String>> map_IndVal_FilteredFreqStrings;
        
        for (Double classValue : map_ClassValues_UnfilteredFreqStrings.keySet()) {
            numPositiveInsts = this.getClassLabelsToInstanceCounts().get(classValue);
            numNegativeInsts = TOTAL_INSTANCES - numPositiveInsts;
            srcEntropy = this.getEntropy(numPositiveInsts, 1.0, numNegativeInsts, 1.0);
            
            for (Double positiveFreq : map_ClassValues_UnfilteredFreqStrings.get(classValue).keySet()) {

                for (Double negativeFreq : map_ClassValues_UnfilteredFreqStrings.get(classValue).get(positiveFreq).keySet()) {
                    
                    for (PatternIndependenceTest indTest : map_PattIndTest_Tbl_FreqPatts.keySet()) {
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
                                independenceVal = chiSqTest.chiSquare(counts) / TOTAL_INSTANCES;  // /totalInstances ?? normalizes the independence value [0, 1]
                                independenceVal = Math.round(independenceVal*1e3)/1e3;    // Convert to an integer then divide again to reduce floating point accuracy
                                break;
                                
                            case IG:
                                independenceVal = getInformationGain(TOTAL_INSTANCES,
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
                        
                        map_ClassValues_FilteredFreqStrings = map_PattIndTest_Tbl_FreqPatts.get(indTest)
                                                                                                .get(alpha, omega);
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
                        map_PattIndTest_PatternFilterringTime.put(indTest, timer + map_PattIndTest_PatternFilterringTime.get(indTest));
                    }
                }
            }
        }
    }
    
    private void writeSymbolicTrainingSplitToFiles(String fileNamePrefix, AlphaOmegaPair pair) throws IOException {
        long timer = System.currentTimeMillis();
        Map<Double, BufferedWriter[]> map_ClassValues_BufferedWriters = new HashMap<>();

        for (Double classValue : this.getClassLabelsToInstanceCounts().keySet()) {
            this.createBufferedWriters(fileNamePrefix,
                                       classValue,
                                       map_ClassValues_BufferedWriters);
        }

        Instances saxedInstances = this.getSymbolizedInsts(SplitType.TRAIN, pair);

        for (int instanceIndex = 0; instanceIndex < saxedInstances.numInstances(); instanceIndex++) {
            this.writeData(instanceIndex, saxedInstances.get(instanceIndex), map_ClassValues_BufferedWriters);
        }

        this.closeBufferedWriters(map_ClassValues_BufferedWriters);
        
        timer = (System.currentTimeMillis() - timer)
                + this.getDataTransformationTime(SplitType.TRAIN, pair);
        
        this.setDataTransformationTime(SplitType.TRAIN, pair, timer);
    }

    private void createBufferedWriters(String fileNamePrefix, Double classValue, Map<Double, BufferedWriter[]> map_ClassValues_BufferedWriters) throws IOException {
        BufferedWriter[] bwArray = new BufferedWriter[2];
        try {
            bwArray[0]  = Files.newBufferedWriter(Paths.get(TEMP_DIR,
                                                            fileNamePrefix
                                                            + classValue
                                                            + "vsALL.fasta"),
                                                  StandardOpenOption.CREATE,
                                                  StandardOpenOption.TRUNCATE_EXISTING);
            bwArray[1]  = Files.newBufferedWriter(Paths.get(TEMP_DIR,
                                                            fileNamePrefix
                                                            + "ALLvs"
                                                            + classValue
                                                            + ".fasta"),
                                                  StandardOpenOption.CREATE,
                                                  StandardOpenOption.TRUNCATE_EXISTING);
            
            map_ClassValues_BufferedWriters.put(classValue, bwArray);
        } catch (IOException e) {
            throw new IOException("Error occurred while creating buffered writers for writing binary class dataset splits.", e);
        }
    }
    
    private void writeData(int instInd, Instance instSymbolicForm, Map<Double, BufferedWriter[]> map_ClassValues_BufferedWriters) throws IOException {
        String symbolicForm = instSymbolicForm.stringValue(0);
        String currInstClassLabel = instSymbolicForm.stringValue(instSymbolicForm.classIndex());
        BufferedWriter bw;
        try {
            for (Double destFileClassValue : map_ClassValues_BufferedWriters.keySet()) {
                if (this.nearlyEqual(instSymbolicForm.classValue(), destFileClassValue, 1e-3)) {
                    bw = map_ClassValues_BufferedWriters.get(destFileClassValue)[0];
                } else {
                    bw = map_ClassValues_BufferedWriters.get(destFileClassValue)[1];
                }
                bw.write("> label:" + currInstClassLabel + " - id:" + instInd + "\n" + symbolicForm + "\n");
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
        private final InputStream INPUT_STREAM;
        private Map<Double, Map<Double, List<String>>> frequentStringsMap;
        private final Comparator<String> strComp = (String s1, String s2) -> {
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
            INPUT_STREAM = inputStream;
        }
        
        @Override
        public void run() {
            String line;
            String[] freqPattArray;
            frequentStringsMap = new TreeMap<>(Collections.reverseOrder());
            Map<Double, List<String>> negMap;
            List<String> strList;
            double posFreq, negFreq;
            try (BufferedReader br = new BufferedReader(new InputStreamReader(INPUT_STREAM))) {
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
            frequentStringsMap
                .keySet()
                .stream()
                .forEach( posFreq -> frequentStringsMap.get(posFreq)
                                         .keySet()
                                         .stream()
                                         .forEach( negFreq -> Collections.sort(frequentStringsMap
                                                                                   .get(posFreq)
                                                                                   .get(negFreq),
                                                                               strComp)
                                                 )
                        );
            
//            frequentStringsMap.keySet().stream().limit(2)
//                .forEach( posFreq -> frequentStringsMap
//                                         .get(posFreq).keySet().stream().limit(2)
//                                         .forEach( negFreq -> System.out.println(posFreq + " " + negFreq + " " + frequentStringsMap.get(posFreq).get(negFreq))));
            return frequentStringsMap;
        }
    }
    
    private String getCurrentFilePrefix(AlphaOmegaPair pair) {
        StringBuilder strb = new StringBuilder();
        return strb.append("Alpha_")
                   .append(pair.getAlpha())
                   .append("_Omega_")
                   .append(pair.getOmega())
                   .append("_")
                   .toString();
    }
    
    private double getEntropy(int posInsts, double posFreq, int negInsts, double negFreq) {
        double frac = 1.0 / (posFreq*posInsts + negFreq*negInsts);
        double entropy = 0.0;
        double temp;
        temp = posFreq*posInsts*frac;
        entropy -= (temp > 0.0) ? temp*Math.log(temp)/LOG2 : 0.0;
        temp = negFreq*negInsts*frac;
        entropy -= (temp > 0.0) ? temp*Math.log(temp)/LOG2 : 0.0;
        
        return entropy;
    }
    
    private double getInformationGain(int totalInstances, double srcEntropy, int posInsts, double posFreq, int negInsts, double negFreq) {
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
}
