package timeseriesweka.kramerlab.pbtsm;

import com.google.common.base.Splitter;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;

import java.io.IOException;

import java.nio.file.Paths;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ml.dmlc.xgboost4j.java.XGBoostError;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import timeseriesweka.kramerlab.pbtsm.misticl.ClassifierType;
import timeseriesweka.kramerlab.pbtsm.misticl.FeatureSetType;
import timeseriesweka.kramerlab.pbtsm.misticl.MiSTiCl;
import timeseriesweka.kramerlab.pbtsm.misticl.ParameterSelectionMethod;
import timeseriesweka.kramerlab.pbtsm.misticl.PatternApproximationMethod;
import timeseriesweka.kramerlab.pbtsm.misticl.PatternIndependenceTest;

public class LaunchMiSTiCl extends BaseLauncherSAX {

    public static void main(String[] args) {
        BaseLauncher obj = new LaunchMiSTiCl();
        obj.setupAndPerformExperiments(args);
    }

    private final String L_OPT_CLASSIFIERS = "classifiers";
    private final String L_OPT_EXEC_SLOTS = "exec-slots";
    private final String L_OPT_FEATURE_SET_TYPE = "feature-set-type";
    private final String L_OPT_MAX_PATT_PER_CLASS = "patt-per-class";
    private final String L_OPT_PARAM_SELECT_METHOD = "param-selection";
    private final String L_OPT_PATT_APPROX_METHOD = "patt-approx-method";
    private final String L_OPT_PATT_IND_TEST = "patt-independence-test";
    private final String L_OPT_TEMP_DIR = "temp-dir";

    private final String DEF_CLASSIFIERS = ClassifierType.ET.name();
    private final String DEF_EXEC_SLOTS = "1";
    private final String DEF_FEATURE_SET_TYPE = FeatureSetType.NUM.name();
    private final String DEF_MAX_PATT_PER_CLASS = "2";
    private final String DEF_PARAM_SELECT_METHOD = ParameterSelectionMethod.HSC.name();
    private final String DEF_PATT_APPROX_METHOD = PatternApproximationMethod.S.name();
    private final String DEF_PATT_IND_TEST = PatternIndependenceTest.CS.name();
    private final String DEF_TEMP_DIR = Paths.get(PROGRAM_DIR, "temp").toString();

    private final Set<ClassifierType> CLASSIFIERS;
    private final Set<FeatureSetType> FEATURE_SET_TYPES;
    private final Set<ParameterSelectionMethod> PARAM_SELECTION_METHODS;
    private final Set<PatternApproximationMethod> PATTERN_APPROX_METHODS;
    private final Set<PatternIndependenceTest> PATTERN_IND_TESTS;
    private final Set<Integer> PATTERNS_PER_CLASS;

    // TODO: Use the Comparator.comparing(...).thenComparing(...) methods
    private final Comparator<AlphaWindowPair> PAIR_COMPARATOR = (AlphaWindowPair p1, AlphaWindowPair p2) -> p1.toString().compareTo(p2.toString());

    private final Comparator<List<AlphaWindowPair>> AW_LIST_COMPARATOR = (List<AlphaWindowPair> l1, List<AlphaWindowPair> l2) -> {
        if (l1.size() < l2.size()) {
            return -1;
        } else if (l1.size() > l2.size()) {
            return 1;
        } else {
            return l1.toString().compareTo(l2.toString());
        }
    };

    private List<AlphaWindowPair> defaultPairsList, bestPairsList;

    /**
     * The name of the string mining executable (must be in the program directory).
     */
    private String stringMiningExecutable;

    // String mining parameters
    private int logV;
    private int blockSize;

    /**
     * Number of (alpha, window) pairs used for creating the final feature set.
     */
    private int numOfAWPairsToUse;

    private double minPositiveFreq;
    private double maxNegativeFreq;

    LaunchMiSTiCl() {
        super();

        this.DEF_ALPHAS = "3 4 5 6 7 8";

        this.DEF_WINDOWS = "2 3 4 5 6 7";

        this.RESULTS_FILE_HEADER = "Iter,Acc-Train,Acc-Test,TotalTime-Train,TotalTime-Test,"
                                   + "ParamOptTime,DiscretizationTime-Train,FPExtractionTime,"
                                   + "FeatureDataCreationTime-Train,ModelTrainingTime,"
                                   + "DiscretizationTime-Test,FeatureDataCreationTime-Test,"
                                   + "ModelTestingTime,PatternsPerClass,SetofParams\n";

        PATTERNS_PER_CLASS = new HashSet<>();

        CLASSIFIERS = EnumSet.noneOf(ClassifierType.class);

        FEATURE_SET_TYPES = EnumSet.noneOf(FeatureSetType.class);

        PATTERN_IND_TESTS = EnumSet.noneOf(PatternIndependenceTest.class);

        PATTERN_APPROX_METHODS = EnumSet.noneOf(PatternApproximationMethod.class);

        PARAM_SELECTION_METHODS = EnumSet.noneOf(ParameterSelectionMethod.class);
    }

    /**
     * Load the {@link Properties} from the program directory.
     *
     * @throws IOException
     */
    @Override
    void loadBaseProperties() throws IOException {
        super.loadBaseProperties();

        stringMiningExecutable = this.props.getProperty("string_miner_executable");

        logV = Integer.parseInt(this.props.getProperty("log_v", "5"));

        blockSize = Integer.parseInt(this.props.getProperty("block_size", "100"));

        minPositiveFreq = Double.parseDouble(this.props.getProperty("min-pos-freq", "0.2"));

        maxNegativeFreq = Double.parseDouble(this.props.getProperty("max-neg-freq", "0.1"));

        numOfAWPairsToUse = Integer.parseInt(this.props.getProperty("pairs_to_use", "4"));
    }

    @Override
    void parseCliArguments(String[] args) throws IllegalArgumentException, IOException,
                                                 NumberFormatException, ParseException {
        Options opts = new Options();
        // Add the different CLI paramaters to the OPTIONS variable
                        // Temporary directory path
        opts.addOption(Option.builder()
                             .longOpt(L_OPT_TEMP_DIR)
                             .hasArg().argName("path/to/directory")
                             .desc("An absolute/relative path to a temporary directory. (Preferably"
                                   + " a memory resident location. [Default: " + DEF_TEMP_DIR + "]")
                             .build())
                        // Number of execution slots to be used for MiSTiCl and Weka Classifiers
            .addOption(Option.builder()
                             .longOpt(L_OPT_EXEC_SLOTS)
                             .hasArg().argName("num-of-threads")
                             .desc("Number of threads to use. [Default: " + DEF_EXEC_SLOTS + "]")
                             .build())
                        // Maximum number of patterns per class to be used
            .addOption(Option.builder()
                             .longOpt(L_OPT_MAX_PATT_PER_CLASS)
                             .hasArgs().argName("p1 [p2 p3 ...]")
                             .desc("Space delimited set of numbers for determining maximum patterns"
                                   + " to use per class. Parameter optimization will be employed to"
                                   + " find the best number of patterns per class to be used. Valid"
                                   + " range: (0, 10]. [Default: " + DEF_MAX_PATT_PER_CLASS + "]")
                             .build())
                        // Independence tests to be used for pattern filering
            .addOption(Option.builder()
                             .longOpt(L_OPT_PATT_IND_TEST)
                             .hasArgs().argName("independence-test-name(s)")
                             .desc("Space delimited Pattern selection and variant removal method(s)\n"
                                   + PatternIndependenceTest.CS.name() + ": "
                                   + PatternIndependenceTest.CS.toString() + "\n"
                                   + PatternIndependenceTest.IG.name() + ": "
                                   + PatternIndependenceTest.IG.toString() + "\n"
                                   + PatternIndependenceTest.NF.name() + ": "
                                   + PatternIndependenceTest.NF.toString() + "\n"
                                   + PatternIndependenceTest.NO.name() + ": "
                                   + PatternIndependenceTest.NF.toString() + "\n"
                                   + "[Default: " + DEF_PATT_IND_TEST + "]")
                             .build())
                        // Pattern to subsequence approximation method
            .addOption(Option.builder()
                             .longOpt(L_OPT_PATT_APPROX_METHOD)
                             .hasArgs().argName("pattern-approximation-method(s)")
                             .desc("Space delimited Pattern approximation method(s)\n"
                                   + PatternApproximationMethod.S.name() + ": "
                                   + PatternApproximationMethod.S.toString() + "\n"
                                   + PatternApproximationMethod.I.name() + ": "
                                   + PatternApproximationMethod.I.toString() + "\n"
                                   + "[Default: " + DEF_PATT_APPROX_METHOD + "]")
                             .build())
                        // Feature set type to be used
            .addOption(Option.builder()
                             .longOpt(L_OPT_FEATURE_SET_TYPE)
                             .hasArgs().argName("feature-set-type(s)")
                             .desc("Space delimited Feature set type(s) to be created and used for "
                                   + "model induction\n"
                                   + FeatureSetType.NUM.name() + ": "
                                   + FeatureSetType.NUM.toString() + "\n"
                                   + FeatureSetType.BIN.name() + ": "
                                   + FeatureSetType.BIN.toString() + "\n"
                                   + "[Default: " + DEF_FEATURE_SET_TYPE + "]")
                             .build())
                        // Different classifiers to be trained
            .addOption(Option.builder()
                             .longOpt(L_OPT_CLASSIFIERS)
                             .hasArgs().argName("classifier(s)")
                             .desc("Space delimited classifier names to be used for model induction\n"
                                   + ClassifierType.AB.name() + ": "
                                   + ClassifierType.AB.toString() + "\n"
                                   + ClassifierType.ET.name() + ": "
                                   + ClassifierType.ET.toString() + "\n"
                                   + ClassifierType.RF.name() + ": "
                                   + ClassifierType.RF.toString() + "\n"
                                   + "[Default: " + DEF_CLASSIFIERS + "]")
                             .build())
                        // Parameter selection methods
            .addOption(Option.builder()
                             .longOpt(L_OPT_PARAM_SELECT_METHOD)
                             .hasArgs().argName("parameter-selection-method(s)")
                             .desc("Space delimited set of Parameter selection method(s)\n"
                                   + ParameterSelectionMethod.NOP.name() + ": "
                                   + ParameterSelectionMethod.NOP.toString() + "\n"
                                   + ParameterSelectionMethod.BRF.name() + ": "
                                   + ParameterSelectionMethod.BRF.toString() + "\n"
                                   + ParameterSelectionMethod.HSC.name() + ": "
                                   + ParameterSelectionMethod.HSC.toString() + "\n"
                                   + "[Default: " + DEF_PARAM_SELECT_METHOD + "]")
                             .build());

        this.addAdditionalCLIOptions(opts);

        super.parseCliArguments(args);

        // Populate required objects from the parsed CLI arguments
        List<String> optVals;

        Splitter splitter = Splitter.on(" ").omitEmptyStrings().trimResults();

        // Populate PATT_PER_CLASS_SET
        if (parsedCLIArgs.hasOption(L_OPT_MAX_PATT_PER_CLASS)) {
            optVals = Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_MAX_PATT_PER_CLASS));
        } else {
            optVals = splitter.splitToList(DEF_MAX_PATT_PER_CLASS);
        }
        this.addValuesTo(PATTERNS_PER_CLASS, 1, Integer.MAX_VALUE, optVals);

        //Populate PATT_IND_TEST_SET
        if (parsedCLIArgs.hasOption(L_OPT_PATT_IND_TEST)) {
            optVals = Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_PATT_IND_TEST));
        } else {
            optVals = splitter.splitToList(DEF_PATT_IND_TEST);
        }
        this.addValuesTo(PatternIndependenceTest.class, PATTERN_IND_TESTS, optVals);

        // Populate PATT_APPROX_METHOD
        if (parsedCLIArgs.hasOption(L_OPT_PATT_APPROX_METHOD)) {
            optVals = Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_PATT_APPROX_METHOD));
        } else {
            optVals = splitter.splitToList(DEF_PATT_APPROX_METHOD);
        }
        this.addValuesTo(PatternApproximationMethod.class, PATTERN_APPROX_METHODS, optVals);

        // Populate FEATURE_SET_TYPE_SET
        if (parsedCLIArgs.hasOption(L_OPT_FEATURE_SET_TYPE)) {
            optVals = Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_FEATURE_SET_TYPE));
        } else {
            optVals = splitter.splitToList(DEF_FEATURE_SET_TYPE);
        }
        this.addValuesTo(FeatureSetType.class, FEATURE_SET_TYPES, optVals);

        // Populate CLASSIFIERS_SET
        if (parsedCLIArgs.hasOption(L_OPT_CLASSIFIERS)) {
            optVals = Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_CLASSIFIERS));
        } else {
            optVals = splitter.splitToList(DEF_CLASSIFIERS);
        }
        this.addValuesTo(ClassifierType.class, CLASSIFIERS, optVals);

        //Populate PARAM_SELECT_METHOD_SET
        if (parsedCLIArgs.hasOption(L_OPT_PARAM_SELECT_METHOD)) {
            optVals = Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_PARAM_SELECT_METHOD));
        } else {
            optVals = splitter.splitToList(DEF_PARAM_SELECT_METHOD);
        }
        this.addValuesTo(ParameterSelectionMethod.class, PARAM_SELECTION_METHODS, optVals);
    }

    @Override
    void executeExperiments() throws Exception {
        MiSTiCl misticl;

        // A Table used to save the best accuracy values for each Alpha and Window combination
        Table<FeatureSetType, ClassifierType, Double> bestAccuracyTable = HashBasedTable.create();

        // A Table to save the result string for each Alpha and Window combination
        Table<FeatureSetType, ClassifierType, String> resultStrings = HashBasedTable.create();

        long paramOptTime;

        int seed;
        // For each dataset
        for (String dataset : datasets) {
            // Create a RealValuedDataset object which loads the training and testing splits
            rvDataset = new RealValuedDataset(dataDir, dataset);

            // Get all possible pairs of Alphabet and Window sizes
            defaultPairsList = this.generateListAlphaWindowPair(setOfAlphabets, setOfWindows);

            while ((seed = this.getNextSeed()) != -1) {
                misticl = new MiSTiCl(rvDataset, defaultPairsList, getNumberOfExecSlots(), seed,
                                      getTempDir(), stringMiningExecutable, logV, blockSize,
                                      minPositiveFreq, maxNegativeFreq, PATTERN_IND_TESTS,
                                      FEATURE_SET_TYPES);

                // For each subsequence approximation method
                for (PatternApproximationMethod approxMethod : this.PATTERN_APPROX_METHODS) {
                    // For each independence test to be used
                    for (PatternIndependenceTest indTest : this.PATTERN_IND_TESTS) {
                        // For each parameter optimization method
                        for (ParameterSelectionMethod paramSelectionMethod : this.PARAM_SELECTION_METHODS) {
                            // Clear the best accuracy and results strings tables
                            bestAccuracyTable.clear();
                            resultStrings.clear();

                            // For each provided number of patterns per class
                            for (Integer patternsPerClass : this.PATTERNS_PER_CLASS) {
                                // Create the feature sets using the current subsequence approximation method,
                                // the current independence test, and the current pattern per class count
                                misticl.createFeatureSets(approxMethod, indTest, patternsPerClass);
                                // For each feature set type that is to be evaluated
                                for (FeatureSetType featureSetType : this.FEATURE_SET_TYPES) {
                                    // For each classifier type that is to be tested
                                    for (ClassifierType classifier : this.CLASSIFIERS) {

                                        //////////////////////////////////
                                        // Perform parameter optimization
                                        paramOptTime = 0L;
                                        // If parameter optimization is not being used
                                        if ((paramSelectionMethod == ParameterSelectionMethod.NOP)) {
                                            // Simply select the current 
                                            bestPairsList = defaultPairsList;
                                        } else {
                                            paramOptTime = System.currentTimeMillis();
                                            switch (paramSelectionMethod) {
                                                case BRF:
                                                    bestPairsList
                                                        = findAWPairsUsingBruteForce(misticl,
                                                                                     featureSetType,
                                                                                     classifier);
                                                    break;
                                                case HSC:
                                                    bestPairsList
                                                        = findAWPairsUsingHeuristicSC(misticl,
                                                                                      featureSetType,
                                                                                      classifier);
                                                    break;
                                                case NSC:
                                                    bestPairsList
                                                        = findAWPairsUsingNaiveSC(misticl,
                                                                                  featureSetType,
                                                                                  classifier);
                                                    break;
                                            }
                                            paramOptTime = System.currentTimeMillis() - paramOptTime;
                                        }
                                        misticl.setAlphaWindowPairs(bestPairsList);

                                        misticl.combineFeatureSets(featureSetType);

                                        // misticl.saveTransformedData(paramSelectionMethod, indTest, f, patternsPerClass);

                                        misticl.trainClassifier(classifier, featureSetType);

                                        if (!bestAccuracyTable.contains(featureSetType, classifier) || bestAccuracyTable.get(featureSetType, classifier) < misticl.getSplitAccuracy(SplitType.TEST, featureSetType)) {
                                            Collections.sort(bestPairsList, PAIR_COMPARATOR);
                                            bestAccuracyTable.put(featureSetType, classifier, misticl.getSplitAccuracy(SplitType.TEST, featureSetType));
                                            RESULT_BLDR.delete(0, RESULT_BLDR.length())
                                                       .append(seed)
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(ACCURACY_FORMAT, misticl.getSplitAccuracy(SplitType.TRAIN, featureSetType)))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(ACCURACY_FORMAT, misticl.getSplitAccuracy(SplitType.TEST, featureSetType)))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, (misticl.getTotalTime(SplitType.TRAIN, featureSetType, classifier) + paramOptTime) / 1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, misticl.getTotalTime(SplitType.TEST, featureSetType, classifier) / 1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, paramOptTime /1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, misticl.getDataTransformationTime(SplitType.TRAIN) / 1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, misticl.getPatternExtractionTime() /1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, misticl.getFeatureSetCreationTime(SplitType.TRAIN, featureSetType) / 1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, misticl.getModelTrainTestTime(SplitType.TRAIN, featureSetType, classifier) / 1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, misticl.getDataTransformationTime(SplitType.TEST) / 1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, misticl.getFeatureSetCreationTime(SplitType.TEST, featureSetType) /1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(String.format(RUNTIME_FORMAT, misticl.getModelTrainTestTime(SplitType.TEST, featureSetType, classifier) / 1e3))
                                                       .append(RESULT_FIELD_SEP)
                                                       .append(patternsPerClass)
                                                       .append(RESULT_FIELD_SEP)
                                                       .append('\'').append(bestPairsList).append('\'')
                                                       .append(System.lineSeparator());
                                            resultStrings.put(featureSetType, classifier, RESULT_BLDR.toString());
                                        }
                                    }
                                }
                            }

                            if (!this.isWarmUpRun()) {
                                for (FeatureSetType featureSetType : resultStrings.rowKeySet()) {
                                    for (ClassifierType classifier : resultStrings.columnKeySet()) {
                                        this.writeResultsToFile(dataset,
                                                                "MiSTiCl_"
                                                                + paramSelectionMethod.name() + "_"
                                                                + indTest.name() + "_"
                                                                + featureSetType.name() + "_"
                                                                + classifier.name(),
                                                                resultStrings.get(featureSetType, classifier));
                                    }
                                }
                            }
                        }
                    }
                }
                if (this.isWarmUpRun()) {
                    this.disableWarmUp();
                    break;
                }
            }
            this.resetSeeds();
        }
    }

    private <E extends Enum<E>> void addValuesTo(Class<E> enumClass, Set<E> destSet, List<String> values) {
        values.forEach(val -> destSet.add(Enum.valueOf(enumClass, val.toUpperCase(Locale.ENGLISH))));
    }

    private List<AlphaWindowPair> findAWPairsUsingBruteForce(MiSTiCl misticl,
                                                             FeatureSetType featureSetType,
                                                             ClassifierType classifier) throws XGBoostError,
                                                                                               Exception {
        SplitType optimizationSplit = SplitType.TEST;

        List<AlphaWindowPair> listOfPairs;

        List<List<AlphaWindowPair>> listOfBestAWPairsLists = new ArrayList<>();

        int numOfCoveredInsts, bestCover = 0;

        for (Set<Integer> alphas : Sets.powerSet(setOfAlphabets)) {
            if (alphas.isEmpty()) {
                continue;
            }

            for (Set<Integer> windows : Sets.powerSet(setOfWindows)) {
                if (windows.isEmpty()) {
                    continue;
                }

                listOfPairs = this.generateListAlphaWindowPair(alphas, windows);
                misticl.setAlphaWindowPairs(listOfPairs);
                misticl.combineFeatureSets(featureSetType);

                misticl.trainClassifier(classifier, featureSetType, 50);
                numOfCoveredInsts = misticl.getSplitClassifications(optimizationSplit).size();

                if (numOfCoveredInsts > bestCover) {
                    bestCover = numOfCoveredInsts;
                    listOfBestAWPairsLists.clear();
                    Collections.sort(listOfPairs, PAIR_COMPARATOR);
                    listOfBestAWPairsLists.add(listOfPairs);
                } else if (numOfCoveredInsts == bestCover) {
                    Collections.sort(listOfPairs, PAIR_COMPARATOR);
                    listOfBestAWPairsLists.add(listOfPairs);
                }
            }
        }

        Collections.sort(listOfBestAWPairsLists, AW_LIST_COMPARATOR);
        return listOfBestAWPairsLists.get(0);
    }

    private List<AlphaWindowPair> findAWPairsUsingHeuristicSC(MiSTiCl misticl,
                                                              FeatureSetType featureSetType,
                                                              ClassifierType classifier) throws XGBoostError,
                                                                                                Exception {
        SplitType optimizationSplit = SplitType.TEST;

        Table<Integer, Integer, Set> splitClassificationsTable = HashBasedTable.create();
        Map<Integer, List<AlphaWindowPair>> totalCovInstsToAWPairsMap
                                            = new TreeMap<>(Collections.reverseOrder());

        List<AlphaWindowPair> listOfPairs = new ArrayList<>();

        List<AlphaWindowPair> listOfBestPairs = new ArrayList<>();

        for (Integer alpha : setOfAlphabets) {
            AlphaWindowPair pair;
            int numOfCoveredInsts;
            for (Integer step : setOfWindows) {
                pair = new AlphaWindowPair(alpha, step);
                listOfPairs.clear();
                listOfPairs.add(pair);
                misticl.setAlphaWindowPairs(listOfPairs);
                misticl.combineFeatureSets(featureSetType);
                misticl.trainClassifier(classifier, featureSetType, 50);

                splitClassificationsTable.put(alpha,
                                              step,
                                              new HashSet(misticl.getSplitClassifications(optimizationSplit)));
                numOfCoveredInsts = misticl.getSplitClassifications(optimizationSplit).size();

                if (numOfCoveredInsts > 0) {
                    if (!totalCovInstsToAWPairsMap.containsKey(numOfCoveredInsts)) {
                        totalCovInstsToAWPairsMap.put(numOfCoveredInsts, new ArrayList<>());
                    }
                    totalCovInstsToAWPairsMap.get(numOfCoveredInsts).add(pair);
                }
            }
        }

        Set<Integer> allOptSplitInsts
                     = new TreeSet<>(IntStream.range(0, misticl.getSplitSize(optimizationSplit))
                                              .boxed()
                                              .collect(Collectors.toList()));

        Set<Integer> currentCombo = Sets.newHashSet();

        mapLoop:
        for (Integer numOfCoveredInsts : totalCovInstsToAWPairsMap.keySet()) {
            listOfPairs = totalCovInstsToAWPairsMap.get(numOfCoveredInsts);

            listLoop:
            for (AlphaWindowPair pair : listOfPairs) {
                if (currentCombo.isEmpty()
                    || !Sets.difference(splitClassificationsTable.get(pair.getNumAlphabets(),
                                                                      pair.getWindowSize()),
                                        currentCombo)
                            .isEmpty()) {
                    listOfBestPairs.add(pair);
                    Sets.union(currentCombo, splitClassificationsTable.get(pair.getNumAlphabets(),
                                                                           pair.getWindowSize()))
                        .copyInto(currentCombo);

                    if (Sets.difference(allOptSplitInsts, currentCombo).isEmpty()
                        || (listOfBestPairs.size() == numOfAWPairsToUse)) {
                        break mapLoop;
                    }
                }
            }
        }
        return listOfBestPairs;
    }
    
    private List<AlphaWindowPair> findAWPairsUsingNaiveSC(MiSTiCl m, FeatureSetType f, ClassifierType c) throws XGBoostError, Exception {
        SplitType optimizationSplit = SplitType.TEST;

        AlphaWindowPair pair;

        List<AlphaWindowPair> listOfPairs = new ArrayList<>();
        List<AlphaWindowPair> awPairCorrespondingToClassification = new ArrayList<>();
        
        List<Set<Integer>> classificationsList = new ArrayList<>();

        for (Integer alpha : setOfAlphabets) {
            for (Integer step : setOfWindows) {
                pair = new AlphaWindowPair(alpha, step);
                listOfPairs.clear();
                listOfPairs.add(pair);
                m.setAlphaWindowPairs(listOfPairs);
                m.combineFeatureSets(f);
                m.trainClassifier(c, f, 50);
                awPairCorrespondingToClassification.add(pair);
                classificationsList.add(new HashSet(m.getSplitClassifications(optimizationSplit)));
            }
        }

        Set<Integer> totalClassifications
                     = new TreeSet<>(IntStream.range(0, classificationsList.size())
                                              .boxed()
                                              .collect(Collectors.toList()));

        Set<Integer> coveredInstances = new HashSet<>();

        List<List<AlphaWindowPair>> listOfBestAWPairsLists = new ArrayList<>();

        int bestCover = Integer.MIN_VALUE;

        for (Set<Integer> currentCombination : Sets.powerSet(totalClassifications)) {
            coveredInstances.clear();
            currentCombination.stream().forEach((ind) -> coveredInstances.addAll(classificationsList.get(ind)));
            if (coveredInstances.size() > bestCover) {
                bestCover = coveredInstances.size();
                listOfPairs = new ArrayList<>();
                for (Integer ind : currentCombination) {
                    listOfPairs.add(awPairCorrespondingToClassification.get(ind));
                }
                Collections.sort(listOfPairs, PAIR_COMPARATOR);
                listOfBestAWPairsLists.clear();
                listOfBestAWPairsLists.add(listOfPairs);
            } else if (coveredInstances.size() == bestCover) {
                listOfPairs = new ArrayList<>();
                for (Integer ind : currentCombination) {
                    listOfPairs.add(awPairCorrespondingToClassification.get(ind));
                }
                Collections.sort(listOfPairs, PAIR_COMPARATOR);
                listOfBestAWPairsLists.add(listOfPairs);
            }
        }
        Collections.sort(listOfBestAWPairsLists, AW_LIST_COMPARATOR);
        return listOfBestAWPairsLists.get(0);
    }

    private int getNumberOfExecSlots() {
        return Integer.parseInt(parsedCLIArgs.getOptionValue(L_OPT_EXEC_SLOTS, DEF_EXEC_SLOTS));
    }

    private String getTempDir() {
        return parsedCLIArgs.getOptionValue(L_OPT_TEMP_DIR, DEF_TEMP_DIR);
    }
}
