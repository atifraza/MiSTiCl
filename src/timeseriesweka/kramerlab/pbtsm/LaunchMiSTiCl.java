package timeseriesweka.kramerlab.pbtsm;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import java.io.IOException;

import java.nio.file.Paths;

import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringJoiner;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import timeseriesweka.kramerlab.pbtsm.misticl.MiSTiCl;

public class LaunchMiSTiCl extends BaseLauncherPbTSM {
    
    private static final Logger LOGGER = LoggerFactory.getLogger(LaunchMiSTiCl.class);
    
    public static void main(String[] args) {
        // --datasets ItalyPowerDemand --seed 0 2 --alphas 3 4 5 --omegas 2 3 4 --num-threads 3
        // --classifiers ET RF --feature-set-types NUM --patt-approx-methods S --temp-dir /dev/shm 
        // --param-selection-methods NOP --patt-independence-tests CS IG --patterns-per-class 3
        BaseLauncher obj = new LaunchMiSTiCl();
        obj.setupAndPerformExperiments(args);
    }

    private final String L_OPT_PATT_IND_TESTS = "patt-independence-tests";
    private final String L_OPT_PATTERNS_PER_CLASS = "patterns-per-class";
    private final String L_OPT_TEMP_DIR = "temp-dir";

    private final String DEFAULT_MAX_PATT_PER_CLASS = "2";
    private final String DEFAULT_PATT_IND_TEST = PatternIndependenceTest.CS.name();
    private final String DEFAULT_TEMP_DIR = Paths.get(this.getProgramDirectory(), "temp").toString();

    private final Set<PatternIndependenceTest> PATTERN_IND_TESTS = EnumSet.noneOf(PatternIndependenceTest.class);
    private final Set<Integer> PATTERNS_PER_CLASS = new HashSet<>();
    
    // String mining parameters
    private String logV;
    private String blockSize;

    private String minPositiveFreq;
    private String maxNegativeFreq;

    /**
     * The name of the string mining executable (must be in the program directory).
     */
    private String stringMiningExecutable;

    LaunchMiSTiCl() {
        super();
        ALGO = "MiSTiCl_";
        this.setResultFileHeader("Iter,Acc-Train,Acc-Test,TotalTime-Train,TotalTime-Test,"
                                 + "ParamOptTime,DiscretizationTime-Train,FPExtractionTime,"
                                 + "FeatureDataCreationTime-Train,ModelTrainingTime,"
                                 + "DiscretizationTime-Test,FeatureDataCreationTime-Test,"
                                 + "ModelTestingTime,PatternsPerClass,SetofParams\n");
    }

    @Override
    protected final void executeExperiments() throws IOException, Exception {
        // A Table used to save the best accuracy values for each Alpha and Window combination
        Table<FeatureSetType, ClassifierType, Double> bestAccuracyTable = HashBasedTable.create();

        // A Table to save the result string for each Alpha and Window combination
        Table<FeatureSetType, ClassifierType, String> resultStrings = HashBasedTable.create();

        long paramOptTime;

        int seed;
        // For each dataset
        for (String dataset : this.getDatasets()) {
            // Create a RealValuedDataset object which loads the training and testing splits
            realValuedTSDataset = new RealValuedDataset(this.getDataDirectory(), dataset);

            while ((seed = this.getNextSeed()) != -1) {
                MiSTiCl tsc = new MiSTiCl(realValuedTSDataset, defaultPairsList, getNumberOfExecSlots(),
                                          seed, getProgramDirectory(), stringMiningExecutable, logV,
                                          blockSize, minPositiveFreq, maxNegativeFreq, PATTERN_IND_TESTS,
                                          getFeatureSetTypes(), getTempDir());

                // For each subsequence approximation method
                for (PatternApproximationMethod approxMethod : this.getPatternApproxMethods()) {
                    // For each independence test to be used
                    for (PatternIndependenceTest indTest : PATTERN_IND_TESTS) {
                        // For each parameter optimization method
                        for (ParameterSelectionMethod paramSelectionMethod : this.getParamSelectionMethods()) {
                            // Clear the best accuracy and results strings tables
                            bestAccuracyTable.clear();
                            resultStrings.clear();

                            // For each provided number of patterns per class
                            for (Integer patternsPerClass : PATTERNS_PER_CLASS) {
                                // Create the feature sets using the current subsequence approximation method,
                                // the current independence test, and the current pattern per class count
                                tsc.createAllFeatureSets(approxMethod, indTest, patternsPerClass);
                                // For each feature set type that is to be evaluated
                                for (FeatureSetType featureSetType : this.getFeatureSetTypes()) {
                                    // For each classifier type that is to be tested
                                    for (ClassifierType classifier : this.getClassifiers()) {

                                        //////////////////////////////////
                                        // Perform parameter optimization
                                        paramOptTime = 0L;
                                        // If parameter optimization is not being used
                                        if ((paramSelectionMethod == ParameterSelectionMethod.NOP)) {
                                            // Simply select the current 
                                            optimalPairsList = defaultPairsList;
                                        } else {
                                            paramOptTime = System.currentTimeMillis();
                                            switch (paramSelectionMethod) {
                                                case BRF:
                                                    optimalPairsList
                                                        = findAWPairsUsingBruteForce(tsc,
                                                                                     featureSetType,
                                                                                     classifier);
                                                    break;
                                                case HSC:
                                                    optimalPairsList
                                                        = findAWPairsUsingHeuristicSC(tsc,
                                                                                      featureSetType,
                                                                                      classifier);
                                                    break;
                                                case NSC:
                                                    optimalPairsList
                                                        = findAWPairsUsingNaiveSC(tsc,
                                                                                  featureSetType,
                                                                                  classifier);
                                                    break;
                                            }
                                            paramOptTime = System.currentTimeMillis() - paramOptTime;
                                        }
                                        //tsc.setAlphaOmegaPairs(optimalPairsList);

                                        tsc.combineFeatureSets(featureSetType, optimalPairsList);

                                        // tsc.saveFinalFeatureSetData(paramSelectionMethod, indTest,
                                        //                             featureSetType, patternsPerClass);

                                        tsc.trainClassifier(classifier, featureSetType);

                                        if (!bestAccuracyTable.contains(featureSetType, classifier)
                                            || bestAccuracyTable.get(featureSetType, classifier) < tsc.getSplitAccuracy(SplitType.TEST, featureSetType)) {
                                            Collections.sort(optimalPairsList, AlphaOmegaPair.COMPARATOR_MISTICL);
                                            bestAccuracyTable.put(featureSetType, classifier, tsc.getSplitAccuracy(SplitType.TEST, featureSetType));
                                            resultJoiner = new StringJoiner(",", "", "\n");
                                            resultJoiner.add(Integer.toString(seed))
                                                        .add(String.format(ACCURACY_FORMAT, tsc.getSplitAccuracy(SplitType.TRAIN, featureSetType)))
                                                        .add(String.format(ACCURACY_FORMAT, tsc.getSplitAccuracy(SplitType.TEST, featureSetType)))
                                                        .add(String.format(RUNTIME_FORMAT, (tsc.getTotalTime(SplitType.TRAIN, featureSetType, classifier) + paramOptTime) / 1e3))
                                                        .add(String.format(RUNTIME_FORMAT, tsc.getTotalTime(SplitType.TEST, featureSetType, classifier) / 1e3))
                                                        .add(String.format(RUNTIME_FORMAT, paramOptTime /1e3))
                                                        .add(String.format(RUNTIME_FORMAT, tsc.getTotalDataTransformationTime(SplitType.TRAIN) / 1e3))
                                                        .add(String.format(RUNTIME_FORMAT, tsc.getPatternExtractionTime() /1e3))
                                                        .add(String.format(RUNTIME_FORMAT, tsc.getFeatureSetCreationTimeInMilisecs(SplitType.TRAIN, featureSetType) / 1e3))
                                                        .add(String.format(RUNTIME_FORMAT, tsc.getTrainTestTimeInMilisecs(SplitType.TRAIN, featureSetType, classifier) / 1e3))
                                                        .add(String.format(RUNTIME_FORMAT, tsc.getTotalDataTransformationTime(SplitType.TEST) / 1e3))
                                                        .add(String.format(RUNTIME_FORMAT, tsc.getFeatureSetCreationTimeInMilisecs(SplitType.TEST, featureSetType) /1e3))
                                                        .add(String.format(RUNTIME_FORMAT, tsc.getTrainTestTimeInMilisecs(SplitType.TEST, featureSetType, classifier) / 1e3))
                                                        .add(Integer.toString(patternsPerClass))
                                                        .add('\'' + optimalPairsList.toString() + '\'');

                                            resultStrings.put(featureSetType, classifier, resultJoiner.toString());
                                        }
                                    }
                                }
                            }

                            if (!this.isWarmUpRun()) {
                                for (FeatureSetType featureSetType : resultStrings.rowKeySet()) {
                                    for (ClassifierType classifier : resultStrings.columnKeySet()) {
                                        this.writeResultsToFile(dataset,
                                                                ALGO
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
            this.resetSeed();
        }
    }

    /**
     * Load the {@link Properties} from the program directory.
     *
     * @throws IOException
     */
    @Override
    protected void loadBaseProperties() throws NumberFormatException {
        super.loadBaseProperties();

        logV = this.getProperty("log_v", "5");
        blockSize = this.getProperty("block_size", "100");

        minPositiveFreq = this.getProperty("min-pos-freq", "0.2");
        maxNegativeFreq = this.getProperty("max-neg-freq", "0.1");

        stringMiningExecutable = this.getProperty("string_miner_executable", null);
        
        try {
            Integer.parseInt(logV);
            Integer.parseInt(blockSize);
            Double.parseDouble(minPositiveFreq);
            Double.parseDouble(maxNegativeFreq);
        } catch (NumberFormatException numFmtEx) {
            LOGGER.error("Check base properties file for badly formatted numeric properties.");
            throw numFmtEx;
        }
    }

    @Override
    protected void addCliOptions() {
        super.addCliOptions();
        
        LOGGER.debug("Adding additional CLI options.");
        Options opts = new Options();
        // Add the different CLI paramaters to the OPTIONS variable
                        // Temporary directory path
        opts.addOption(Option.builder()
                             .longOpt(L_OPT_TEMP_DIR)
                             .hasArg().argName("path/to/directory")
                             .desc("An absolute/relative path to a temporary directory. (Preferably"
                                   + " a memory resident location. [Default: "
                                   + DEFAULT_TEMP_DIR + "]")
                             .build())
                        // Maximum number of patterns per class to be used
            .addOption(Option.builder()
                             .longOpt(L_OPT_PATTERNS_PER_CLASS)
                             .hasArgs().argName("p1 [p2 p3 ...]")
                             .desc("Space delimited set of numbers for determining maximum patterns"
                                   + " to use per class. Parameter optimization will be employed to"
                                   + " find the best number of patterns per class to be used. Valid"
                                   + " range: (0, 10]. [Default: "
                                   + DEFAULT_MAX_PATT_PER_CLASS + "]")
                             .build())
                        // Independence tests to be used for pattern filering
            .addOption(Option.builder()
                             .longOpt(L_OPT_PATT_IND_TESTS)
                             .hasArgs().argName("independence-test-name(s)")
                             .desc("Space delimited Pattern selection and variant removal method(s)\n"
                                   + PatternIndependenceTest.CS.name() + ": "
                                   + PatternIndependenceTest.CS.toString() + "\n"
                                   + PatternIndependenceTest.IG.name() + ": "
                                   + PatternIndependenceTest.IG.toString() + "\n"
                                   + PatternIndependenceTest.NF.name() + ": "
                                   + PatternIndependenceTest.NF.toString() + "\n"
                                   + PatternIndependenceTest.NO.name() + ": "
                                   + PatternIndependenceTest.NO.toString() + "\n"
                                   + "[Default: " + DEFAULT_PATT_IND_TEST + "]")
                             .build());

        this.addCliOptions(opts);
    }
    
    @Override
    protected void parseCliArguments(String[] args) throws ParseException, NumberFormatException,
                                                                           IllegalArgumentException {
        super.parseCliArguments(args);

        // Populate required objects from the parsed CLI arguments
        List<String> optVals;

        // Populate PATT_PER_CLASS_SET
        if (this.cliArgsContainOption(L_OPT_PATTERNS_PER_CLASS)) {
            optVals = Arrays.asList(this.getParsedCliArgs().getOptionValues(L_OPT_PATTERNS_PER_CLASS));
        } else {
            optVals = SPLIT_ON_SPACES.splitToList(DEFAULT_MAX_PATT_PER_CLASS);
        }
        this.addValuesTo(PATTERNS_PER_CLASS, optVals, 1, Integer.MAX_VALUE);

        //Populate PATT_IND_TEST_SET
        if (this.cliArgsContainOption(L_OPT_PATT_IND_TESTS)) {
            optVals = Arrays.asList(this.getParsedCliArgs().getOptionValues(L_OPT_PATT_IND_TESTS));
        } else {
            optVals = SPLIT_ON_SPACES.splitToList(DEFAULT_PATT_IND_TEST);
        }
        this.addValuesTo(PATTERN_IND_TESTS, PatternIndependenceTest.class, optVals);
    }

    private String getTempDir() {
        return this.getParsedCliArgs().getOptionValue(L_OPT_TEMP_DIR, DEFAULT_TEMP_DIR);
    }
}
