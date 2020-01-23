package timeseriesweka.kramerlab.pbtsm;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;

import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ml.dmlc.xgboost4j.java.XGBoostError;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

abstract class BaseLauncherPbTSM extends BaseLauncherSAX {
    
    private static final Logger LOGGER = LoggerFactory.getLogger(BaseLauncherPbTSM.class);

    protected List<AlphaOmegaPair> defaultPairsList;
    protected List<AlphaOmegaPair> optimalPairsList;
    
    private final SplitType optimizationSplit = SplitType.TEST;

    private final String L_OPT_EXEC_SLOTS = "num-threads";
    private final String L_OPT_CLASSIFIERS = "classifiers";
    private final String L_OPT_FEATURE_SET_TYPE = "feature-set-types";
    private final String L_OPT_PATT_APPROX_METHOD = "patt-approx-methods";
    private final String L_OPT_PARAM_SELECT_METHOD = "param-selection-methods";
    
    private final String DEFAULT_EXEC_SLOTS = "1";
    private final String DEFAULT_CLASSIFIERS = ClassifierType.ET.name();
    private final String DEFAULT_FEATURE_SET_TYPE = FeatureSetType.NUM.name();
    private final String DEFAULT_PATT_APPROX_METHOD = PatternApproximationMethod.S.name();
    private final String DEFAULT_PARAM_SELECT_METHOD = ParameterSelectionMethod.NOP.name();
    
    private final Set<ClassifierType> CLASSIFIERS = EnumSet.noneOf(ClassifierType.class);
    private final Set<FeatureSetType> FEATURE_SET_TYPES = EnumSet.noneOf(FeatureSetType.class);
    private final Set<ParameterSelectionMethod> PARAM_SELECT_METHODS = EnumSet.noneOf(ParameterSelectionMethod.class);
    private final Set<PatternApproximationMethod> PATT_APROX_METHODS = EnumSet.noneOf(PatternApproximationMethod.class);
    
    private final Comparator<List<AlphaOmegaPair>> AW_LIST_COMPARATOR = Comparator.comparing(List<AlphaOmegaPair>::size)
                                                                                  .thenComparing(List<AlphaOmegaPair>::toString);

    /**
     * Number of (alpha, window) pairs used for creating the final feature set.
     */
    private int numOfAlphaOmegaPairsToUse;

    BaseLauncherPbTSM() {
        super();
        
        defaultAlphas = "3 4 5 6 7";
        defaultOmegas = "2 3 4 5 6 7";
    }
    
    protected final <E extends Enum<E>> void addValuesTo(Set<E> destinationSet, Class<E> destinationEnum, List<String> values) {
        values.forEach(value -> destinationSet.add(Enum.valueOf(destinationEnum, value.toUpperCase(Locale.ENGLISH))));
    }

    protected final int getNumberOfExecSlots() {
        return Integer.parseInt(this.getParsedCliArgs().getOptionValue(L_OPT_EXEC_SLOTS, DEFAULT_EXEC_SLOTS));
    }
    
    protected final Set<ClassifierType> getClassifiers() {
        return CLASSIFIERS;
    }

    protected final Set<FeatureSetType> getFeatureSetTypes() {
        return FEATURE_SET_TYPES;
    }

    protected final Set<ParameterSelectionMethod> getParamSelectionMethods() {
        return PARAM_SELECT_METHODS;
    }

    protected final Set<PatternApproximationMethod> getPatternApproxMethods() {
        return PATT_APROX_METHODS;
    }

    protected final List<AlphaOmegaPair> createListOfAlphaOmegaPairs(Set<Integer> alphas,
                                                                     Set<Integer> omegas) {
        List<AlphaOmegaPair> result = new ArrayList<>();
        
        alphas.forEach(alpha -> {
            omegas.forEach(omega -> {
                result.add(new AlphaOmegaPair(alpha, omega));
            });
        });
        return result;
    }

    protected final List<AlphaOmegaPair> findAWPairsUsingBruteForce(BasePbTSM tscModel,
                                                                    FeatureSetType featureSetType,
                                                                    ClassifierType classifierType) throws XGBoostError,
                                                                                                          Exception {

        List<AlphaOmegaPair> listOfPairs;

        List<List<AlphaOmegaPair>> listOfBestAWPairsLists = new ArrayList<>();

        int numOfCoveredInsts, bestCover = 0;

        for (Set<Integer> alphas : Sets.powerSet(this.getAlphas())) {
            if (alphas.isEmpty()) {
                continue;
            }

            for (Set<Integer> windows : Sets.powerSet(this.getOmegas())) {
                if (windows.isEmpty()) {
                    continue;
                }

                listOfPairs = this.createListOfAlphaOmegaPairs(alphas, windows);
                //tscModel.setAlphaOmegaPairs(listOfPairs);
                tscModel.combineFeatureSets(featureSetType, listOfPairs);

                tscModel.trainClassifier(classifierType, featureSetType, 50);
                numOfCoveredInsts = tscModel.getClassificationsFor(optimizationSplit).size();

                if (numOfCoveredInsts > bestCover) {
                    bestCover = numOfCoveredInsts;
                    listOfBestAWPairsLists.clear();
                    Collections.sort(listOfPairs, AlphaOmegaPair.COMPARATOR_MISTICL);
                    listOfBestAWPairsLists.add(listOfPairs);
                } else if (numOfCoveredInsts == bestCover) {
                    Collections.sort(listOfPairs, AlphaOmegaPair.COMPARATOR_MISTICL);
                    listOfBestAWPairsLists.add(listOfPairs);
                }
            }
        }

        Collections.sort(listOfBestAWPairsLists, AW_LIST_COMPARATOR);
        return listOfBestAWPairsLists.get(0);
    }

    protected final List<AlphaOmegaPair> findAWPairsUsingHeuristicSC(BasePbTSM tscModel,
                                                                     FeatureSetType featureSetType,
                                                                     ClassifierType classifierType) throws XGBoostError,
                                                                                                           Exception {

        Table<Integer, Integer, Set> splitClassificationsTable = HashBasedTable.create();
        Map<Integer, List<AlphaOmegaPair>> totalCovInstsToAWPairsMap = new TreeMap<>(Collections.reverseOrder());

        List<AlphaOmegaPair> listOfPairs = new ArrayList<>();

        List<AlphaOmegaPair> listOfBestPairs = new ArrayList<>();

        for (Integer alpha : this.getAlphas()) {
            for (Integer step : this.getOmegas()) {
                int numOfCoveredInsts;
                AlphaOmegaPair pair = new AlphaOmegaPair(alpha, step);
                listOfPairs.clear();
                listOfPairs.add(pair);
                //tscModel.setAlphaOmegaPairs(listOfPairs);
                tscModel.combineFeatureSets(featureSetType, listOfPairs);
                tscModel.trainClassifier(classifierType, featureSetType, 50);

                splitClassificationsTable.put(alpha,
                                              step,
                                              new HashSet(tscModel.getClassificationsFor(optimizationSplit)));
                numOfCoveredInsts = tscModel.getClassificationsFor(optimizationSplit).size();

                if (numOfCoveredInsts > 0) {
                    if (!totalCovInstsToAWPairsMap.containsKey(numOfCoveredInsts)) {
                        totalCovInstsToAWPairsMap.put(numOfCoveredInsts, new ArrayList<>());
                    }
                    totalCovInstsToAWPairsMap.get(numOfCoveredInsts).add(pair);
                }
            }
        }

        Set<Integer> allOptSplitInsts
                     = new TreeSet<>(IntStream.range(0, tscModel.getRealValuedInstsOf(optimizationSplit).numInstances())
                                              .boxed()
                                              .collect(Collectors.toList()));

        Set<Integer> currentCombo = Sets.newHashSet();

        mapLoop:
        for (Integer numOfCoveredInsts : totalCovInstsToAWPairsMap.keySet()) {
            listOfPairs = totalCovInstsToAWPairsMap.get(numOfCoveredInsts);

            listLoop:
            for (AlphaOmegaPair pair : listOfPairs) {
                if (currentCombo.isEmpty()
                    || !Sets.difference(splitClassificationsTable.get(pair.getAlpha(), pair.getOmega()),
                                        currentCombo)
                            .isEmpty()) {
                    listOfBestPairs.add(pair);
                    Sets.union(currentCombo, splitClassificationsTable.get(pair.getAlpha(),
                                                                           pair.getOmega()))
                        .copyInto(currentCombo);

                    if (Sets.difference(allOptSplitInsts, currentCombo).isEmpty()
                        || (listOfBestPairs.size() == numOfAlphaOmegaPairsToUse)) {
                        break mapLoop;
                    }
                }
            }
        }
        return listOfBestPairs;
    }
    
    protected final List<AlphaOmegaPair> findAWPairsUsingNaiveSC(BasePbTSM tscModel,
                                                           FeatureSetType featureSetType,
                                                           ClassifierType classifierType) throws XGBoostError, Exception {

        AlphaOmegaPair pair;

        List<AlphaOmegaPair> listOfPairs = new ArrayList<>();
        List<AlphaOmegaPair> awPairCorrespondingToClassification = new ArrayList<>();
        
        List<Set<Integer>> classificationsList = new ArrayList<>();

        for (Integer alpha : this.getAlphas()) {
            for (Integer step : this.getOmegas()) {
                pair = new AlphaOmegaPair(alpha, step);
                listOfPairs.clear();
                listOfPairs.add(pair);
                //tscModel.setAlphaOmegaPairs(listOfPairs);
                tscModel.combineFeatureSets(featureSetType, listOfPairs);
                tscModel.trainClassifier(classifierType, featureSetType, 50);
                awPairCorrespondingToClassification.add(pair);
                classificationsList.add(new HashSet(tscModel.getClassificationsFor(optimizationSplit)));
            }
        }

        Set<Integer> totalClassifications
                     = new TreeSet<>(IntStream.range(0, classificationsList.size())
                                              .boxed()
                                              .collect(Collectors.toList()));

        Set<Integer> coveredInstances = new HashSet<>();

        List<List<AlphaOmegaPair>> listOfBestAWPairsLists = new ArrayList<>();

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
                Collections.sort(listOfPairs, AlphaOmegaPair.COMPARATOR_MISTICL);
                listOfBestAWPairsLists.clear();
                listOfBestAWPairsLists.add(listOfPairs);
            } else if (coveredInstances.size() == bestCover) {
                listOfPairs = new ArrayList<>();
                for (Integer ind : currentCombination) {
                    listOfPairs.add(awPairCorrespondingToClassification.get(ind));
                }
                Collections.sort(listOfPairs, AlphaOmegaPair.COMPARATOR_MISTICL);
                listOfBestAWPairsLists.add(listOfPairs);
            }
        }
        Collections.sort(listOfBestAWPairsLists, AW_LIST_COMPARATOR);
        return listOfBestAWPairsLists.get(0);
    }

    /**
     * Load the {@link Properties} from the program directory.
     *
     * @throws IOException
     */
    @Override
    protected void loadBaseProperties() throws NumberFormatException {
        super.loadBaseProperties();

        numOfAlphaOmegaPairsToUse = Integer.parseInt(this.getProperty("pairs_to_use", "4"));
    }

    @Override
    protected void addCliOptions() {
        super.addCliOptions();
        
        LOGGER.debug("Adding additional CLI options.");
        Options opts = new Options();
        // Add the different CLI paramaters to the OPTIONS variable
                        // Temporary directory path
        opts.addOption(Option.builder()
                             .longOpt(L_OPT_EXEC_SLOTS)
                             .hasArg().argName("num-of-threads")
                             .desc("Number of threads to use. [Default: " + DEFAULT_EXEC_SLOTS + "]")
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
                                   + "[Default: " + DEFAULT_PATT_APPROX_METHOD + "]")
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
                                   + "[Default: " + DEFAULT_FEATURE_SET_TYPE + "]")
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
                                   + "[Default: " + DEFAULT_CLASSIFIERS + "]")
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
                                   + "[Default: " + DEFAULT_PARAM_SELECT_METHOD + "]")
                             .build());

        this.addCliOptions(opts);
    }
    
    @Override
    protected void parseCliArguments(String[] args) throws ParseException, NumberFormatException,
                                                                           IllegalArgumentException {
        super.parseCliArguments(args);

        // Get all possible pairs of Alphabet and Window sizes
        defaultPairsList = this.createListOfAlphaOmegaPairs(this.getAlphas(), this.getOmegas());

        // Populate required objects from the parsed CLI arguments
        List<String> optVals;

        // Populate PATT_APPROX_METHOD
        if (this.cliArgsContainOption(L_OPT_PATT_APPROX_METHOD)) {
            optVals = Arrays.asList(this.getParsedCliArgs().getOptionValues(L_OPT_PATT_APPROX_METHOD));
        } else {
            optVals = SPLIT_ON_SPACES.splitToList(DEFAULT_PATT_APPROX_METHOD);
        }
        this.addValuesTo(PATT_APROX_METHODS, PatternApproximationMethod.class, optVals);

        // Populate FEATURE_SET_TYPE_SET
        if (this.cliArgsContainOption(L_OPT_FEATURE_SET_TYPE)) {
            optVals = Arrays.asList(this.getParsedCliArgs().getOptionValues(L_OPT_FEATURE_SET_TYPE));
        } else {
            optVals = SPLIT_ON_SPACES.splitToList(DEFAULT_FEATURE_SET_TYPE);
        }
        this.addValuesTo(FEATURE_SET_TYPES, FeatureSetType.class, optVals);

        // Populate CLASSIFIERS_SET
        if (this.cliArgsContainOption(L_OPT_CLASSIFIERS)) {
            optVals = Arrays.asList(this.getParsedCliArgs().getOptionValues(L_OPT_CLASSIFIERS));
        } else {
            optVals = SPLIT_ON_SPACES.splitToList(DEFAULT_CLASSIFIERS);
        }
        this.addValuesTo(CLASSIFIERS, ClassifierType.class, optVals);

        //Populate PARAM_SELECT_METHOD_SET
        if (this.cliArgsContainOption(L_OPT_PARAM_SELECT_METHOD)) {
            optVals = Arrays.asList(this.getParsedCliArgs().getOptionValues(L_OPT_PARAM_SELECT_METHOD));
        } else {
            optVals = SPLIT_ON_SPACES.splitToList(DEFAULT_PARAM_SELECT_METHOD);
        }
        this.addValuesTo(PARAM_SELECT_METHODS, ParameterSelectionMethod.class, optVals);
    }
}
