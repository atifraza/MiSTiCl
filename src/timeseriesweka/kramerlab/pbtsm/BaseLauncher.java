package timeseriesweka.kramerlab.pbtsm;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.StringJoiner;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;

import weka.core.Instances;

abstract class BaseLauncher {
    
    private static final Logger LOGGER = LoggerFactory.getLogger(BaseLauncher.class);
    
    /**
     * String format to be used for reporting Accuracy results.
     */
    protected final String ACCURACY_FORMAT = "%.3f";

    /**
     * String format to be used for reporting run time results.
     */
    protected final String RUNTIME_FORMAT = "%.4f";

    /**
     * A wrapper object for loading the training and testing splits. Also used to generate different
     * shuffles of the data while maintaining the source split instance distributions.
     */
    protected RealValuedDataset realValuedTSDataset;

    /**
     * {@link StringJoiner} used to format and construct the results of each execution run.
     */
    protected StringJoiner resultJoiner;
    
    protected String ALGO;
    
    private final int DEFAULT_SEED = 0;

    private final String L_OPT_SEED = "seed";
    private final String L_OPT_DATASETS = "datasets";
    private final String L_OPT_RESULTS_DIR = "results-dir";
    private final String L_OPT_FILE_POSTFIX = "filename-postfix";

    /**
     * The directory from which the program is being executed.
     */
    private final String PROGRAM_DIR = System.getProperty("user.dir");

    private final String DEFAULT_RESULTS_DIR = Paths.get(PROGRAM_DIR, "results").toString();
    private final String DEFAULT_FILE_POSTFIX = "";

    /**
     * The command line {@link Options} accepted by the program.
     */
    private final Options VALID_CLI_OPTIONS = new Options();

    private final Properties PROPERTIES = new Properties();

    /**
     * {@link List} of {@link String}s containing the dataset names to be processed.
     */
    private final List<String> DATASETS = new ArrayList<>();

    /**
     * Used for recording the training and testing accuracy of the current run.
     */
    private final Map<SplitType, Double> TRAIN_TEST_ACCURACY = new HashMap<>() {
        {
            put(SplitType.TRAIN, 0.0);
            put(SplitType.TEST, 0.0);
        }
    };
    
    /**
     * Used for recording the training and testing time taken by the current run.
     */
    private final Map<SplitType, Double> TRAIN_TEST_RUNTIME = new HashMap<>() {
        {
            put(SplitType.TRAIN, 0.0);
            put(SplitType.TEST, 0.0);
        }
    };
    
    /**
     * Parsed CLI parameters as {@link org.apache.commons.cli.CommandLine} object.
     */
    private CommandLine parsedCLIArgs;

    /**
     * Variable indicating whether a warm-up run should be executed before processing the actual
     * data set.
     */
    private boolean executeWarmUp;

    /**
     * Starting value for integer seed.
     */
    private int startingSeed;

    /**
     * Ending value for integer seed.
     */
    private int endingSeed;

    /**
     * Current value for integer seed.
     */
    private int currentSeed;

    /**
     * String representation of the path to the data directory.
     */
    private String dataDir;

    /**
     * Path to the results directory.
     */
    private String resultDir;

    /**
     * File name postfix, used to distinguish different experimentation setups.
     */
    private String resultFileNamePostfix;

    private String resultFileHeader = "Iter,Acc-Train,Acc-Test,TotalTime-Train,TotalTime-Test\n";

    BaseLauncher() {}

    /**
     * Setup and execute the experiments.
     *
     * Setup includes loading the properties file and parsing all CLI arguments followed with the
     * call to execute experiments.
     *
     * @param args String array of user provided CLI arguments
     */
    protected final void setupAndPerformExperiments(String[] args) {
        LOGGER.debug("Setting up and starting experiment execution.");
        boolean successful = false;
        try {
            this.loadBaseProperties();

            // Add the common CLI options
            this.addCliOptions();

            this.parseCliArguments(args);

            this.executeExperiments();

            successful = true;
        } catch (Exception e) {
            LOGGER.error("!!! Error !!!\n{}", e.getMessage());
        } finally {
            if (!successful) {
                this.printHelp(this.getClass().getName());
            }
        }
    }

    protected final void writeResultsToFile(String dataset, String method, String result) throws IOException {
        Path path = Paths.get(resultDir, dataset, method + resultFileNamePostfix + ".csv");
        try {
            // create a directory named after the dataset in the results directory
            Files.createDirectories(path.getParent());
        } catch (IOException ioEx) {
            LOGGER.error("Cannot create results directory structure.\n{}", ioEx.getMessage());
            throw ioEx;
        }

        boolean newFileCreated = Files.notExists(path);

        try (BufferedWriter resultsWriter = Files.newBufferedWriter(path,
                                                                    StandardOpenOption.CREATE,
                                                                    StandardOpenOption.APPEND)) {
            if (newFileCreated) {
                resultsWriter.write(resultFileHeader);
            }
            resultsWriter.write(result);
        } catch (IOException e) {
            LOGGER.error("Error writing results file.\n{}", e.getMessage());
            throw e;
        }
    }

    protected final void setResultFileHeader(String header) {
        resultFileHeader = header;
    }

    protected final void modelTrainingAndTesting(Classifier c, int seed) throws Exception {
        Instances[] shuffledSplits = realValuedTSDataset.getShuffledDatasetSplits(seed);
        SplitType currSplit;
        double tic, toc;
        
        tic = System.currentTimeMillis();     // Note start time
        currSplit = SplitType.TRAIN;
        try {
            c.buildClassifier(shuffledSplits[0]);
        } catch (Exception e) {
            LOGGER.error("Error occurred while training classifier.\n{}", e.getMessage());
            throw e;
        }
        try {
            TRAIN_TEST_ACCURACY.put(currSplit, this.getAccuracy(c, shuffledSplits[0]));
        } catch (Exception e) {
            LOGGER.error("Error occurred while classifying training split.\n{}", e.getMessage());
            throw e;
        }
        toc = System.currentTimeMillis();
        TRAIN_TEST_RUNTIME.put(currSplit, (toc - tic)  / 1e3);
        
        tic = System.currentTimeMillis();     // Note start time
        currSplit = SplitType.TEST;
        try {
            TRAIN_TEST_ACCURACY.put(currSplit, this.getAccuracy(c, shuffledSplits[1]));
        } catch (Exception e) {
            LOGGER.error("Error occurred while classifying testing split.\n{}", e.getMessage());
            throw e;
        }
        toc = System.currentTimeMillis();
        TRAIN_TEST_RUNTIME.put(currSplit, (toc - tic)  / 1e3);
    }

    protected final void disableWarmUp() {
        executeWarmUp = false;
    }

    protected final void resetSeed() {
        currentSeed = startingSeed;
    }

    protected final int getNextSeed() {
        return (currentSeed < endingSeed) ? currentSeed++ : -1;
    }

    protected final boolean isWarmUpRun() {
        return executeWarmUp;
    }

    protected final String getDataDirectory() {
        return dataDir;
    }

    protected final String getProgramDirectory() {
        return PROGRAM_DIR;
    }

    protected final String getProperty(String propertyName, String defaultValue) {
        return PROPERTIES.getProperty(propertyName, defaultValue);
    }

    protected final String getAccuracy(SplitType splitType) {
        return String.format(ACCURACY_FORMAT, TRAIN_TEST_ACCURACY.get(splitType));
    }

    protected final String getRuntime(SplitType splitType) {
        return String.format(RUNTIME_FORMAT, TRAIN_TEST_RUNTIME.get(splitType));
    }

    protected final List<String> getDatasets() {
        return DATASETS;
    }

    protected final void addCliOptions(Options cliOptions) {
        cliOptions.getOptions().forEach(opt -> VALID_CLI_OPTIONS.addOption(opt));
    }

    protected void addCliOptions() {
        LOGGER.debug("Adding CLI Options.");
        Options opts = new Options();
        
        opts.addOption(Option.builder() // Dataset names
                             .required()
                             .longOpt(L_OPT_DATASETS)
                             .hasArgs().argName("dataset-name(s)")
                             .desc("[REQUIRED] - Space delimited list of datasets.")
                             .build())
            .addOption(Option.builder() // Dataset shuffle number
                             .longOpt(L_OPT_SEED)
                             .hasArgs().argName("start-seed [end-seed]")
                             .desc("Seed for dataset resampling, classifier seeding, etc. Only "
                                   + "positive integer value(s) accepted. If a pair of seeds is "
                                   + "provided the 'start-seed' should be less than 'end-seed'.\n"
                                   + "[Default: " + DEFAULT_SEED + "]")
                             .build())
            .addOption(Option.builder() // Results directory path
                             .longOpt(L_OPT_RESULTS_DIR)
                             .hasArg().argName("path/to/directory")
                             .desc("An absolute or relative path to the results directory.\n"
                                   + "[Default: " + DEFAULT_RESULTS_DIR+ "]")
                             .build())
            .addOption(Option.builder() // Results file postfix
                             .longOpt(L_OPT_FILE_POSTFIX)
                             .hasArg().argName("csv-file-postfix")
                             .desc("An optional postfix to distinguish between result "
                                   + "files using different experimental protocol.")
                             .build());

        this.addCliOptions(opts);
    }

    /**
     * Load the {@link Properties} from the program directory.
     *
     * @throws IOException
     */
    protected void loadBaseProperties() throws NumberFormatException {
        LOGGER.debug("Loading base properties from properties file.");
        try {
            // Create a Path object pointing to the properties file in the program directory
            Path path = Paths.get(this.getProgramDirectory(), "base.properties");
            PROPERTIES.load(Files.newBufferedReader(path));
        } catch (IOException ioException) {
            LOGGER.warn("Properties file can not be read. The program will continue with defaults.");
        }

        // Get the data directory or assume it is in 'data' subdirectory in the program directory
        dataDir = this.getProperty("data_dir", Paths.get(this.getProgramDirectory(), "data").toString());

        // Load the warm up execution flag; default is false
        executeWarmUp = Boolean.parseBoolean(this.getProperty("execute_warm_up", "false"));
    }

    /**
     * Parses the user provided CLI arguments.
     *
     * @param args String array of user provided CLI arguments
     *
     * @throws IllegalArgumentException
     * @throws IOException
     * @throws NumberFormatException
     * @throws ParseException
     */
    protected void parseCliArguments(String[] args) throws ParseException, NumberFormatException,
                                                                           IllegalArgumentException {
        // Create a CLI argument parser
        CommandLineParser cliParser = new DefaultParser();
        try {
            // Parse the CLI parameters provided by the user
            parsedCLIArgs = cliParser.parse(VALID_CLI_OPTIONS, args);

            // Create a set of strings and save the provided DATASETS in it to
            // discard any duplicate enteries.
            Set<String> setOfDatasetNames = new HashSet<>(Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_DATASETS)));

            // If a warm-up run is required, add a small sized dataset to the list of DATASETS before
            // adding the user provided dataset names.
            if (executeWarmUp) {
                DATASETS.add("Coffee");
            }
            DATASETS.addAll(setOfDatasetNames);   // Add all given dataset names

            // Get the results directory
            resultDir = this.getResultsDirectory();

            // Get the file name postfix
            resultFileNamePostfix = this.getResultFilePostfix();

            // Extract seed value(s) and check if they follow the requirements
            String[] opts = parsedCLIArgs.getOptionValues(L_OPT_SEED);

            // If no seed values provided
            if (Objects.isNull(opts)) {
                // Set the start seed to default
                startingSeed = DEFAULT_SEED;
            } else {
                // else, use first value as start
                startingSeed = Integer.parseInt(opts[0]);
            }

            // If two seed values are provided
            if (Objects.nonNull(opts) && opts.length == 2) {
                // set the end seed
                endingSeed = Integer.parseInt(opts[1]);
            } else {
                // else set start+1 as end seed
                endingSeed = startingSeed + 1;
            }

            // If seeds are negative, end < start or more than 2 values provided
            if (startingSeed < 0 || endingSeed < 0 || endingSeed <= startingSeed
                || (Objects.nonNull(opts) && opts.length > 2)) {
                // throw the exception
                LOGGER.error("Invalid seed value(s) provided.");
                throw new IllegalArgumentException();
            } else {
                // else, everything is fine, set the current seed to start seed
                this.resetSeed();
            }
        } catch (ParseException parseException) {
            LOGGER.error("Error parsing the provided CLI arguments.\n{}\nException message: {}",
                         args, parseException.getMessage());
            throw parseException;
        } catch (NumberFormatException numFmtEx) {
            LOGGER.error("Parsing the provided seed values failed. Check the format of seeds.\n{}",
                         numFmtEx.getMessage());
            throw numFmtEx;
        }
    }

    protected boolean cliArgsContainOption(String longOptionString) {
        return this.getParsedCliArgs().hasOption(longOptionString);
    }

    protected CommandLine getParsedCliArgs() {
        return parsedCLIArgs;
    }
    
    protected abstract void executeExperiments() throws IOException, Exception;

    private void printHelp(String className) {
        String call = "java -cp lib/:JAR_FILENAME.jar " + className + " ";
        PrintWriter writer = new PrintWriter(System.out);
        HelpFormatter helpFormatter = new HelpFormatter();
        helpFormatter.printHelp(writer, 150, call, "", VALID_CLI_OPTIONS, 5, 1, "", true);
        writer.flush();
    }

    /**
     * Get a {@link String} filename postfix.
     * 
     * @return String to append to each result file to identify experiments
     */
    private String getResultFilePostfix() {
        // IF a postfix is provided, then return it with an "_" prepended
        if (parsedCLIArgs.hasOption(L_OPT_FILE_POSTFIX)) {
            return "_" + parsedCLIArgs.getOptionValue(L_OPT_FILE_POSTFIX);
        }
        // else return the default postfix
        return DEFAULT_FILE_POSTFIX;
    }

    /**
     * Get the {@link String} representation of the path to results directory
     *
     * @return String representation of absolute path of results directory
     *
     * @throws IOException
     */
    private String getResultsDirectory() {
        return Paths.get(parsedCLIArgs.getOptionValue(L_OPT_RESULTS_DIR, DEFAULT_RESULTS_DIR)).toString();
    }

    private double getAccuracy(Classifier c, Instances instances) throws Exception {
        int correctlyClassified = 0;
        int instInd = 0;
        double actual, predicted;
        try {
            for (instInd = 0; instInd < instances.numInstances(); instInd++) {
                actual = instances.get(instInd).classValue();
                predicted = c.classifyInstance(instances.get(instInd));
                if ((int) actual == (int) predicted) {
                    correctlyClassified++;
                }
            }
        } catch (Exception e) {
            LOGGER.error("Error classifying instance. Instance Ind: {}, Class: {}",
                         instances.get(instInd),
                         instances.get(instInd).stringValue(instances.get(instInd).classIndex()));
            throw e;
        }
        return 100.0 * correctlyClassified / instances.numInstances();
    }
}
