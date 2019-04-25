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
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import weka.classifiers.Classifier;

import weka.core.Instances;

abstract class BaseLauncher {
    
    /**
     * The directory from which the program is being executed.
     */
    final String PROGRAM_DIR = System.getProperty("user.dir");

    /**
     * String format to be used for reporting Accuracy results.
     */
    final String ACCURACY_FORMAT = "%.3f";

    /**
     * String format to be used for reporting Runtime results.
     */
    final String RUNTIME_FORMAT = "%.4f";

    /**
     * Separator to use between result columns.
     */
    final String RESULT_FIELD_SEP = ",";

    /**
     * {@link StringBuilder} object used to format and construct the results for each execution run.
     */
    final StringBuilder RESULT_BLDR;
    
    /**
     * Used for recording the training accuracy of the current run.
     */
    double trainAcc;

    /**
     * Used for recording the testing accuracy of the current run.
     */
    double testAcc;

    /**
     * Used for recording the training time taken by the current run.
     */
    double trainTime;

    /**
     * Used for recording the testing time taken by the current run.
     */
    double testTime;

    String RESULTS_FILE_HEADER = "Iter,Acc-Train,Acc-Test,TotalTime-Train,TotalTime-Test\n";

    /**
     * String representation of the path to the data directory.
     */
    String dataDir;

    /**
     * {@link List} of {@link String}s containing the dataset names to be processed.
     */
    List<String> datasets;

    /**
     * A wrapper object for loading the training and testing splits. Also used to generate different
     * shuffles of the data while maintaining the source split instance distributions.
     */
    RealValuedDataset rvDataset;

    /**
     * Parsed CLI parameters as {@link org.apache.commons.cli.CommandLine} object.
     */
    CommandLine parsedCLIArgs;

    Properties props;

    private final int DEFAULT_SEED = 0;

    private final String L_OPT_SEED = "seed";

    private final String L_OPT_DATASETS = "datasets";

    private final String L_OPT_RESULTS_DIR = "results-dir";

    private final String L_OPT_FILE_POSTFIX = "file-postfix";

    private final String DEFAULT_RESULTS_DIR = Paths.get(PROGRAM_DIR, "results").toString();

    private final String DEFAULT_FILE_POSTFIX = "";

    /**
     * The command line {@link Options} accepted by the program.
     */
    private final Options VALID_CLI_OPTIONS;

    /**
     * Variable indicating whether a warm-up run should be executed before processing the actual
     * data set.
     */
    private boolean executeWarmUp;

    /**
     * Path to the results directory.
     */
    private String resultDir;

    /**
     * File name postfix, used to distinguish different experimentation setups.
     */
    private String filenamePostfix;

    /**
     * Starting value for integer seed.
     */
    private int startSeed;

    /**
     * Ending value for integer seed.
     */
    private int endSeed;

    /**
     * Current value for integer seed.
     */
    private int currentSeed;

    BaseLauncher() {
        RESULT_BLDR = new StringBuilder(200);
        VALID_CLI_OPTIONS = new Options();
        datasets = new ArrayList<>();
        props = new Properties();
    }

    /**
     * Setup and execute the experiments.
     *
     * Setup includes loading the properties file and parsing all CLI arguments followed with the
     * call to execute experiments.
     *
     * @param args String array of user provided CLI arguments
     */
    void setupAndPerformExperiments(String[] args) {
        boolean successful = false;
        try {
            this.loadBaseProperties();

            this.parseCliArguments(args);

            this.executeExperiments();

            successful = true;
        } catch (ParseException e) {
            System.err.println("!!! Error parsing command line arguments. !!!\n" + e.getMessage());
        } catch (NumberFormatException e) {
            System.err.println("!!!Error parsing number. !!!\n" + e.getMessage());
        } catch (IllegalArgumentException e) {
            System.err.println("!!! Unsupported argument provided. !!!\n" + e.getMessage());
        } catch (IOException e) {
            System.err.println("!!! Error creating/opening file/directory. !!!\n" + e.getMessage());
        } catch (Exception e) {
            System.err.println("!!! Error !!!\n" + e.getMessage());
        } finally {
            if (!successful) {
                this.printHelp(this.getClass().getName());
            }
        }
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
    void parseCliArguments(String[] args) throws IllegalArgumentException, IOException,
                                                 NumberFormatException, ParseException {
        // Add the common CLI options
        this.addCommonCLIOptions();

        // Create a CLI argument parser
        CommandLineParser cliParser = new DefaultParser();

        // Parse the CLI parameters provided by the user
        parsedCLIArgs = cliParser.parse(VALID_CLI_OPTIONS, args);

        // Create a set of strings and save the provided datasets in it to
        // discard any duplicate enteries.
        Set<String> setOfDatasetNames
                    = new HashSet<>(Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_DATASETS)));

        // If a warm-up run is required, add a small sized dataset to the list of datasets before
        // adding the user provided dataset names.
        if (executeWarmUp) {
            datasets.add("Coffee");
        }
        datasets.addAll(setOfDatasetNames);   // Add all given dataset names

        // Get the results directory
        resultDir = this.getResultsDir();

        // Get the file name postfix
        filenamePostfix = this.getResultFilePostfix();

        // Extract seed value(s) and check if they follow the requirements
        String[] opts = parsedCLIArgs.getOptionValues(L_OPT_SEED);

        // If no seed values provided
        if (opts == null) {
            // Set the start seed to default
            startSeed = DEFAULT_SEED;
        } else {
            // else, use first value as start
            startSeed = Integer.parseInt(opts[0]);
        }

        // If two seed values are provided
        if (opts != null && opts.length == 2) {
            // set the end seed
            endSeed = Integer.parseInt(opts[1]);
        } else {
            // else set start+1 as end seed
            endSeed = startSeed + 1;
        }

        // If seeds are negative, end < start or more than 2 values provided
        if (startSeed < 0
            || endSeed < 0
            || endSeed <= startSeed
            || (opts != null && opts.length > 2)) {
            // throw the exception
            throw new IllegalArgumentException("Invalid seed value(s).");
        } else {
            // else, everything is fine, set the current seed to start seed
            this.resetSeeds();
        }
    }

    /**
     * Load the {@link Properties} from the program directory.
     *
     * @throws IOException
     */
    void loadBaseProperties() throws IOException {
        // Create a Path object pointing to the properties file in the program directory
        Path path = Paths.get(PROGRAM_DIR, "base.properties");

        // Load the file (if it exists)
        if (Files.exists(path)) {
            this.props.load(Files.newBufferedReader(path));
        }

        // Get the data directory or assume it is in 'data' subdirectory in the program directory
        dataDir = this.props.getProperty("data_dir", Paths.get(PROGRAM_DIR, "data").toString());

        // Load the warm up execution flag; default is false
        executeWarmUp = Boolean.parseBoolean(this.props.getProperty("execute_warm_up", "false"));
    }

    void resetSeeds() {
        currentSeed = startSeed;
    }

    int getNextSeed() {
        return (currentSeed < endSeed) ? currentSeed++ : -1;
    }

    void modelTrainingAndTesting(Classifier c, int seed) throws Exception {
        Instances[] shuffledSplits = rvDataset.getShuffledDataset(seed);

        trainTime = System.currentTimeMillis();     // Note start time
        c.buildClassifier(shuffledSplits[0]);
        trainAcc = this.getAccuracy(c, shuffledSplits[0]);
        // Subtract the current time from the noted time to get elapsed time
        trainTime = System.currentTimeMillis() - trainTime;

        testTime = System.currentTimeMillis();      // Note start time
        testAcc = this.getAccuracy(c, shuffledSplits[1]);
        // Subtract the current time from the noted time to get elapsed time
        testTime = System.currentTimeMillis() - testTime;
    }

    void writeResultsToFile(String dataset, String method, String result) throws IOException {

        Path path = Paths.get(resultDir, dataset, method + filenamePostfix + ".csv");

        // create a directory named after the dataset in the results directory
        Files.createDirectories(path.getParent());

        boolean newFileCreated = Files.notExists(path);

        try (BufferedWriter bwResults = Files.newBufferedWriter(path,
                                                                StandardOpenOption.CREATE,
                                                                StandardOpenOption.APPEND)) {
            if (newFileCreated) {
                bwResults.write(RESULTS_FILE_HEADER);
            }
            bwResults.write(result);
        } catch (IOException e) {
            throw new IOException("Error writing results.", e);
        }
    }

    boolean isWarmUpRun() {
        return executeWarmUp;
    }

    void disableWarmUp() {
        executeWarmUp = false;
    }

    void printHelp(String className) {
        String call = "java -cp lib/:JAR_FILENAME.jar " + className + " ";
        PrintWriter writer = new PrintWriter(System.out);
        HelpFormatter helpFormatter = new HelpFormatter();
        helpFormatter.printHelp(writer, 150, call, "", VALID_CLI_OPTIONS, 5, 1, "", true);
        writer.flush();
    }

    void addAdditionalCLIOptions(Options opts) {
        opts.getOptions().forEach(opt -> VALID_CLI_OPTIONS.addOption(opt));
    }

    abstract void executeExperiments() throws Exception;

    /**
     * Add the CLI {@link Options}, common to the entire program.
     */
    private void addCommonCLIOptions() {
        VALID_CLI_OPTIONS.addOption(Option.builder() // Dataset names
                                          .required()
                                          .longOpt(L_OPT_DATASETS)
                                          .hasArgs().argName("dataset-name(s)")
                                          .desc("[REQUIRED] - Space delimited list of datasets.")
                                          .build())
                         .addOption(Option.builder() // Dataset shuffle number
                                          .longOpt(L_OPT_SEED)
                                          .hasArgs().argName("start-seed [end-seed]")
                                          .desc("Seed for dataset resampling, classifier seeding, "
                                                + "etc. Seed value(s) can only be positive "
                                                + "integers. If a pair of seeds is provided the "
                                                + "'start-seed' should be less than 'end-seed'.\n"
                                                + "[Default: " + DEFAULT_SEED + "]")
                                          .build())
                         .addOption(Option.builder() // Results directory path
                                          .longOpt(L_OPT_RESULTS_DIR)
                                          .hasArg().argName("path/to/directory")
                                          .desc("An absolute or relative path to the results "
                                                + "directory.\n[Default: " + DEFAULT_RESULTS_DIR
                                                + "]")
                                          .build())
                         .addOption(Option.builder() // Results file postfix
                                          .longOpt(L_OPT_FILE_POSTFIX)
                                          .hasArg().argName("csv-file-postfix")
                                          .desc("An optional postfix to distinguish between result "
                                                + "files using different experimental protocol.")
                                          .build());
    }

    /**
     * Get the {@link String} representation of the path to results directory
     *
     * @return String representation of absolute path of results directory
     *
     * @throws IOException
     */
    private String getResultsDir() throws IOException {
        return Paths.get(parsedCLIArgs.getOptionValue(L_OPT_RESULTS_DIR, DEFAULT_RESULTS_DIR))
                    .toString();
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

    private double getAccuracy(Classifier c, Instances in) throws Exception {
        int count = 0;
        double actual, predicted;
        for (int i = 0; i < in.numInstances(); i++) {
            actual = in.get(i).classValue();
            predicted = c.classifyInstance(in.get(i));
            if ((int) actual == (int) predicted) {
                count++;
            }
        }
        return 100.0 * count / in.numInstances();
    }
}
