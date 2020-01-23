package timeseriesweka.kramerlab.pbtsm;

import com.google.common.base.Splitter;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

abstract class BaseLauncherSAX extends BaseLauncher {
    
    private static final Logger LOGGER = LoggerFactory.getLogger(BaseLauncherSAX.class);

    protected final Splitter SPLIT_ON_SPACES = Splitter.on(" ").omitEmptyStrings().trimResults();

    protected String defaultAlphas = "";
    protected String defaultOmegas = "";

    private final int minAlpha = 2;
    private final int maxAlpha = 10;
    
    private final int minOmega = 1;
    private final int maxOmega = 10;
    
    private final String L_OPT_ALPHAS = "alphas";
    private final String L_OPT_OMEGAS = "omegas";
    
    private final Set<Integer> setOfAlphas = new HashSet<>();
    private final Set<Integer> setOfOmegas = new HashSet<>();

    BaseLauncherSAX() {
        super();
    }

    protected final void addValuesTo(Set<Integer> destinationSet, List<String> listOfValues, int minAllowed,
                                     int maxAllowed) throws NumberFormatException, IllegalArgumentException {
        int convertedValue;
        try {
            for (String value : listOfValues) {
                convertedValue = Integer.parseInt(value);
                if (minAllowed <= convertedValue && convertedValue <= maxAllowed) {
                    destinationSet.add(convertedValue);
                } else {
                    LOGGER.error("Invalid range for alphas/omegas.");
                    throw new IllegalArgumentException();
                }
            }
        } catch (NumberFormatException nfe) {
            LOGGER.error("A provided numeric argument is ill-formatted. Please verify.");
            throw nfe;
        }
    }
    
    protected final Set<Integer> getAlphas() {
        return setOfAlphas;
    }
    
    protected final Set<Integer> getOmegas() {
        return setOfOmegas;
    }
    
    @Override
    protected void addCliOptions() {
        super.addCliOptions();
        
        LOGGER.debug("Adding additional CLI options.");
        Options opts = new Options();
        // Add the different CLI paramaters to the OPTIONS variable
        opts.addOption(Option.builder()          // Set of numbers for alphabets to be used
                             .longOpt(L_OPT_ALPHAS)
                             .hasArgs().argName("a1 [a2 a3 ...]")
                             .desc("Space delimited set of alphabet-sizes. Valid range: ["
                                   + minAlpha + ", " + maxAlpha + "].")
                             .build())
            .addOption(Option.builder()          // Set of numbers for window sizes to be used
                             .longOpt(L_OPT_OMEGAS)
                             .hasArgs().argName("w1 [w2 w3 ...]")
                             .desc("Space delimited set of window-sizes. Valid range: ["
                                   + minOmega + ", " + maxOmega + "].")
                             .build());

        this.addCliOptions(opts);
    }

    @Override
    protected void parseCliArguments(String[] args) throws ParseException, NumberFormatException,
                                                                           IllegalArgumentException {
        super.parseCliArguments(args);

        // Populate required objects from the parsed CLI arguments
        List<String> optVals;
        
        // Populate setOfAlphas
        if (this.cliArgsContainOption(L_OPT_ALPHAS)) {
            optVals = Arrays.asList(this.getParsedCliArgs().getOptionValues(L_OPT_ALPHAS));
        } else {
            optVals = SPLIT_ON_SPACES.splitToList(defaultAlphas);
        }
        this.addValuesTo(setOfAlphas, optVals, minAlpha, maxAlpha);

        // Populate setOfOmegas
        if (this.cliArgsContainOption(L_OPT_OMEGAS)) {
            optVals = Arrays.asList(this.getParsedCliArgs().getOptionValues(L_OPT_OMEGAS));
        } else {
            optVals = SPLIT_ON_SPACES.splitToList(defaultOmegas);
        }
        this.addValuesTo(setOfOmegas, optVals, minOmega, maxOmega);
    }
}
