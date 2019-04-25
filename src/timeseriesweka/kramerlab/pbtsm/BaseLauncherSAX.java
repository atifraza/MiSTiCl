package timeseriesweka.kramerlab.pbtsm;

import com.google.common.base.Splitter;

import java.io.IOException;
import java.util.ArrayList;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

abstract class BaseLauncherSAX extends BaseLauncher {

    String DEF_ALPHAS = "";
    String DEF_WINDOWS = "";

    Set<Integer> setOfAlphabets;
    Set<Integer> setOfWindows;

    private final String L_OPT_ALPHAS = "alphas";
    private final String L_OPT_WINDOWS = "windows";

    BaseLauncherSAX() {
        super();
        setOfAlphabets = new HashSet<>();
        setOfWindows = new HashSet<>();
    }

    @Override
    void parseCliArguments(String[] args) throws IllegalArgumentException, IOException,
                                                 NumberFormatException, ParseException {
        Options opts = new Options();
        // Add the different CLI paramaters to the OPTIONS variable
        opts.addOption(Option.builder()          // Set of numbers for alphabets to be used
                             .longOpt(L_OPT_ALPHAS)
                             .hasArgs().argName("a1 [a2 a3 ...]")
                             .desc("Space delimited set of alphabet-sizes. Valid range: [2, 10].")
                             .build())
            .addOption(Option.builder()          // Set of numbers for window sizes to be used
                             .longOpt(L_OPT_WINDOWS)
                             .hasArgs().argName("w1 [w2 w3 ...]")
                             .desc("Space delimited set of window-sizes. Valid range: [1, 10].")
                             .build());

        this.addAdditionalCLIOptions(opts);

        super.parseCliArguments(args);

        // Populate required objects from the parsed CLI arguments
        List<String> optVals;
        
        Splitter splitter = Splitter.on(" ").omitEmptyStrings().trimResults();

        // Populate setOfAlphabets
        if (parsedCLIArgs.hasOption(L_OPT_ALPHAS)) {
            optVals = Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_ALPHAS));
        } else {
            optVals = splitter.splitToList(this.DEF_ALPHAS);
        }
        this.addValuesTo(setOfAlphabets, 2, 10, optVals);

        // Populate setOfWindows
        if (parsedCLIArgs.hasOption(L_OPT_WINDOWS)) {
            optVals = Arrays.asList(parsedCLIArgs.getOptionValues(L_OPT_WINDOWS));
        } else {
            optVals = splitter.splitToList(this.DEF_WINDOWS);
        }
        this.addValuesTo(setOfWindows, 1, 10, optVals);
    }

    void addValuesTo(Set<Integer> destSet, int minAllowed, int maxAllowed, List<String> values) {
        int temp;
        for (String optVal : values) {
            temp = Integer.parseInt(optVal);
            if (minAllowed <= temp && temp <= maxAllowed) {
                destSet.add(temp);
            } else {
                throw new IllegalArgumentException("Values cannot be smaler than " + minAllowed
                                                   + " or greater than " + maxAllowed + ".");
            }
        }
    }

    /**
     *
     * @param alphabets
     * @param windowSizes
     * @return
     */
    List<AlphaWindowPair> generateListAlphaWindowPair(Set<Integer> alphabets,
                                                      Set<Integer> windowSizes) {
        List<AlphaWindowPair> result = new ArrayList<>();
        alphabets.stream().forEach((alpha) -> {
            windowSizes.stream().forEach((windowSize) -> {
                result.add(new AlphaWindowPair(alpha, windowSize));
            });
        });
        return result;
    }
}
