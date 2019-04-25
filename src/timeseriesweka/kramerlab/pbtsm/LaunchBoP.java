package timeseriesweka.kramerlab.pbtsm;

import timeseriesweka.classifiers.BagOfPatterns;

public class LaunchBoP extends BaseLauncherSAX {

    public static void main(String[] args) {
        BaseLauncher obj = new LaunchBoP();
        obj.setupAndPerformExperiments(args);
    }

    LaunchBoP() {
        super();
    }

    @Override
    void executeExperiments() throws Exception {
        BagOfPatterns c;

        int seed;
        // For each dataset
        for (String dataset : datasets) {
            // Create a RealValuedDataset object which loads the training and testing splits
            rvDataset = new RealValuedDataset(dataDir, dataset);

            while ((seed = this.getNextSeed()) != -1) {
                c = new BagOfPatterns(setOfAlphabets, setOfWindows);

                this.modelTrainingAndTesting(c, seed);

                if (this.isWarmUpRun()) {
                    this.disableWarmUp();
                    break;
                } else {
                    RESULT_BLDR.delete(0, RESULT_BLDR.length())
                               .append(seed)
                               .append(RESULT_FIELD_SEP)
                               .append(String.format(ACCURACY_FORMAT, trainAcc))
                               .append(RESULT_FIELD_SEP)
                               .append(String.format(ACCURACY_FORMAT, testAcc))
                               .append(RESULT_FIELD_SEP)
                               .append(String.format(RUNTIME_FORMAT, trainTime / 1e3))
                               .append(RESULT_FIELD_SEP)
                               .append(String.format(RUNTIME_FORMAT, testTime / 1e3))
                               .append(System.lineSeparator());

                    this.writeResultsToFile(dataset, "BoP", RESULT_BLDR.toString());
                }
            }
            this.resetSeeds();
        }
    }
}
