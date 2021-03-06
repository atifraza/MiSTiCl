package timeseriesweka.kramerlab.pbtsm;

import java.io.IOException;

import java.util.StringJoiner;

import timeseriesweka.classifiers.BagOfPatterns;

public class LaunchBoP extends BaseLauncherSAX {

    public static void main(String[] args) {
        // --datasets ItalyPowerDemand --seed 0 2 --alphas 3 4 5 --omegas 2 3 4
        BaseLauncher obj = new LaunchBoP();
        obj.setupAndPerformExperiments(args);
    }

    LaunchBoP() {
        super();
        ALGO = "BoP";
    }

    @Override
    protected void executeExperiments() throws IOException, Exception {
        int seed;
        // For each dataset
        for (String dataset : this.getDatasets()) {
            // Create a RealValuedDataset object which loads the training and testing splits
            realValuedTSDataset = new RealValuedDataset(this.getDataDirectory(), dataset);

            while ((seed = this.getNextSeed()) != -1) {
                BagOfPatterns c = new BagOfPatterns(this.getAlphas(), this.getOmegas());
                this.modelTrainingAndTesting(c, seed);
                if (this.isWarmUpRun()) {
                    this.disableWarmUp();
                    break;
                } else {
                    resultJoiner = new StringJoiner(",", "", "\n");
                    resultJoiner.add(Integer.toString(seed))
                                .add(this.getAccuracy(SplitType.TRAIN))
                                .add(this.getAccuracy(SplitType.TEST))
                                .add(this.getRuntime(SplitType.TRAIN))
                                .add(this.getRuntime(SplitType.TEST));

                    this.writeResultsToFile(dataset, ALGO, resultJoiner.toString());
                }
            }
            this.resetSeed();
        }
    }
}
