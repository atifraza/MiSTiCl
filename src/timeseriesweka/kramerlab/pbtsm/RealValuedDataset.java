package timeseriesweka.kramerlab.pbtsm;

import java.io.IOException;

import java.nio.file.Paths;

import utilities.ClassifierTools;
import utilities.InstanceTools;

import weka.core.Instances;

public class RealValuedDataset {

    private final String DATASET_NAME;

    private final Instances TRAIN_SET;
    private final Instances TEST_SET;

    /**
     * Construct a Real-valued dataset object containing a training and test set in ARFF format.
     *
     * The data directory should contain a directory named after the dataset. The dataset directory
     * should have the training and testing splits as TRAIN.arff and TEST.arff files, respectively.
     *
     * @param pathToDataDir Path to the directory containing the ARFF data files
     * @param datasetName Name of dataset to be loaded
     * @throws IOException
     */
    public RealValuedDataset(String pathToDataDir, String datasetName) throws IOException {
        this.DATASET_NAME = datasetName;

        try {
            TRAIN_SET = ClassifierTools.loadData(Paths.get(pathToDataDir, datasetName, "TRAIN.arff")
                                                      .toFile());

            TEST_SET = ClassifierTools.loadData(Paths.get(pathToDataDir, datasetName, "TEST.arff")
                                                     .toFile());
        } catch (IOException e) {
            throw new IOException("Error loading dataset file(s).\n\n"
                                  + "Make sure files are located as per the following structure:\n"
                                  + "pathToDataDir\n"
                                  + "    |\n"
                                  + "    |-datasetName\n"
                                  + "          |\n"
                                  + "          |-TRAIN.arff\n"
                                  + "          |-TEST.arff\n", e);
        }
    }

    /**
     * Get the name of the dataset.
     *
     * @return Name of the dataset
     */
    public String getDatasetName() {
        return this.DATASET_NAME;
    }

    /**
     * Get shuffled dataset splits using the given seed.
     *
     * @param seed An integer value
     *
     * @return An array of {@link Instances} containing the shuffled training and testing splits
     * respectively
     */
    public Instances[] getShuffledDataset(int seed) {
        return InstanceTools.resampleTrainAndTestInstances(this.TRAIN_SET, this.TEST_SET, seed);
    }
}
