package timeseriesweka.kramerlab.pbtsm;

import java.io.IOException;

import java.nio.file.Paths;

import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import utilities.ClassifierTools;
import utilities.InstanceTools;

import weka.core.Instances;

public class RealValuedDataset {
    
    private static final Logger LOGGER = LoggerFactory.getLogger(RealValuedDataset.class);

    private final String DATASET_NAME;

    private final Map<SplitType, Instances> DATASET_SPLITS = new HashMap<>();

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
        Instances split;
        try {
            split = ClassifierTools.loadData(Paths.get(pathToDataDir, datasetName,
                                                       "TRAIN.arff").toFile());
            DATASET_SPLITS.put(SplitType.TRAIN, split);
            
            split = ClassifierTools.loadData(Paths.get(pathToDataDir, datasetName,
                                                       "TEST.arff").toFile());
            DATASET_SPLITS.put(SplitType.TEST, split);
        } catch (IOException e) {
            LOGGER.error("Error loading dataset file(s). Make sure files are located as per the following structure:\n"
                         + "path/to/data-dir/\n"
                         + "                |-dataset-name/\n"
                         + "                              |-TRAIN.arff\n"
                         + "                              |-TEST.arff\n\n{}",
                         e.getMessage());
            throw e;
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
    public Instances[] getShuffledDatasetSplits(int seed) {
        return InstanceTools.resampleTrainAndTestInstances(DATASET_SPLITS.get(SplitType.TRAIN),
                                                           DATASET_SPLITS.get(SplitType.TEST),
                                                           seed);
    }
}
