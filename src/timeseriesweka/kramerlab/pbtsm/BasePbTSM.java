package timeseriesweka.kramerlab.pbtsm;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.collect.Tables;

import java.io.IOException;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import java.util.function.Consumer;

import timeseriesweka.filters.SAX;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public abstract class BasePbTSM {

    /**
     * Name of data set being processed.
     */
    private final String datasetName;

    /**
     * Map the "Class Values" in the data set to the corresponding Instance counts.
     */
    private final Map<Double, Integer> clsValsToInstCountsMap;

    /**
     * Map the {@link SplitType} to real valued Instances.
     */
    private final Map<SplitType, Instances> splitTypeToRVInstsMap;

    /**
     * Map of {@link SplitType} to {@link com.google.common.collect.Table} of SAXed Instances. The
     * Table object is keyed with Alpha (rows) and Window (column) values.
     */
    private final Map<SplitType, Table<Integer, Integer, Instances>> splitTypeToSymInstsMap;

    /**
     * Map of {@link SplitType} to {@link com.google.common.collect.Table} of Transformation Time
     * values. The Table object is keyed with Alpha (rows) and Window (column) values.
     */
    private final Map<SplitType, Table<Integer, Integer, Long>> splitTypeToTransformationTimesTable;

    /**
     * List of {@link AlphaWindowPair} objects. Used to create discrete versions of the data.
     */
    private List<AlphaWindowPair> alphaWindowPairs;

    /**
     * Number of execution threads.
     */
    private final int numExecutionThreads;

    public BasePbTSM(String datasetName, Instances[] instanceSplits, int numThreads) {

        this.datasetName = datasetName;

        // Initialize thread count variable for controlling level of parallelization
        this.numExecutionThreads = numThreads;

        // Initialize the map for tracking all class labels and their corresponding instance counts
        this.clsValsToInstCountsMap = new HashMap<>();

        // Initialize map for training/testing Instances objects
        this.splitTypeToRVInstsMap = new EnumMap<>(SplitType.class);

        this.setDataSplits(instanceSplits);

        // Initialize map for symbolized training/testing instances
        this.splitTypeToSymInstsMap = new EnumMap<>(SplitType.class);

        // Initialize map for saving transformation times for each SplitType
        // Transformation time for each Alpha Window pair is saved in the Table
        this.splitTypeToTransformationTimesTable = new EnumMap<>(SplitType.class);

        for (SplitType splitType : SplitType.values()) {
            // Place a Table object for each SplitType
            this.splitTypeToSymInstsMap.put(splitType,
                                            Tables.synchronizedTable(HashBasedTable.create()));

            // Place a Table Object for each SplitType
            // The table saved the transformation times for each Alpha/Window value pair
            this.splitTypeToTransformationTimesTable.put(splitType,
                                                         Tables.synchronizedTable(HashBasedTable.create()));
        }
    }

    protected final String getDatasetName() {
        return this.datasetName;
    }

    protected final int getNumExecutionThreads() {
        return numExecutionThreads;
    }

    protected final int getRVInstLength() {
        return (this.splitTypeToRVInstsMap.get(SplitType.TRAIN).numAttributes() - 1);
    }

    protected final List<AlphaWindowPair> getAlphaWindowPairs() {
        return alphaWindowPairs;
    }

    protected final Map<Double, Integer> getClsValsToInstCountsMap() {
        return clsValsToInstCountsMap;
    }

    protected final Map<SplitType, Instances> getSplitTypeToRVInstsMap() {
        return splitTypeToRVInstsMap;
    }

    protected final Map<SplitType, Table<Integer, Integer, Instances>> getSplitTypeToSymInstsMap() {
        return splitTypeToSymInstsMap;
    }

    protected final Map<SplitType, Table<Integer, Integer, Long>> getSplitTypeToTransformationTimesTable() {
        return splitTypeToTransformationTimesTable;
    }

    /**
     *
     * @param pairs
     */
    protected final void setAlphaWindowPairs(List<AlphaWindowPair> pairs) {
        this.alphaWindowPairs = pairs;
    }

    protected void transformData(AlphaWindowPair pair) throws Exception {
        int alpha = pair.getNumAlphabets();
        int window = pair.getWindowSize();

        SAX saxer = new SAX();
        saxer.useRealValuedAttributes(false);
        saxer.setAlphabetSize(alpha);
        saxer.setNumIntervals(this.getRVInstLength() / window);

        Instances realValuedSplit;
        Instances saxedSplit;

        Table<Integer, Integer, Instances> tbl_SAXedInstances;
        Table<Integer, Integer, Long> tbl_TransformationTime;

        long timer;

        try {
            for (SplitType splitType : SplitType.values()) {
                realValuedSplit = this.splitTypeToRVInstsMap.get(splitType);
                tbl_SAXedInstances = this.splitTypeToSymInstsMap.get(splitType);
                tbl_TransformationTime = this.splitTypeToTransformationTimesTable.get(splitType);

                timer = System.currentTimeMillis();

                saxedSplit = saxer.process(realValuedSplit);
                timer = System.currentTimeMillis() - timer;

                tbl_SAXedInstances.put(alpha, window, this.getSymbolizedForm(saxedSplit));

                tbl_TransformationTime.put(alpha, window, timer);
            }
        } catch (Exception e) {
            System.err.println("Exception occured while transforming data to SAX");
            System.err.format("Alpha: %d, Step: %d%n" + alpha, window);
            System.err.println(e);
        }
    }

    abstract protected void performPreprocessing() throws IOException;

    @FunctionalInterface
    protected interface ThrowingConsumer<T, E extends Exception> {

        void accept(T t) throws E;
    }

    protected <T> Consumer<T> throwingConsumerWrapper(ThrowingConsumer<T, Exception> throwingConsumer) {
        return i -> {
            try {
                throwingConsumer.accept(i);
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        };
    }

    private Instances getSymbolizedForm(Instances saxedSplit) {
        int totalInstances = saxedSplit.numInstances();
        int totalAttributes = saxedSplit.numAttributes() - 1;

        Attribute stringAttribute = new Attribute("symbolic-instance", true);
        Attribute targetAttribute = this.getSplitTypeToRVInstsMap().get(SplitType.TRAIN).classAttribute();

        ArrayList<Attribute> listOfAttributes = new ArrayList<Attribute>() {
            {
                add(stringAttribute);
                add(targetAttribute);
            }
        };

        Instances symbolicDataset = new Instances("", listOfAttributes, 0);
        symbolicDataset.setClassIndex(listOfAttributes.size() - 1);

        StringBuilder currSymbolizedInstance = new StringBuilder(totalAttributes * 2);
        Instance currSAXedInstance;
        Instance newDestInst;
        for (int instanceIndex = 0; instanceIndex < totalInstances; instanceIndex++) {
            currSAXedInstance = saxedSplit.get(instanceIndex);
            currSymbolizedInstance.setLength(0);

            for (int attributeIndex = 0; attributeIndex < totalAttributes; attributeIndex++) {
                currSymbolizedInstance.append(currSAXedInstance.stringValue(attributeIndex));
            }

            newDestInst = new DenseInstance(symbolicDataset.numAttributes());
            newDestInst.setDataset(symbolicDataset);
            newDestInst.setValue(0, currSymbolizedInstance.toString());
            newDestInst.setClassValue(currSAXedInstance.classValue());
            symbolicDataset.add(newDestInst);
        }
        return symbolicDataset;
    }

    /**
     *
     * @param insts
     */
    private void setDataSplits(Instances[] insts) {
        this.splitTypeToRVInstsMap.put(SplitType.TRAIN, insts[0]);
        this.splitTypeToRVInstsMap.put(SplitType.TEST, insts[1]);

        Map<Double, Integer> destMap = this.clsValsToInstCountsMap;
        destMap.clear();

        Instances trainSplit = this.splitTypeToRVInstsMap.get(SplitType.TRAIN);

        // Enumerate all training set instances and their counts
        trainSplit.forEach(in -> destMap.compute(in.classValue(), (k, v) -> v == null ? 1 : v + 1));

    }
}
