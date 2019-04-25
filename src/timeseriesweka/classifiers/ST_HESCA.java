/*
Shaplet transform with the weighted ensemble
 */
package timeseriesweka.classifiers;

import timeseriesweka.filters.shapelet_transforms.ShapeletTransformFactory;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransform;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformFactoryOptions;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformTimingUtilities;
import java.io.File;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.AbstractClassifier;
import vector_classifiers.HESCA;
import weka.core.Instance;
import weka.core.Instances;
import static timeseriesweka.filters.shapelet_transforms.ShapeletTransformTimingUtilities.nanoToOp;
import timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQuality;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
import timeseriesweka.classifiers.cote.HiveCoteModule;
import utilities.ClassifierResults;
import utilities.TrainAccuracyEstimate;

/**
 *
 * @author raj09hxu
 */
public class ST_HESCA  extends AbstractClassifier implements HiveCoteModule, SaveParameterInfo, TrainAccuracyEstimate{

    public enum ST_TimeLimit {MINUTE, HOUR, DAY};

    //Minimum number of instances per class in the train set
    public static final int minimumRepresentation = 25;
    
    private boolean preferShortShapelets = false;
    private String shapeletOutputPath;
    
    private HESCA hesca;
    private ShapeletTransform transform;
    private Instances format;
    int[] redundantFeatures;
    private boolean doTransform=true;
    
    
    private SearchType searchType = SearchType.IMP_RANDOM;
    
    private long numShapelets = 0;
    private long seed = 0;
    private long timeLimit = Long.MAX_VALUE;
    
    public void setSeed(long sd){
        seed = sd;
    }
    
    //careful when setting search type as you could set a type that violates the contract.
    public void setSearchType(ShapeletSearch.SearchType type) {
        searchType = type;
    }

    @Override
    public void writeCVTrainToFile(String train) {
        hesca.writeCVTrainToFile(train);
    }

    /*//if you want HESCA to perform CV.
    public void setPerformCV(boolean b) {
        hesca.setPerformCV(b);
    }*/
    
    @Override
    public ClassifierResults getTrainResults() {
        return  hesca.getTrainResults();
    }
        
    @Override
    public String getParameters(){
        String paras=transform.getParameters();
        String ensemble=hesca.getParameters();
        return paras+",timeLimit"+timeLimit+","+ensemble;
    }
    
    @Override
    public double getEnsembleCvAcc() {
        return hesca.getEnsembleCvAcc();
    }

    @Override
    public double[] getEnsembleCvPreds() {
        return hesca.getEnsembleCvPreds();
    }
    
    public void doSTransform(boolean b){
        doTransform=b;
    }
    
    public long getTransformOpCount(){
        return transform.getCount();
    }
    
    
    public Instances transformDataset(Instances data){
        if(transform.isFirstBatchDone())
            return transform.process(data);
        return null;
    }
    
    //set any value in nanoseconds you like.
    public void setTimeLimit(long time){
        timeLimit = time;
    }
    
    //pass in an enum of hour, minut, day, and the amount of them. 
    public void setTimeLimit(ST_TimeLimit time, int amount){
        //min,hour,day in longs.
        long[] times = {ShapeletTransformTimingUtilities.dayNano/24/60, ShapeletTransformTimingUtilities.dayNano/24, ShapeletTransformTimingUtilities.dayNano};
        
        timeLimit = times[time.ordinal()] * amount;
    }
    
    public void setOneDayLimit(){
        setTimeLimit(ST_TimeLimit.DAY, 1);
    }
    
    public void setOneHourLimit(){
        setTimeLimit(ST_TimeLimit.HOUR, 1);
    }
    
    public void setOneMinuteLimit(){
        setTimeLimit(ST_TimeLimit.MINUTE, 1);
    }
    
    public void setNumberOfShapelets(long numS){
        numShapelets = numS;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        format = doTransform ? createTransformData(data, timeLimit) : data;
        
        hesca=new HESCA();
        hesca.setRandSeed((int) seed);
                
        redundantFeatures=InstanceTools.removeRedundantTrainAttributes(format);

        hesca.buildClassifier(format);
        format=new Instances(data,0);
    }
    
     @Override
    public double classifyInstance(Instance ins) throws Exception{
        format.add(ins);
        
        Instances temp  = doTransform ? transform.process(format) : format;
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
        
        Instance test  = temp.get(0);
        format.remove(0);
        return hesca.classifyInstance(test);
    }
     @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
        format.add(ins);
        
        Instances temp  = doTransform ? transform.process(format) : format;
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
        
        Instance test  = temp.get(0);
        format.remove(0);
        return hesca.distributionForInstance(test);
    }
    
    public void setShapeletOutputFilePath(String path){
        shapeletOutputPath = path;
    }
    
    public void preferShortShapelets(){
        preferShortShapelets = true;
    }

    public Instances createTransformData(Instances train, long time){
        int n = train.numInstances();
        int m = train.numAttributes()-1;

        //construct the options for the transform.
        ShapeletTransformFactoryOptions.Builder optionsBuilder = new ShapeletTransformFactoryOptions.Builder();
        optionsBuilder.setDistanceType(SubSeqDistance.DistanceType.IMP_ONLINE);
        optionsBuilder.setQualityMeasure(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN);
        if(train.numClasses() > 2){
            optionsBuilder.useBinaryClassValue();
            optionsBuilder.useClassBalancing();
        }
        optionsBuilder.useRoundRobin();
        optionsBuilder.useCandidatePruning();
        
        //create our search options.
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);

        //clamp K to 2000.
        int K = n > 2000 ? 2000 : n;   
        
        //how much time do we have vs. how long our algorithm will take.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        if(opCount.compareTo(opCountTarget) == 1){
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            
            //if we've not set a shapelet count, calculate one, based on the time set.
            if(numShapelets == 0){
                numShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
                numShapelets *= prop.doubleValue();
            }
             
             //we need to find atleast one shapelet in every series.
            searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(searchType);
            searchBuilder.setNumShapelets(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        optionsBuilder.setKShapelets(K);
        optionsBuilder.setSearchOptions(searchBuilder.build());
        transform = new ShapeletTransformFactory(optionsBuilder.build()).getTransform();
        transform.supressOutput();
        
        if(shapeletOutputPath != null)
            transform.setLogOutputFile(shapeletOutputPath);
        
        if(preferShortShapelets)
            transform.setShapeletComparator(new Shapelet.ShortOrder());
        
        return transform.process(train);
    }
    
    public static void main(String[] args) throws Exception {
        String dataLocation = "C:\\LocalData\\Dropbox\\TSC Problems\\";
        //String dataLocation = "..\\..\\resampled transforms\\BalancedClassShapeletTransform\\";
        String saveLocation = "..\\..\\resampled results\\RefinedRandomTransform\\";
        String datasetName = "Earthquakes";
        int fold = 0;
        
        Instances train= ClassifierTools.loadData(dataLocation+datasetName+File.separator+datasetName+"_TRAIN");
        Instances test= ClassifierTools.loadData(dataLocation+datasetName+File.separator+datasetName+"_TEST");
        String trainS= saveLocation+datasetName+File.separator+"TrainCV.csv";
        String testS=saveLocation+datasetName+File.separator+"TestPreds.csv";
        String preds=saveLocation+datasetName;

        ST_HESCA st= new ST_HESCA();
        //st.saveResults(trainS, testS);
        st.doSTransform(true);
        st.setOneMinuteLimit();
        st.buildClassifier(train);
        double accuracy = utilities.ClassifierTools.accuracy(test, st);
        
        System.out.println("accuracy: " + accuracy);
    }    
}
