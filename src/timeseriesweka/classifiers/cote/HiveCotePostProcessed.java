package timeseriesweka.classifiers.cote;

import java.util.ArrayList;
import timeseriesweka.classifiers.HiveCote;
import static utilities.ClassifierTools.loadData;
import weka.core.Instances;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class HiveCotePostProcessed extends AbstractPostProcessedCote{
    
    {
        HiveCotePostProcessed.CLASSIFIER_NAME = "HIVE-COTE";
    }
    public HiveCotePostProcessed(String resultsDir, String datasetName, int resampleId, ArrayList<String> classifierNames) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifierNames = classifierNames;
    }
    
    public HiveCotePostProcessed(String resultsDir, String datasetName, ArrayList<String> classifierNames) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = 0;
        this.classifierNames = classifierNames;
    }
    
    public HiveCotePostProcessed(String resultsDir, String datasetName, int resampleId) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifierNames = getDefaultClassifierNames();
    }
    
    public HiveCotePostProcessed(String resultsDir, String datasetName) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = 0;
        this.classifierNames = getDefaultClassifierNames();
    }

    private ArrayList<String> getDefaultClassifierNames(){
        ArrayList<String> names = new ArrayList<>();
        names.add("EE");
        names.add("ST");
        names.add("RISE");
        names.add("BOSS");
        names.add("TSF");
        return names;
    }
    
    @Override
    public double[] distributionForInstance(int testInstanceId) throws Exception{
        if(this.testDists==null){
            throw new Exception("Error: classifier not initialised correctly. Load results before classifiying.");
        }
        
        int numClasses = this.testDists[0][0].length;
        double[] outDist = new double[numClasses];
        double cvAccSum = 0;
        
        for(int classifier = 0; classifier < testDists.length; classifier++){
            for(int classVal = 0; classVal < numClasses; classVal++){
                outDist[classVal]+= testDists[classifier][testInstanceId][classVal]*this.cvAccs[classifier];
            }
            cvAccSum+=this.cvAccs[classifier];
        }
        
        for(int classVal = 0; classVal < numClasses; classVal++){
            outDist[classVal]/= cvAccSum;
        }
        
        return outDist;
    }
    
    public static void main(String[] args) throws Exception{
        String datasetName = "ItalyPowerDemand";
        Instances train = loadData("C:/users/sjx07ngu/dropbox/tsc problems/"+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = loadData("C:/users/sjx07ngu/dropbox/tsc problems/"+datasetName+"/"+datasetName+"_TEST");      
        /*
            Step 1: build Hive and write to file`
        */
//        
//        HiveCote hc = new HiveCote();
//        hc.makeShouty();
//        hc.turnOnFileWriting("hiveWritingProto/", datasetName);
//        hc.buildClassifier(train);
//        hc.writeTestPredictionsToFile(test, "hiveWritingProto/", datasetName);
        
        /*
            Step 2: read from file and (hhopefully) recreate the same results
        */
        
        HiveCotePostProcessed hcpp = new HiveCotePostProcessed("hiveWritingProto/", datasetName);
        hcpp.writeTestSheet("hiveWritingProtoRewrite/");
    }

}
