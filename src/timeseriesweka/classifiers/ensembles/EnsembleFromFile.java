package timeseriesweka.classifiers.ensembles;

import utilities.ClassifierResults;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import utilities.DebugPrinting;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instances;

/**
 * Given a results directory and classifiers to use (the results for which lie in that directory),
 * will ensemble those classifier's classifications to form a collective classification
 * 
 * By default, will use each member's train accuracy as it's vote weighting, and simple 
 * majority voting to form an ensemble classification
 * 
 * Unless a specific name for the ensemble is provided, will attempt to construct a name
 * by concatenating the names of its constituents. ensembleIdentifier is (as of writing) 
 * only used for saving the cv accuracy of the ensemble (for SaveCVAccuracy)
 * 
 * @author James Large
 */
public abstract class EnsembleFromFile extends AbstractClassifier implements DebugPrinting {
    
    //results file reading/writing
    protected boolean readIndividualsResults = false;
    protected boolean writeIndividualsResults = false;
    
    protected boolean resultsFilesParametersInitialised;
    protected String resultsFilesDirectory;
    protected String ensembleIdentifier = "EnsembleFromFile";
    protected int resampleIdentifier;

    protected String datasetName;

    /**
     * must be called in order to build ensemble from results files or to write individual's 
     * results files
     * 
     * exitOnFilesNotFound defines whether the ensemble will simply throw exception/exit if results files 
     * arnt found, or will try to carry on (e.g train the classifiers normally)
     */
    public void setResultsFileLocationParameters(String individualResultsFilesDirectory, String datasetName, int resampleIdentifier) {
        resultsFilesParametersInitialised = true;
        
        this.resultsFilesDirectory = individualResultsFilesDirectory;
        this.datasetName = datasetName;
        this.resampleIdentifier = resampleIdentifier;
    }
    
    public void setBuildIndividualsFromResultsFiles(boolean b) {
        readIndividualsResults = b;
        if (b)
            writeIndividualsResults = false;
    }
    
    public void setWriteIndividualsResultsFiles(boolean b) {
        writeIndividualsResults = b;
        if (b)
            readIndividualsResults = false;
    }
    
    public File findResultsFile(String classifierName, String trainOrTest) {
        File file = new File(resultsFilesDirectory+classifierName+"/Predictions/"+datasetName+"/"+trainOrTest+"Fold"+resampleIdentifier+".csv");
        if(!file.exists() || file.length() == 0)
            return null;
        else 
            return file;
    }
    
    public static ClassifierResults loadResultsFile(File file, int numClasses) throws Exception {    
                
        ArrayList<Double> alpreds = new ArrayList<>();
        ArrayList<Double> alclassvals = new ArrayList<>();
        ArrayList<ArrayList<Double>> aldists = new ArrayList<>();
        
        Scanner scan = new Scanner(file);
        scan.useDelimiter("\n");
        String nameLine = scan.next();
        String paramLine = scan.next();
                       
        //gets the buildTime from the params line
        long buildTime = -1;
        if (paramLine!=null && !paramLine.isEmpty()) {
            String[] paras = paramLine.split(",");
            if (paras[0].equalsIgnoreCase("BuildTime")) 
                buildTime = Long.parseLong(paras[1].trim());
        }        
        
        double acc = Double.parseDouble(scan.next().trim());
        
        boolean fileHasPreds = false;
        
        String [] lineParts = null;
        while(scan.hasNext()){
            lineParts = scan.next().split(",");

            if (lineParts == null || lineParts.length < 2) //empty lines
                continue;
            
            fileHasPreds = true;
            
            alclassvals.add(Double.parseDouble(lineParts[0].trim()));
            alpreds.add(Double.parseDouble(lineParts[1].trim()));
            
            if (lineParts.length > 3) {//dist for inst is present
                ArrayList<Double> dist = new ArrayList<>();
                for (int i = 3; i < lineParts.length; ++i)  //act,pred,[empty],firstclassprob.... therefore 3 start
                    if (lineParts[i] != null && !lineParts[i].equals("")) //may have an extra comma on the end...
                        dist.add(Double.parseDouble(lineParts[i].trim()));
                aldists.add(dist);
            }
        }
        
        scan.close();
        
//        validateResultsFile(file, acc, alclassvals, alpreds, aldists, numClasses); //todo maybe expand/include
        
        double [] classVals = null;
        if (!alclassvals.isEmpty()) {
            classVals = new double[alclassvals.size()];
            for (int i = 0; i < alclassvals.size(); ++i)
                classVals[i]= alclassvals.get(i);
        }
    
        double [] preds = null;
        if (!alpreds.isEmpty()) {
            preds = new double[alpreds.size()];
            for (int i = 0; i < alpreds.size(); ++i)
                preds[i]= alpreds.get(i);
        }
        
        double[][] distsForInsts = null;
        if (!aldists.isEmpty()) {
            distsForInsts = new double[aldists.size()][aldists.get(0).size()];
            for (int i = 0; i < aldists.size(); ++i)
                for (int j = 0; j < aldists.get(i).size(); ++j) 
                    distsForInsts[i][j] = aldists.get(i).get(j);
        }
        
        ClassifierResults results = null;
        if (fileHasPreds)
            results = new ClassifierResults(acc, classVals, preds, distsForInsts, numClasses);
        else 
            results = new ClassifierResults(acc, numClasses);
        //now need to account for fact that some files might only have the first 3 lines and 
        //not the full train cv preds
        //blame tony.
        
        results.buildTime = buildTime;
        results.setName(nameLine);
        results.setParas(paramLine);
        
        return results;
    }
    
    /**
     * todo current validations are pretty off the hoof/incomplete
     * 
     * expand/make more thorough, and add some flag to only perform validation if e.g debug = true
     */
    private static void validateResultsFile(File file, double acc, ArrayList<Double> alclassvals, ArrayList<Double> alpreds, ArrayList<ArrayList<Double>> aldists, int numClasses) throws Exception {
        if (!aldists.isEmpty()) {
            if (alpreds.size() != aldists.size())
                throw new Exception("validateResultsFile Exception: "
                        + "somehow have different number of predictions and distForInstances: " + file.getAbsolutePath());

            for (ArrayList<Double> dist : aldists)
                if (dist.size() != numClasses)
                    throw new Exception("validateResultsFile Exception: "
                            + "instance reports different numbers of classes: " + file.getAbsolutePath());
        }
        
        double count = 0.0;
        for (int i = 0; i < alpreds.size(); ++i)
            if (alpreds.get(i).equals(alclassvals.get(i)))
                count++;
        
        double a = count/alpreds.size();
        if (a != acc)
            throw new Exception("validateResultsFile Exception: "
                    + "incorrect accuracy (" + acc + "reported vs" +a +"actual) reported in: " + file.getAbsolutePath());
    }
    
    protected void writeResultsFile(String classifierName, String parameters, ClassifierResults results, String trainOrTest) throws IOException {                
        StringBuilder st = new StringBuilder();
        st.append(this.datasetName).append(",").append(this.ensembleIdentifier).append(classifierName).append(","+trainOrTest+"\n");
        st.append(parameters + "\n"); //st.append("internalHesca\n");
        st.append(results.acc).append("\n");
        
        double[] trueClassVals=results.getTrueClassVals();
        double[] predClassVals=results.getPredClassVals();
         if (predClassVals != null) {
            for(int i = 0; i < predClassVals.length;i++) {
                st.append(trueClassVals[i]).append(",").append(predClassVals[i]).append(","); //pred
                       double[] distForInst=results.getDistributionForInstance(i);

                if (distForInst != null)
                    for (int j = 0; j < distForInst.length; j++)
                        st.append("," + distForInst[j]);

                st.append("\n");
            }
        }
        
        String fullPath = this.resultsFilesDirectory+classifierName+"/Predictions/"+datasetName;
        new File(fullPath).mkdirs();
        FileWriter out = new FileWriter(fullPath+"/" + trainOrTest + "Fold"+this.resampleIdentifier+".csv");
        out.append(st);
        out.close();
    }

    
    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
