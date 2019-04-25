
package vector_classifiers;

import weka.classifiers.functions.PLSClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * Built for my (James') alcohol datasets, to allow comparative testing between TSC approaches 
 * and the de-facto chemometrics approach, Partial Least Squares regression
 * 
 * Extends the weka PLSClassifier, and essentially just converts the nominal class valued
 * dataset passed (initial intention being the ifr non-invasive whiskey datasets)
 * and does the standard regression, before converting the output back into a discrete class value
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class PLSNominalClassifier extends PLSClassifier {

    protected Attribute classAttribute;
    protected double[] numericClassVals;
    protected int classind;
    
    public PLSNominalClassifier() {
        super();
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        Instances train = new Instances(data);
        
        classind = train.classIndex();
        classAttribute = train.classAttribute();
        numericClassVals = new double[classAttribute.numValues()];
              
        for (int i = 0; i < numericClassVals.length; i++)
            numericClassVals[i] = convertNominalToNumeric(classAttribute.value(i));
        
        FastVector<Attribute> atts = new FastVector<>(train.numAttributes());
        for (int i = 0; i < train.numAttributes(); i++) {
            if (i != classind)
                atts.add(train.attribute(i));
            else {
                //class attribute
                Attribute numericClassAtt = new Attribute(train.attribute(i).name());
                atts.add(numericClassAtt);
            }
        }
        
        Instances temp = new Instances(train.relationName(), atts, train.numInstances());
        temp.setClassIndex(classind);
        
        for (int i = 0; i < train.numInstances(); i++) {
            temp.add(new DenseInstance(1.0, train.instance(i).toDoubleArray()));
            temp.instance(i).setClassValue(numericClassVals[(int)train.instance(i).classValue()]);
        }
        
        train = temp;
        
        //datset is in the proper format, now do the model fitting as normal
        super.buildClassifier(train);
    }
    
    protected double convertNominalToNumeric(String strClassVal) {
        return Double.parseDouble(strClassVal.replaceAll("[A-Za-z ]", ""));
    }
    
    public double regressInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return utilities.GenericTools.indexOfMax(distributionForInstance(instance));
    }
    
    public double[] distributionForInstance(Instance instance) throws Exception {
        double regpred = super.classifyInstance(instance);
        
        double[] dist = new double[numericClassVals.length];
        
        if (regpred < numericClassVals[0])
            dist[0] = 1.0;
        else if (regpred > numericClassVals[numericClassVals.length-1])
            dist[dist.length-1] = 1.0;
        else {
            for (int i = 1; i < numericClassVals.length; i++) {
                if (regpred < numericClassVals[i]) {
                    double end = numericClassVals[i] - numericClassVals[i-1];
                    double t = regpred - numericClassVals[i-1];
                    double propToRight = t / end;
                    
                    dist[i] = propToRight;
                    dist[i-1] = 1-propToRight;
                }    
            }
        }
        
        return dist;
    }
    
}
