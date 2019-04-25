/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class MultivariateInstanceTools {
    
    
    //given some univariate datastreams, we want to merge them to be interweaved.
    //so given dataset X, Y, Z.
    //X_0,Y_0,Z_0,X_1,.....,Z_m

    //Needs more testing.
    public static Instances mergeStreams(String dataset, Instances[] inst, String[] dimChars){
        
        String name;
        
        Instances firstInst = inst[0];
        int dimensions = inst.length;
        int length = (firstInst.numAttributes()-1)*dimensions;

        FastVector atts = new FastVector();
        for (int i = 0; i < length; i++) {
            name = dataset + "_" + dimChars[i%dimensions] + "_" + (i/dimensions);
            atts.addElement(new Attribute(name));
        }
        
        //clone the class values over. 
        //Could be from x,y,z doesn't matter.
        Attribute target = firstInst.attribute(firstInst.classIndex());
        FastVector vals = new FastVector(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.addElement(target.value(i));
        }
        atts.addElement(new Attribute(firstInst.attribute(firstInst.classIndex()).name(), vals));
        
        //same number of xInstances 
        Instances result = new Instances(dataset, atts, firstInst.numInstances());

        int size = result.numAttributes()-1;
        
        for(int i=0; i< firstInst.numInstances(); i++){
            result.add(new DenseInstance(size+1));
            
            for(int j=0; j<size;){
                for(int k=0; k< dimensions; k++){
                    result.instance(i).setValue(j,inst[k].get(i).value(j/dimensions)); j++;
                }
            }
        }
        
        for (int j = 0; j < result.numInstances(); j++) {
            result.instance(j).setValue(size, firstInst.get(j).classValue());
        }
        
        return result;
    }
    
    
    private static Instances createRelationFrom(Instances header, double[][] data){
        int numAttsInChannel = data[0].length;
        Instances output = new Instances(header, data.length);

        //each dense instance is row/ which is actually a channel.
        for(int i=0; i< data.length; i++){
            output.add(new DenseInstance(numAttsInChannel));
            for(int j=0; j<numAttsInChannel; j++)
                output.instance(i).setValue(j, data[i][j]);
        }
        
        return output;
    }
    
    private static Instances createRelationHeader(int numAttsInChannel, int numChannels){
        //construct relational attribute vector.
        FastVector relational_atts = new FastVector(numAttsInChannel);
        for (int i = 0; i < numAttsInChannel; i++) {
            relational_atts.addElement(new Attribute("att" + i));
        }
        
        return new Instances("", relational_atts, numChannels);
    }
    
    public static Instances mergeToMultivariateInstances(Instances[] instances){
        //given a set of seperate channels, can we merge them back into a relation object.
        
        Instance firstInst = instances[0].firstInstance();
        int numAttsInChannel = instances[0].numAttributes()-1;
        

        FastVector attributes = new FastVector();
        
        //construct relational attribute.#
        Instances relationHeader = createRelationHeader(numAttsInChannel, instances.length);
        relationHeader.setRelationName("relationalAtt");
        Attribute relational_att = new Attribute("relationalAtt", relationHeader, numAttsInChannel);        
        attributes.addElement(relational_att);
        
        //clone the class values over. 
        //Could be from x,y,z doesn't matter.
        Attribute target = firstInst.attribute(firstInst.classIndex());
        FastVector vals = new FastVector(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.addElement(target.value(i));
        }
        attributes.addElement(new Attribute(firstInst.attribute(firstInst.classIndex()).name(), vals));
        
        Instances output = new Instances("", attributes, instances[0].numInstances());
        
        for(int i=0; i < instances[0].numInstances(); i++){
            //create each row.
            //only two attribtues, relational and class.
            output.add(new DenseInstance(2));
            
            double[][] data = new double[instances.length][numAttsInChannel];
            for(int j=0; j<instances.length; j++)
                for(int k=0; k<numAttsInChannel; k++)
                    data[j][k] = instances[j].get(i).value(k);
            
            //set relation for the dataset/
            Instances relational = createRelationFrom(relationHeader, data);
            
            int index = output.instance(i).attribute(0).addRelation(relational);
            output.instance(i).setValue(0, index);           
            
            //set class value.
            output.instance(i).setValue(1, instances[0].get(i).classValue());
        }
        
        output.setClassIndex(output.numAttributes()-1);
        //System.out.println(relational);
        return output; 
    }
    
    //function which returns the seperate channels of a multivariate problem as Instances[].
    public static Instances[] splitMultivariateInstances(Instances multiInstances){
        Instances[] output = new Instances[numChannels(multiInstances)];
        
        int length = channelLength(multiInstances); //all the values + a class value.

        //each channel we want to build an Instances object which contains the data, and the class attribute.
        for(int i=0; i< output.length; i++){
            //construct numeric attributes
            FastVector atts = new FastVector();
            for (int att = 0; att < length; att++) {
                atts.addElement(new Attribute("channel_"+i+"_"+att));
            }
            
            //construct the class values atttribute.
            Attribute target = multiInstances.attribute(multiInstances.classIndex());
            FastVector vals = new FastVector(target.numValues());
            for (int k = 0; k < target.numValues(); k++) {
                vals.addElement(target.value(k));
            }
            atts.addElement(new Attribute(multiInstances.attribute(multiInstances.classIndex()).name(), vals));
            
            output[i] = new Instances(multiInstances.relationName() + "_channel_" + i, atts, multiInstances.numInstances());
            output[i].setClassIndex(length);
            
            //for each Instance in 
            for(int j =0; j< multiInstances.numInstances(); j++){
                
                //add the denseinstance to write too.
                output[i].add(new DenseInstance(length+1));

                //System.out.println(index);
                double [] channel = multiInstances.get(j).relationalValue(0).get(i).toDoubleArray();
                int k=0;
                for(; k<channel.length; k++){
                    output[i].instance(j).setValue(k, channel[k]);
                }
                
                double classVal = multiInstances.get(j).classValue();
                output[i].instance(j).setValue(k, classVal);
            }
        }
        
        return output;
    }
    
    public static Instances[] resampleMultivariateInstances(Instances dataset, long seed, double prop){
        Instances[] data_channels = splitMultivariateInstances(dataset);
        Instances[] resample_train_channels = new Instances[data_channels.length];
        Instances[] resample_test_channels = new Instances[data_channels.length];
        
       for (int i = 0; i < resample_train_channels.length; i++) {
            Instances[] temp = utilities.InstanceTools.resampleInstances(data_channels[i], seed, prop);
            resample_train_channels[i] = temp[0];
            resample_test_channels[i] = temp[1];
        }
        
        Instances[] output = new Instances[2];
        output[0] = mergeToMultivariateInstances(resample_train_channels);
        output[1] = mergeToMultivariateInstances(resample_test_channels);
       
        return output;
    }

        
    public static Instances[] resampleMultivariateTrainAndTestInstances(Instances train, Instances test, long seed){
        Instances[] train_channels = splitMultivariateInstances(train);
        Instances[] test_channels = splitMultivariateInstances(test);
        
        Instances[] resample_train_channels = new Instances[train_channels.length];
        Instances[] resample_test_channels = new Instances[test_channels.length];
        
        for (int i = 0; i < resample_train_channels.length; i++) {
            Instances[] temp = utilities.InstanceTools.resampleTrainAndTestInstances(train_channels[i], test_channels[i], seed);
            resample_train_channels[i] = temp[0];
            resample_test_channels[i] = temp[1];
        }
        
        Instances[] output = new Instances[2];
        output[0] = mergeToMultivariateInstances(resample_train_channels);
        output[1] = mergeToMultivariateInstances(resample_test_channels);
        
        return output;
    }
    
    
    public static Instance[] splitMultivariateInstance(Instance instance){
        Instance[] output = new Instance[numChannels(instance)];
        for(int i=0; i< output.length; i++){
            output[i] = instance.relationalValue(0).get(i);
        }    
        return output;
    }
    
    
    //this won't include class value.    
    public static double[][] convertMultiInstanceToArrays(Instance[] data){
        double[][] output = new double[data.length][data[0].numAttributes()];
        for(int i=0; i<output.length; i++){
            for(int j=0; j<output[i].length; j++){
                output[i][j] = data[i].value(j);
            }
        }
        return output;
    }
    
    //this won't include class value.
    public static double[][] convertMultiInstanceToTransposedArrays(Instance[] data){ 
        double[][] output = new double[data[0].numAttributes()][data.length];
        for(int i=0; i<output.length; i++){
            for(int j=0; j<output[i].length; j++){
                output[i][j] = data[j].value(i);
            }
        }
        
        return output;
    }
    
    public static int indexOfRelational(Instances inst, Instances findRelation){
        int index = -1;
        Attribute relationAtt = inst.get(0).attribute(0);
        for(int i=0; i< inst.numInstances(); i++){
            
            if(relationAtt.relation(i).equals(findRelation)){
                index  = i;
                break;
            }
        }
        return index;
    }
    
    public static int numChannels(Instance multiInstance){
        return multiInstance.relationalValue(0).numInstances();
    }
    
    public static int channelLength(Instance multiInstance){
        return multiInstance.relationalValue(0).numAttributes();
    }
    
    public static int numChannels(Instances multiInstances){
        //get the first attribute which we know is 
        return numChannels(multiInstances.firstInstance());
    }
    
    public static int channelLength(Instances multiInstances){
        return channelLength(multiInstances.firstInstance());
    }
    
}
