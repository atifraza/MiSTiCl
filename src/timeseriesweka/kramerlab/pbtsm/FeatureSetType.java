/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.kramerlab.pbtsm;

/**
 *
 * @author atif
 */
public enum FeatureSetType {
    /**
     * 
     */
    NUM,
    
    /**
     * 
     *
     */
    BIN;
    
    private final String[] featureSetTypeDesc = {"NUM -- Numeric (Euclidean distance based)",
                                                 "BIN -- Boolean (Absence or presence based)"};
    
    @Override
    public String toString() {
        return this.featureSetTypeDesc[this.ordinal()];
    }
}
