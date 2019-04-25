/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.kramerlab.pbtsm.misticl;

/**
 *
 * @author atif
 */
public enum ClassifierType {
    /**
     * 
     */
    ET,
    /**
     * 
     */
    RF,
    /**
     * 
     */
    AB,
    /**
     * 
     */
    XG;

    private final String[] classifierTypeDesc = {"ET -- Extremely Randomized Trees",
                                                 "RF -- Random Forests",
                                                 "AB -- AdaBoostM1 (with Decision Stumps)",
                                                 "XG -- XGBoost"};
    
    @Override
    public String toString() {
        return this.classifierTypeDesc[this.ordinal()];
    }
}
