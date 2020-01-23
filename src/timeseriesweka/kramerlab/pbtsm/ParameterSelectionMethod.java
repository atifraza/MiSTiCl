/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.kramerlab.pbtsm;

/**
 * BRF: Brute force
 * HSC: Heuristic/Greedy Set cover
 * NOP: No optimization
 * NSC: Naive Set cover
 * 
 * @author atif
 */
public enum ParameterSelectionMethod {
    /**
     * No parameter optimization, combine all feature sets to form one giant one
     */
    NOP,
    
    /**
     * Use a naive set cover approach to optimize the parameters and
     * combine corresponding feature sets to form the final feature set
     */
    NSC,
    
    /**
     * Use heuristic/greedy set cover approach to optimize the parameters and
     * combine corresponding feature sets to form the final feature set
     */
    HSC,
    
    /**
     * Use brute force approach to optimize the parameters and
     * combine corresponding feature sets to form the final feature set
     */
    BRF;
    
    private final String[] selectionMethodDesc = {"NOP -- No parameter optimization performed, concatenate all created feature sets",
                                                  "NSC -- Use a set cover approach with brute force method, try all possible combinations of feature sets",
                                                  "HSC -- Use a greedy/heuristic set cover approach to combine a few feture sets",
                                                  "BRF -- Use a brute force approach to get the best combination of feature sets"};
    
    @Override
    public String toString() {
        return this.selectionMethodDesc[this.ordinal()];
    }
}
