/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.kramerlab.pbtsm;

/**
 *
 * CS: Add patterns based on Chi-Square test of independence
 * IG: Add patterns based on Information Gain value
 * NF: No independence testing but use filtering for removing variant patterns
 * NO: No independence testing, naively add all patterns
 *
 * @author atif
 */
public enum PatternIndependenceTest {
    /**
     * Use the Chi-square test for pattern independence
     */
    CS,
    
    /**
     * Use the information gain for pattern independence
     */
    IG,
    
    /**
     * No independence test is performed but patterns are filtered to remove variants
     */
    NF,
    
    /**
     * No independence test or filtering is performed
     */
    NO;

    private final String[] testDesc = {"CS -- Chi-Square test based pattern selection with subsequent variant removal",
                                       "IG -- Information gain based pattern selection with subsequent variant removal",
                                       "NF -- Occurrence frequency based pattern selection with subsequent variant removal",
                                       "NO -- Occurrence frequency based pattern selection without variant removal"};

    @Override
    public String toString() {
        return this.testDesc[this.ordinal()];
    }
}
