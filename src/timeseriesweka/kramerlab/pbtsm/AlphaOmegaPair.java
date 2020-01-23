package timeseriesweka.kramerlab.pbtsm;

import java.util.Comparator;

public class AlphaOmegaPair {
    
    public static Comparator<AlphaOmegaPair> PAIR_COMPARATOR = Comparator.comparingInt(AlphaOmegaPair::getAlpha)
                                                                         .thenComparing(AlphaOmegaPair::getOmega,
                                                                                        Comparator.reverseOrder());
    
    public static Comparator<AlphaOmegaPair> COMPARATOR_MISTICL = Comparator.comparingInt(AlphaOmegaPair::getAlpha)
                                                                            .thenComparing(AlphaOmegaPair::getOmega);
    
    private final int ALPHA;
    private final int OMEGA;
    private final String STR_FORM;
    
    /**
     * Construct an AlphaWindowPair object.
     * 
     * @param alpha Alphabet size for cardinality reduction
     * @param omega Window size for dimensionality reduction
     */
    public AlphaOmegaPair(int alpha, int omega) {
        this.ALPHA = alpha;
        this.OMEGA = omega;
        this.STR_FORM = "(A: " + this.ALPHA + " W: " + this.OMEGA + ")";
    }
    
    public int getAlpha() {
        return this.ALPHA;
    }
    
    public int getOmega() {
        return this.OMEGA;
    }
    
    @Override
    public String toString() {
        return this.STR_FORM;
    }
}
