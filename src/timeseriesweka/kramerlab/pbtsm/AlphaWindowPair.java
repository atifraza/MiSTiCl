package timeseriesweka.kramerlab.pbtsm;

public class AlphaWindowPair {
    
    private final int SIZE_ALPHA;
    private final int SIZE_WINDOW;
    private final String STR_FORM;
    
    /**
     * Construct an AlphaWindowPair object.
     * 
     * @param numAlphabets Alphabet size for cardinality reduction
     * @param windowSize Window size for dimensionality reduction
     */
    public AlphaWindowPair(int numAlphabets, int windowSize) {
        this.SIZE_ALPHA = numAlphabets;
        this.SIZE_WINDOW = windowSize;
        this.STR_FORM = "(A: " + this.SIZE_ALPHA + " W: " + this.SIZE_WINDOW + ")";
    }
    
    public int getNumAlphabets() {
        return this.SIZE_ALPHA;
    }
    
    public int getWindowSize() {
        return this.SIZE_WINDOW;
    }
    
    @Override
    public String toString() {
        return this.STR_FORM;
    }
}
