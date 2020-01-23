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
public enum PatternApproximationMethod {
    /**
     * S: Simple approximation. Not precisely aligned to real valued factor of discretization
     */
    S,
    /**
     * I: Interpolated approximation.
     */
    I;

    private final String[] approxMethodDesc = {"S -- Use only simple estimation to create rough subsequence approximations",
                                               "I -- Use interpolation after estimation to create smoother subsequence approximations"};
    
    @Override
    public String toString() {
        return this.approxMethodDesc[this.ordinal()];
    }
}
