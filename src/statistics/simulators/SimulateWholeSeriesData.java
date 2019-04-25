/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package statistics.simulators;

import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class SimulateWholeSeriesData extends DataSimulator {
   
      public SimulateWholeSeriesData(double[][] paras){
        super(paras);
        for(int i=0;i<nosClasses;i++)
            models.add(new SinusoidalModel(paras[i]));
    }
      public void setWarping(){
          for(Model m:models){
              ((SinusoidalModel)m).setWarp(true);
          }
      }
       
}
