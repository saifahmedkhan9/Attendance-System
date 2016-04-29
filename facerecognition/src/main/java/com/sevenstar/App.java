package com.sevenstar;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Hello world!
 *
 */



public class App 
{

    public static void main( String[] args )

    {
        LoadDataset ld = new LoadDataset();
        DataSetIterator iter = ld.getDatasetIterator();

        Models model = new Models(ld);
        model.lenet();
        model.build();

        TrainAndTest tat = new TrainAndTest(model,iter);
        tat.training();
        tat.test();

        SaveModel sm = new SaveModel(tat);
        sm.save();
       // LoadModel lm = new LoadModel();
       // MultiLayerNetwork savedNetwork = lm.load();


    }






}
