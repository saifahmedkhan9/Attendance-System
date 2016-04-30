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
       // DataSetIterator iter = ld.getDatasetIterator();

        //when training and testing dataset are seperated in advance
        DataSetIterator trainiter = ld.getTrainDatasetIterator();
        DataSetIterator testiter = ld.getTestDatasetIterator();

        Models model = new Models(ld);
        model.lenet();
        model.build();

        //TrainAndTest tat = new TrainAndTest(model,iter);
        //tat.train();
        //tat.test();
        //when training and testing dataset are seperated in advance

        TrainAndTest tat = new TrainAndTest(model, trainiter, testiter);
        tat.training();
        tat.testing();

        SaveModel sm = new SaveModel(tat);
        sm.save();

/*
        LoadModel lm = new LoadModel();
        MultiLayerNetwork savedNetwork = lm.load();
       test t = new test(savedNetwork, testiter);
        t.testing();
*/



    }






}
