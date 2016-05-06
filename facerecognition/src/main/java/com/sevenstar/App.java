package com.sevenstar;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.reflections.vfs.CommonsVfs2UrlType;

import java.io.File;


/**
 * Hello world!
 *
 */

public class App 
{


    public static void main( String[] args )

    {


        //Start training from scratch
    /*
        LoadDataset ld = new LoadDataset();
        DataSetIterator trainiter = ld.getTrainDatasetIterator();
        Models model = new Models(ld);
        model.lenet();
        model.build();
        TrainAndTest tat = new TrainAndTest(model, trainiter);
        tat.training();
       // tat.testing();
        SaveModel sm = new SaveModel(tat);
        sm.save2();
    */



        // load saved model and perform testing on it.
    /*
        LoadDataset ld = new LoadDataset();
        DataSetIterator testiter = ld.getTestDatasetIterator();
        LoadModel lm = new LoadModel();
        MultiLayerNetwork savedNetwork = lm.load();
        test t = new test(savedNetwork, testiter);
        t.testing();
    */



        // resume training from where you left off
        LoadDataset ld = new LoadDataset();
        DataSetIterator trainiter = ld.getTrainDatasetIterator();
        LoadModel lm = new LoadModel();
        MultiLayerNetwork savedNetwork = lm.load();
        TrainAndTest tat = new TrainAndTest(savedNetwork, trainiter, ld);
        tat.training();
        SaveModel sm = new SaveModel(tat);
        sm.save2();





    }


}



//TrainAndTest tat = new TrainAndTest(model,iter);
//tat.train();
//tat.test();
//use it, when training and testing dataset are seperated in advance
