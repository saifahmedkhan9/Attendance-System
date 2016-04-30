package com.sevenstar;

/*
 * Created by ahsan on 28/4/16.
 */

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;

import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;



public class LoadDataset {

    int nChannels;
    int width;
    int height;
    int outputNum;
    int batchSize;
    int nEpochs;
    int iterations;
    int seed;
    int splitTrainNum;
    int listenerFreq;
    String labeledPath;
    List<String> labels;
  //  RecordReader recordReader;
    //DataSetIterator iter;

    //constructor
    public LoadDataset() {

        nChannels = 3;
        width=32;
        height=32;
        outputNum = 4;
        batchSize = 20;
        nEpochs = 20;
        iterations = 3;
        seed = 123;
        splitTrainNum = (int) (batchSize * .8);
        listenerFreq = iterations / 5;
       // Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        // Set path to the labeled images
        System.out.println(labeledPath);
        //create array of strings called labels
        labels = new ArrayList<String>();

    }

public DataSetIterator getTrainDatasetIterator(){
    // Set path to the labeled images
    labeledPath = System.getProperty("user.home") + "/DL4J/dl4j_egs/faceScrub/train";
    System.out.println(labeledPath);
    //create array of strings called labels
    labels = new ArrayList<String>();
    return getDatasetIterator();
    }

    public DataSetIterator getTestDatasetIterator(){
        // Set path to the labeled images
        labeledPath = System.getProperty("user.home") + "/DL4J/dl4j_egs/faceScrub/testset";
        System.out.println(labeledPath);
        //create array of strings called labels
        labels = new ArrayList<String>();
        return getDatasetIterator();
    }



    public DataSetIterator getDatasetIterator(){
    //traverse dataset to get each label
    for (File f : new File(labeledPath).listFiles()) {
        labels.add(f.getName());
        System.out.println(f);
    }

    System.out.print(labels.size());


    // Instantiating RecordReader. Specify height and width of images and no of channels.
   RecordReader  recordReader = new ImageRecordReader(width, height, nChannels, true, labels);

        try {

            // Point to data path.
            recordReader.initialize(new FileSplit(new File(labeledPath), new Random(seed)));//2nd parameter to randomize dataset
        }
        catch(IOException e){
            e.printStackTrace();
            System.exit(0);
        }
        catch(InterruptedException e){
            e.printStackTrace();
            System.exit(0);
        }

    System.out.println("\n Total Classes :" + recordReader.getLabels());
    // Canova to DL4J
     DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, width*height*nChannels, labels.size());
        return iter;
    }



}
