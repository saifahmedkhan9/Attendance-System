package com.sevenstar;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
/**
 * Created by ahsan on 29/4/16.
 */
public class TrainAndTest {

    private static final Logger log = LoggerFactory.getLogger(TrainAndTest.class);

    MultiLayerConfiguration.Builder builder;
    MultiLayerConfiguration conf;
    MultiLayerNetwork model;
    DataSetIterator iter, trainiter, testiter;
    int splitTrainNum;
    int seed;
    int outputNum;
    int epoch;
    List<INDArray> testInput;
    List<INDArray> testLabels;

/*
    public TrainAndTest(Models mod, DataSetIterator it){
        splitTrainNum=mod.splitTrainNum;
        seed = mod.seed;
        outputNum=mod.outputNum;
        builder=mod.builder;
        conf = mod.conf;
        model = mod.model;
        iter=it;

    }

public TrainAndTest(Models mod, DataSetIterator trainIter, DataSetIterator testIter){
    seed = mod.seed;
    outputNum=mod.outputNum;
    builder=mod.builder;
    conf = mod.conf;
    model = mod.model;
    trainiter=trainIter;
    testiter=testIter;
    epoch=mod.nEpochs;
}
*/

    public TrainAndTest(Models mod, DataSetIterator trainIter){
        seed = mod.seed;
        outputNum=mod.outputNum;
        builder=mod.builder;
        conf = mod.conf;
        model = mod.model;
        trainiter=trainIter;
        epoch=mod.nEpochs;
    }

    // resume training after certain epoch
    public TrainAndTest(MultiLayerNetwork savedModel, DataSetIterator trainIter, LoadDataset ld){
        seed = ld.seed;
        outputNum=ld.outputNum;
        epoch=ld.nEpochs;

        model = savedModel;
        trainiter=trainIter;

    }


public void training() { // use this method when train and test dataset are seperated beforehand, otherwise use train
    Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
    model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
    System.out.println("training");
    // Training
    for (int i = 0; i < epoch; i++) {
            trainiter.reset();
        while (trainiter.hasNext()) {
            DataSet next = trainiter.next();//load next batch
            //next.shuffle();
            next.normalizeZeroMeanZeroUnitVariance();
            System.out.println(next.numExamples() + " " + next.numInputs() + " " + next.numOutcomes());
            System.out.println(next.getLabels());
            model.fit(next);
        }

    }
}



/*

public void testing(){// use this method when train and test dataset are seperated beforehand, otherwise use test

    log.info("Evaluate model....");
    testiter.reset();
    Evaluation eval = new Evaluation();
    while(testiter.hasNext()){
   //     int total = testiter.totalExamples();
       // DataSet next = testiter.next(total);
        DataSet next = testiter.next();
        next.shuffle();
        INDArray predict2 = model.output(next.getFeatureMatrix());
        for (int i = 0; i < predict2.rows(); i++) {
            String actual = next.getLabels().getRow(i).toString().trim();
            String predicted = predict2.getRow(i).toString().trim();
            log.info("actual " + actual + " vs predicted " + predicted);
        }
        eval.eval(next.getLabels(), predict2);
        log.info(eval.stats());
    }

}

    public void train(){

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        System.out.println("train");
        testInput = new ArrayList<INDArray>();
        testLabels =new ArrayList<INDArray>();

        // Training
        while(iter.hasNext()){
            DataSet next = iter.next();//load next batch
            //next.shuffle();
            next.normalizeZeroMeanZeroUnitVariance();
            System.out.println(next.numExamples() + " " + next.numInputs() + " " + next.numOutcomes());
            System.out.println(next.getLabels());
            SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
            DataSet train = testAndTrain.getTrain();
            testInput.add(testAndTrain.getTest().getFeatureMatrix());
            testLabels.add(testAndTrain.getTest().getLabels());
            model.fit(train);

        }


    }

    public void test(){
        System.out.println("Size : " + testInput.size());

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));

            String actual = testLabels.get(i).toString().trim();
            String predicted = output.toString().trim();
            log.info("actual\n " + actual + " vs predicted \n" + predicted);

            eval.eval(testLabels.get(i), output);
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

*/


}
