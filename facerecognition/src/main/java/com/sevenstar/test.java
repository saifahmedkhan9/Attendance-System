package com.sevenstar;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by ahsan on 30/4/16.
 */
public class test {
    private static final Logger log = LoggerFactory.getLogger(test.class);

    MultiLayerNetwork savedmodel;
    DataSetIterator testiter;
    public test(MultiLayerNetwork mn, DataSetIterator it){

        savedmodel=mn;
        testiter=it;
    }


    public void testing(){// use this method when train and test dataset are seperated beforehand, otherwise use test

        log.info("Evaluate model....");
        testiter.reset();
        Evaluation eval = new Evaluation();
        while(testiter.hasNext()){
            //     int total = testiter.totalExamples();
            // DataSet next = testiter.next(total);
            DataSet next = testiter.next();
            next.shuffle();
            INDArray predict2 = savedmodel.output(next.getFeatureMatrix());
            for (int i = 0; i < predict2.rows(); i++) {
                String actual = next.getLabels().getRow(i).toString().trim();
                String predicted = predict2.getRow(i).toString().trim();
                log.info("actual " + actual + " vs predicted " + predicted);
            }
            eval.eval(next.getLabels(), predict2);
            log.info(eval.stats());
        }

    }
}
