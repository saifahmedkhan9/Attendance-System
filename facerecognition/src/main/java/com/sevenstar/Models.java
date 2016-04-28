package com.sevenstar;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

/**
 * Created by ahsan on 28/4/16.
 */
public class Models {

    int nChannels;
    int outputNum;
    int batchSize;
    int nEpochs;
    int iterations;
    int seed;
    int splitTrainNum;
    int listenerFreq;
    String labeledPath;
    List<String> labels;

    MultiLayerConfiguration.Builder builder;
    MultiLayerConfiguration conf;
    MultiLayerNetwork model;


    public Models(LoadDataset ld){

        nChannels = ld.nChannels;
        outputNum = ld.outputNum;
        batchSize = ld.batchSize;
        nEpochs = ld.nEpochs;
        iterations = ld.iterations;
        seed = ld.seed;
        splitTrainNum = ld.splitTrainNum;
        listenerFreq =ld.listenerFreq;
        labeledPath = ld.labeledPath;
        labels = ld.labels;

    }

    public void lenet(){

         builder = new NeuralNetConfiguration.Builder()

                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                //.activation("relu")
                //.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .miniBatch(true)
                .learningRate(0.01)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(10)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(2, 2)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        //.activation("identity")
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(200)
                        .dropOut(0.5)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,32,32,3);

    }




    public void build(){

        conf = builder.build();
        System.out.println("build");
        model = new MultiLayerNetwork(conf);
        model.init();

    }


}
