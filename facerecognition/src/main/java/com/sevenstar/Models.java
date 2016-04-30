package com.sevenstar;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
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

    MultiLayerConfiguration.Builder builder;
    MultiLayerConfiguration conf;
    MultiLayerNetwork model;


    public Models(LoadDataset ld){

        nChannels = ld.nChannels;
        width=ld.width;
        height=ld.height;
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
                 //.learningRateScoreBasedDecayRate(1e-1)
                 .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(64)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(2, 2)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(128)
                        //.activation("identity")
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(1024)
                        .dropOut(0.5)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,width,height,nChannels);

    }

public void alexnet(){

    //input shape of image 3*227*227
    double nonZeroBias = 1;
    double dropOut = 0.5;
    SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

    builder = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(true)
            .l2(5 * 1e-4)
            //.activation("relu")
            //.biasLearningRate(0.02)
            //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new NormalDistribution(0.0, 0.01))
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .miniBatch(true)
            .learningRate(0.01)
            .learningRateScoreBasedDecayRate(1e-1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .momentum(0.9)
            .list(13)
            //ConvolutionLayer.Builder(int[] kernelSize, int[] stride, int[] padding)
            .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3})
                    .name("cnn1")
                    .nIn(nChannels)
                    .nOut(96)
                    .build())
            .layer(1, new LocalResponseNormalization.Builder()
                    .name("lrn1")
                    .build())
            .layer(2, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                    .name("maxpool1")
                    .build())
            //conv2
            .layer(3, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2})
                    .name("cnn2")
                    .nOut(256)
                    .biasInit(nonZeroBias)
                    .build())
            .layer(4, new LocalResponseNormalization.Builder()
                    .name("lrn2")
                    .k(2).n(5).alpha(1e-4).beta(0.75)
                    .build())
            .layer(5, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                    .name("maxpool2")
                    .build())
            //conv3
            .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                    .name("cnn3")
                    .nOut(384)
                    .build())
            //conv4
            .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                    .name("cnn4")
                    .nOut(384)
                    .biasInit(nonZeroBias)
                    .build())
            //conv5
            .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                    .name("cnn5")
                    .nOut(256)
                    .biasInit(nonZeroBias)
                    .build())
            .layer(9, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                    .name("maxpool3")
                    .build())
            .layer(10, new DenseLayer.Builder()
                    .name("ffn1")
                    .nOut(4096)
                    .dist(new GaussianDistribution(0, 0.005))
                    .biasInit(nonZeroBias)
                    .dropOut(dropOut)
                    .build())
            .layer(11, new DenseLayer.Builder()
                    .name("ffn2")
                    .nOut(4096)
                    .dist(new GaussianDistribution(0, 0.005))
                    .biasInit(nonZeroBias)
                    .dropOut(dropOut)
                    .build())
            .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .name("output")
                    .nOut(outputNum)
                    .activation("softmax")
                    .build())
            .backprop(true)
            .pretrain(false)
            .cnnInputSize(height,width,nChannels);
    new ConvolutionLayerSetup(builder,width,height,nChannels);

}

    //toDo
public void vggnet() {

}



    public void build(){

        conf = builder.build();
        System.out.println("build");
        model = new MultiLayerNetwork(conf);
        model.init();

    }


}
