package com.sevenstar;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Created by ahsan on 29/4/16.
 */
public class LoadModel {

    MultiLayerNetwork savedNetwork;

    public MultiLayerNetwork load() {


        try {
        //Load network configuration from disk:
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));
        //reading model

            DataInputStream dis = new DataInputStream(new FileInputStream("newimagemodel.bin"));

            INDArray newParams = Nd4j.read(dis);
            dis.close();

            savedNetwork = new MultiLayerNetwork(confFromJson);
            savedNetwork.init();
            savedNetwork.setParams(newParams);
            System.out.println(savedNetwork.params());

        }

        catch(IOException e){
            e.printStackTrace();
            System.exit(0);
        }

        return savedNetwork;

    }





}
package com.sevenstar;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Created by ahsan on 29/4/16.
 */
public class LoadModel {

    MultiLayerNetwork savedNetwork;

    public MultiLayerNetwork load() {


        try {
        //Load network configuration from disk:
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));
        //reading model

            DataInputStream dis = new DataInputStream(new FileInputStream("newimagemodel.bin"));

            INDArray newParams = Nd4j.read(dis);
            dis.close();

            savedNetwork = new MultiLayerNetwork(confFromJson);
            savedNetwork.init();
            savedNetwork.setParams(newParams);
            System.out.println(savedNetwork.params());

        }

        catch(IOException e){
            e.printStackTrace();
            System.exit(0);
        }

        return savedNetwork;

    }





}
