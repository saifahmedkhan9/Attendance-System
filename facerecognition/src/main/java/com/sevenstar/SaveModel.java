package com.sevenstar;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
//import java.nio.file.*;
import java.nio.file.Files;
import java.nio.file.Paths;


/**
 * Created by ahsan on 29/4/16.
 */

public class SaveModel {

    MultiLayerNetwork model;


    public SaveModel(TrainAndTest tat){
        model=tat.model;

    }
    public void save(){

        try {
            //Write the network parameters:
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("newimagemodel.bin"));
            Nd4j.write(bos, model.params());

            //Write the network configuration:
            FileUtils.write(new File("conf.yaml"), model.conf().toYaml());
            FileUtils.write(new File("conf.json"), model.conf().toJson());
        }

        catch (IOException e)
        {
            e.printStackTrace();
            System.exit(0);
        }


     /*   //Write the network parameters:
       OutputStream fos = Files.newOutputStream(Paths.get("coefficients.bin"));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        //Write the network configuration:

        FileUtils.writeStringToFile(new File("conf.json"), model.getLayerWiseConfigurations().toJson());
    */

    }


    public void save2() {

        try {
            OutputStream fos = Files.newOutputStream(Paths.get("newimagemodel.bin"));
            DataOutputStream dos = new DataOutputStream(fos);
            Nd4j.write(model.params(), dos);
            dos.flush();
            dos.close();
            //Write the network configuration:

            FileUtils.writeStringToFile(new File("conf.json"), model.getLayerWiseConfigurations().toJson());
        }
        catch(Exception e){

        }
    }



    }
