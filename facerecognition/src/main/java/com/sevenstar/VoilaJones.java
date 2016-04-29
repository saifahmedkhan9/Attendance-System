package com.SevenStar;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

//import org.opencv.highgui.Highgui;

import org.opencv.imgcodecs.Imgcodecs; // imread, imwrite, etc
//import org.opencv.videoio;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import java.io.File;

public class VoilaJones {

    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("Hi Opencv "+Core.VERSION);
        System.out.println("\n Running FaceDetector");
        String cwd = System.getProperty("user.dir");
        System.out.println(cwd);
        cwd= cwd + "/opencv-3.1.0/data/haarcascades/";
        System.out.println(cwd);
        File directory = new File(".");

        // get all the files from a directory
        File[] fList = directory.listFiles();
        for (File file : fList) {
            if (file.isFile()) {
                System.out.println(file);
            } else if (file.isDirectory()) {
              //  listf(file.getAbsolutePath(), files);
            }
        }
        File f = new File("ab.jpg");
        File f2 = new File("/home/ahsan/DL4J/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml");

        // returns true if the file exists
        Boolean bool = f2.exists();
        // if file exists
        if(bool)
        {
            // get absolute path
            // path
            System.out.print("Absolute Pathname "+ f2.getAbsolutePath());
            System.out.print("Absolute Pathname "+ f.getAbsolutePath());

        }
        ///home/ahsan/DL4J/dl4j_egs/test-ND4J/src/main/java/com/SevenStar/
        System.out.print("\n");
        ClassLoader loader = VoilaJones.class.getClassLoader();

      // System.out.println(loader.getResource("is.jpg"));//;? unable to resolve getResource() error, always returning null
       // CascadeClassifier faceDetector = new CascadeClassifier(loader.getResource("haarcascade_frontalface_alt.xml").getPath());
       CascadeClassifier faceDetector = new CascadeClassifier(f2.getAbsolutePath());

       // Mat image = Imgcodecs.imread(loader.getResource("is.jpg").getPath());


         Mat image = Imgcodecs
             .imread(f.getAbsolutePath());

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }

        String filename = "ouput.png";
        System.out.println(String.format("Writing %s", filename));
        Imgcodecs.imwrite(filename, image);


    }


}



/*
  <build>
    <resources>
      <resource>
        <directory>src/main/resources</directory>
        <includes>

</includes>
</resource>
</resources>
</build>

*/
