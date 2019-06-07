package com.skymindglobal.faceverification.common;

import java.io.*;
import java.util.List;

public class LabelManager {

    public static String[] importLabels(String labelFilename) throws IOException, ClassNotFoundException {
        ObjectInputStream in = new ObjectInputStream(new FileInputStream(labelFilename));
        List<String> array = (List<String>) in.readObject();
        in.close();
        return array.toArray(new String[0]);
    }

    public static void exportLabels(String labelFilename, List<String> labels) throws IOException {
        ObjectOutputStream out = new ObjectOutputStream(
                new FileOutputStream(labelFilename)
        );
        out.writeObject(labels);
        out.flush();
        out.close();
    }
}
