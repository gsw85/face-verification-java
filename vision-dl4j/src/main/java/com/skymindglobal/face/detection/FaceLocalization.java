package com.skymindglobal.face.detection;

public class FaceLocalization {
    float left_x;
    float left_y;
    float right_x;
    float right_y;

    public FaceLocalization(float left_x, float left_y, float right_x, float right_y){
        this.left_x = left_x;
        this.left_y = left_y;
        this.right_x = right_x;
        this.right_y = right_y;
    }

    public float getWidth(int imageWidth) {
        if (( this.left_x + this.getRight_x() - this.getLeft_x())> imageWidth){
            return imageWidth - this.left_x;
        }
        return imageWidth;
    }

    public float getHeight(int imageHeight){
        if (( this.left_y + this.getRight_y() - this.getLeft_y())> imageHeight){
            return imageHeight - this.left_y;
        }
        return imageHeight;
    }

    public float getLeft_x(){
        if (this.left_x < 0 ){
            this.left_x = 0;
        }
        return this.left_x;
    }

    public float getLeft_y(){
        if (this.left_y < 0){
            this.left_y = 0;
        }
        return this.left_y;
    }

    public float getRight_x(){
        return this.right_x;
    }

    public float getRight_y(){
        return this.right_y;
    }
}
