package com.MCS.MCS.LinearAlgebra;

public class Matrix {
    public static  float [] matrixVectorMultiplication(float [][] matrix, float [] vector, int size) {
        float [] result = new float[size];
        for (int i = 0; i < size; i++) {
            double sum = 0.0;
            for (int j = 0; j < size; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = (float) sum;
        }
        return result;
    }
    public static  float [] vectorSubtraction(float [] vector1, float [] vector2, int size) {
        float [] result = new float[size];
        for (int i = 0; i < size; i++) {
            result[i] = vector1[i] - vector2[i];
        }
        return result;
    }
}
