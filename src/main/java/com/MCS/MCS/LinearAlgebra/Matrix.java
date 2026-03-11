package com.MCS.MCS.LinearAlgebra;

import java.util.List;

/**
 * Utility class providing basic linear algebra operations on matrices and vectors.
 * <p>
 * This class contains static methods for performing common linear algebra operations
 * such as matrix-vector multiplication and vector subtraction. All operations use
 * single-precision floating-point arithmetic (float) for storage with double-precision
 * intermediate calculations for improved numerical accuracy.
 * </p>
 * <p>
 * Note: This class assumes square matrices and does not perform dimension validation.
 * Callers are responsible for ensuring that matrix and vector dimensions are consistent
 * with the provided size parameter.
 * </p>
 *
 * @author MCS
 * @version 1.0
 * @since 1.0
 */
public class Matrix {

    private static final float PIVOT_TOLERANCE = 1e-8f;

    /**
     * Performs matrix-vector multiplication: result = matrix × vector.
     * <p>
     * Computes the product of a square matrix and a vector. The resulting vector
     * has the same dimension as the input vector. For each element i of the result:
     * <pre>
     * result[i] = Σ(matrix[i][j] × vector[j]) for j = 0 to size-1
     * </pre>
     * <p>
     * This implementation uses double-precision arithmetic for the accumulation of
     * products to minimize floating-point rounding errors, then casts the final
     * result back to float precision.
     * <p>
     * <b>Time complexity:</b> O(n²) where n is the size parameter.<br>
     * <b>Space complexity:</b> O(n) for the result vector.
     *
     * @param matrix A square matrix of size × size elements. Each element matrix[i][j]
     *               represents the coefficient at row i and column j.
     * @param vector An input vector of length size. Each element vector[j] represents
     *               the j-th component of the vector.
     * @param size   The dimension of the matrix and vector. Must match the actual
     *               dimensions of the input arrays (matrix should be size×size,
     *               vector should have length size).
     * @return A new vector of length size containing the result of the matrix-vector
     *         multiplication. Element i contains the dot product of matrix row i
     *         with the input vector.
     * @throws ArrayIndexOutOfBoundsException if the actual dimensions of matrix or vector
     *                                        do not match the specified size parameter
     * @throws NullPointerException if matrix, vector, or any row of the matrix is null
     *
     * @see #vectorSubtraction(float[], float[], int)
     */
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

    public static float [][] matrixMultiplication(float [][] matrix1, float [][] matrix2, int size) {
        float [][] result = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double sum = 0.0;
                for (int k = 0; k < size; k++) {
                    sum += matrix1[i][k] * matrix2[k][j];
                }
                result[i][j] = (float) sum;
            }
        }
        return result;
    }

    /**
     * Performs element-wise subtraction of two vectors: result = vector1 - vector2.
     * <p>
     * Computes the difference between two vectors of the same dimension. For each
     * element i of the result:
     * <pre>
     * result[i] = vector1[i] - vector2[i]
     * </pre>
     * <p>
     * This operation is commonly used in computing residuals when verifying solutions
     * to linear systems (e.g., computing b - Ax).
     * <p>
     * <b>Time complexity:</b> O(n) where n is the size parameter.<br>
     * <b>Space complexity:</b> O(n) for the result vector.
     *
     * @param vector1 The minuend vector of length size (the vector from which to subtract).
     * @param vector2 The subtrahend vector of length size (the vector to subtract).
     * @param size    The length of both input vectors. Must match the actual length
     *                of both vector1 and vector2.
     * @return A new vector of length size where each element i contains
     *         (vector1[i] - vector2[i]).
     * @throws ArrayIndexOutOfBoundsException if the actual length of vector1 or vector2
     *                                        does not match the specified size parameter
     * @throws NullPointerException if vector1 or vector2 is null
     *
     * @see #matrixVectorMultiplication(float[][], float[], int)
     */
    public static  float [] vectorSubtraction(float [] vector1, float [] vector2, int size) {
        float[] result = new float[size];
        for (int i = 0; i < size; i++) {
            result[i] = vector1[i] - vector2[i];
        }
        return result;
    }
    public static float[][] inverseOnlyOneColumn(float [][] matrix, int size, int column) {
        float [][] inverse = new float[size][size];
        for (int i=0; i < size; i++) {
            inverse[i][i] = 1;
        }
        for (int i = column+1; i < size; i++) {
            inverse[i][column] = -matrix[i][column];
        }
        return inverse;
    }


    /**
     * Computes the LU factorization of a square matrix without pivoting.
     * <p>
     * The returned matrices satisfy {@code A = L * U}, where {@code L} is unit lower
     * triangular and {@code U} is upper triangular. Since this implementation does not
     * perform pivoting, it requires every pivot on the diagonal to be non-zero during
     * elimination.
     * </p>
     *
     * @param matrix the square matrix to factorize
     * @param size   the dimension of the matrix
     * @return a list containing {@code L} at index 0 and {@code U} at index 1
     * @throws IllegalArgumentException if the matrix is null, not square, size is not
     *                                  positive, or a zero/near-zero pivot is encountered
     */
    public static List<float[][]> factorizeLU(float[][] matrix, int size) {
        float[][] L = new float[size][size];
        float[][] U = copyMatrix(matrix, size);

        for (int i = 0; i < size; i++) {
            L[i][i] = 1.0f;
        }

        for (int j = 0; j < size - 1; j++) {
            if (Math.abs(U[j][j]) < PIVOT_TOLERANCE) {
                throw new IllegalArgumentException("Zero or near-zero pivot encountered at [" + j + "][" + j + "]: " + U[j][j]);
            }

            for (int i = j + 1; i < size; i++) {
                float multiplier = U[i][j] / U[j][j];
                L[i][j] = multiplier;

                for (int k = j; k < size; k++) {
                    U[i][k] -= multiplier * U[j][k];
                }
                U[i][j] = 0.0f;
            }
        }

        if (Math.abs(U[size - 1][size - 1]) < PIVOT_TOLERANCE) {
            throw new IllegalArgumentException("Zero or near-zero pivot encountered at [" + (size - 1) + "][" + (size - 1) + "]: " + U[size - 1][size - 1]);
        }

        return List.of(L, U);
    }

    private static float[][] copyMatrix(float[][] matrix, int size) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        if (size <= 0) {
            throw new IllegalArgumentException("Size must be positive, got: " + size);
        }
        if (matrix.length != size) {
            throw new IllegalArgumentException("Matrix row count (" + matrix.length + ") does not match size (" + size + ")");
        }

        float[][] copy = new float[size][size];
        for (int i = 0; i < size; i++) {
            if (matrix[i] == null) {
                throw new IllegalArgumentException("Matrix row " + i + " is null");
            }
            if (matrix[i].length != size) {
                throw new IllegalArgumentException("Matrix row " + i + " has length " + matrix[i].length + ", expected " + size);
            }
            System.arraycopy(matrix[i], 0, copy[i], 0, size);
        }
        return copy;
    }
}
