package com.MCS.MCS.LinearAlgebra;

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

    /**
     * Performs matrix-vector multiplication: result = matrix × vector.
     * <p>
     * Computes the product of a square matrix and a vector. The resulting vector
     * has the same dimension as the input vector. For each element i of the result:
     * <pre>
     * result[i] = Σ(matrix[i][j] × vector[j]) for j = 0 to size-1
     * </pre>
     * </p>
     * <p>
     * This implementation uses double-precision arithmetic for the accumulation of
     * products to minimize floating-point rounding errors, then casts the final
     * result back to float precision.
     * </p>
     * <p>
     * <b>Time complexity:</b> O(n²) where n is the size parameter.<br>
     * <b>Space complexity:</b> O(n) for the result vector.
     * </p>
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

    /**
     * Performs element-wise subtraction of two vectors: result = vector1 - vector2.
     * <p>
     * Computes the difference between two vectors of the same dimension. For each
     * element i of the result:
     * <pre>
     * result[i] = vector1[i] - vector2[i]
     * </pre>
     * </p>
     * <p>
     * This operation is commonly used in computing residuals when verifying solutions
     * to linear systems (e.g., computing b - Ax).
     * </p>
     * <p>
     * <b>Time complexity:</b> O(n) where n is the size parameter.<br>
     * <b>Space complexity:</b> O(n) for the result vector.
     * </p>
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
        float [] result = new float[size];
        for (int i = 0; i < size; i++) {
            result[i] = vector1[i] - vector2[i];
        }
        return result;
    }
}
