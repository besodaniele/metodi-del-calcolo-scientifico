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
     * Utility class: non istanziabile.
     */
    private Matrix() {
        throw new UnsupportedOperationException("Utility class");
    }

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

    /**
     * Multiplies two square matrices: result = matrix1 × matrix2.
     * <p>
     * For each output cell {@code result[i][j]}, computes the dot product between
     * row {@code i} of {@code matrix1} and column {@code j} of {@code matrix2}.
     * Uses double-precision accumulation to reduce rounding errors, then stores
     * the final value as float.
     * </p>
     *
     * @param matrix1 the left matrix operand (size × size)
     * @param matrix2 the right matrix operand (size × size)
     * @param size    the common matrix dimension
     * @return a new size × size matrix containing the product
     * @throws ArrayIndexOutOfBoundsException if matrix dimensions do not match {@code size}
     * @throws NullPointerException if any input matrix or row is null
     */
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

    /**
     * Performs element-wise subtraction of two square matrices: result = matrix1 - matrix2.
     *
     * @param matrix1 the minuend matrix (size × size)
     * @param matrix2 the subtrahend matrix (size × size)
     * @param size    the matrix dimension
     * @return a new size × size matrix containing element-wise differences
     * @throws ArrayIndexOutOfBoundsException if matrix dimensions do not match {@code size}
     * @throws NullPointerException if any input matrix or row is null
     */
    public static float [][] matrixSubtraction(float [][] matrix1, float [][] matrix2, int size) {
        float [][] result = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }
        return result;
    }

    /**
     * Performs element-wise addition of two square matrices: result = matrix1 + matrix2.
     *
     * @param matrix1 the first addend matrix (size × size)
     * @param matrix2 the second addend matrix (size × size)
     * @param size    the matrix dimension
     * @return a new size × size matrix containing element-wise sums
     * @throws ArrayIndexOutOfBoundsException if matrix dimensions do not match {@code size}
     * @throws NullPointerException if any input matrix or row is null
     */
    public static float [][] matrixAddition(float [][] matrix1, float [][] matrix2, int size) {
        float [][] result = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        return result;
    }

    /**
     * Multiplies each element of a square matrix by a scalar value.
     *
     * @param matrix the input matrix (size × size)
     * @param scalar the scalar multiplier
     * @param size   the matrix dimension
     * @return a new size × size matrix where each element is {@code matrix[i][j] * scalar}
     * @throws ArrayIndexOutOfBoundsException if matrix dimensions do not match {@code size}
     * @throws NullPointerException if the matrix or any row is null
     */
    public static float [][] scalarMultiplication(float [][] matrix, float scalar, int size) {
        float [][] result = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * Builds an elementary inverse matrix that stores one elimination column.
     * <p>
     * The returned matrix starts as identity, then for rows below {@code column}
     * sets {@code inverse[i][column] = -matrix[i][column]}.
     * This is useful in elimination workflows where one column of multipliers
     * is encoded as an elementary transformation.
     * </p>
     *
     * @param matrix the source matrix containing elimination coefficients
     * @param size   the matrix dimension
     * @param column the target pivot column
     * @return an identity-like matrix with the specified modified column entries
     */
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

    /**
     * Computes the PLU factorization of a square matrix with partial pivoting.
     * <p>
     * The returned matrices satisfy {@code P * A = L * U}, where {@code P} is a
     * permutation matrix, {@code L} is unit lower triangular, and {@code U} is
     * upper triangular.
     * </p>
     *
     * @param matrix the square matrix to factorize
     * @param size   the dimension of the matrix
     * @return a list containing {@code P} at index 0, {@code L} at index 1, and {@code U} at index 2
     * @throws IllegalArgumentException if the matrix is null, not square, size is not
     *                                  positive, or a zero/near-zero pivot is encountered
     */
    public static List<float[][]> factorizePLU(float[][] matrix, int size) {
        float[][] P = createIdentityMatrix(size);
        float[][] L = createIdentityMatrix(size);
        float[][] U = copyMatrix(matrix, size);

        for (int j = 0; j < size - 1; j++) {
            int pivotRow = findPivotRow(U, j, size);

            swapRows(U, j, pivotRow);
            swapRows(P, j, pivotRow);
            swapLRowsUntilColumn(L, j, pivotRow, j);

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

        return List.of(P, L, U);
    }

    private static float[][] createIdentityMatrix(int size) {
        float[][] identity = new float[size][size];
        for (int i = 0; i < size; i++) {
            identity[i][i] = 1.0f;
        }
        return identity;
    }

    private static int findPivotRow(float[][] matrix, int column, int size) {
        int pivotRow = column;
        float max = Math.abs(matrix[column][column]);
        for (int i = column + 1; i < size; i++) {
            float candidate = Math.abs(matrix[i][column]);
            if (candidate > max) {
                max = candidate;
                pivotRow = i;
            }
        }
        return pivotRow;
    }

    private static void swapRows(float[][] matrix, int rowA, int rowB) {
        if (rowA == rowB) {
            return;
        }
        float[] temp = matrix[rowA];
        matrix[rowA] = matrix[rowB];
        matrix[rowB] = temp;
    }

    private static void swapLRowsUntilColumn(float[][] matrixL, int rowA, int rowB, int columnExclusive) {
        if (rowA == rowB) {
            return;
        }
        for (int j = 0; j < columnExclusive; j++) {
            float temp = matrixL[rowA][j];
            matrixL[rowA][j] = matrixL[rowB][j];
            matrixL[rowB][j] = temp;
        }
    }

    /**
     * Creates a defensive deep copy of a square matrix after validating dimensions.
     *
     * @param matrix the matrix to copy
     * @param size   the expected matrix dimension
     * @return a new size × size matrix with the same values as the input
     * @throws IllegalArgumentException if matrix is null, size is non-positive,
     *                                  row count does not match size, a row is null,
     *                                  or a row length does not match size
     */
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

    /**
     * Computes the transpose of a square matrix.
     * <p>
     * Note: method name is kept as {@code traspose} to match existing API.
     * </p>
     *
     * @param matrix the input matrix (size × size)
     * @param size   the matrix dimension
     * @return a new size × size matrix where {@code result[j][i] = matrix[i][j]}
     */
    public static float[][] traspose(float [][] matrix, int size) {
        float [][] result = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    /**
     * Checks if a matrix is symmetric or semi-symmetric within a specified tolerance.
     * <p>
     * A matrix is considered symmetric if {@code matrix[i][j] == matrix[j][i]} for all
     * i, j. It is considered semi-symmetric if the absolute difference
     * {@code |matrix[i][j] - matrix[j][i]|} is less than or equal to a defined tolerance
     * for any pair of indices (i, j). If the matrix is symmetric, it is returned as-is.
     * If it is semi-symmetric, a new symmetric matrix is constructed by averaging the
     * original matrix with its transpose. If the matrix is neither symmetric nor
     * semi-symmetric, an exception is thrown.
     *
     * @param matrix the square matrix to check for symmetry
     * @param size   the dimension of the matrix
     * @return the original matrix if it is symmetric, or a new symmetric matrix if it is semi-symmetric
     * @throws IllegalArgumentException if the input matrix is null, not square, size is not positive,
     *                                  or if the matrix is neither symmetric nor semi-symmetric
     */
    public static float [][] isSymmetric(float [][] matrix, int size) {
        boolean isSymmetric = true;
        boolean semiSymmetric = false;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (matrix[i][j] != matrix[j][i]) {
                    isSymmetric = false;
                    semiSymmetric = false;
                }
                if (Math.abs(matrix[i][j] - matrix[j][i] )<= PIVOT_TOLERANCE) {
                    semiSymmetric = true;
                }

            }
        }
        if(isSymmetric){
            return matrix;
        }
        else if (semiSymmetric) {
            var traspose = traspose(matrix, size);
            return scalarMultiplication(matrixAddition(matrix, traspose, size),0.5f, size);
        }
        else {
            throw new IllegalArgumentException("Matrix is not symmetric");
        }
    }
}
