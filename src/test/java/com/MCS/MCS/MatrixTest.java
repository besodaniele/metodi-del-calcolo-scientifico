package com.MCS.MCS;

import com.MCS.MCS.LinearAlgebra.Matrix;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MatrixTest {
    @Test
    @DisplayName("Test matrix-vector multiplication with a simple 2x2 matrix and 2D vector")
    public void testMatrixVectorMultiplication_SimpleCase() {
        float[][] matrix = {
                {1.0f, 2.0f},
                {3.0f, 4.0f}
        };
        float[] vector = {5.0f, 6.0f};
        int size = 2;

        float[] expected = {17.0f, 39.0f}; // (1*5 + 2*6, 3*5 + 4*6)
        float[] result = Matrix.matrixVectorMultiplication(matrix, vector, size);

        assertArrayEquals(expected, result, "Matrix-vector multiplication did not produce the expected result.");
    }
    @Test
    @DisplayName("Test LU factorization without pivoting on a simple 3x3 matrix")
    public void testLUFactorization_SimpleCase() {
        float[][] matrix = {
                {10.0f, -7.0f, 0.0f},
                {-3.0f, 2.0f, 6.0f},
                {5.0f, -1.0f, 5.0f}
        };
        int size = 3;

        var lu = Matrix.factorizeLU(matrix, size);
        float[][] L = lu.get(0);
        float[][] U = lu.get(1);

        float[][] expectedL = {
                {1.0f, 0.0f, 0.0f},
                {-0.3f, 1.0f, 0.0f},
                {0.5f, -25.0f, 1.0f}
        };
        float[][] expectedU = {
                {10.0f, -7.0f, 0.0f},
                {0.0f, -0.1f, 6.0f},
                {0.0f, 0.0f, 155.0f}
        };

        for (int i = 0; i < size; i++) {
            assertArrayEquals(expectedL[i], L[i], 1e-3f, "L matrix is not correct.");
            assertArrayEquals(expectedU[i], U[i], 1e-3f, "U matrix is not correct.");
        }
    }

    @Test
    @DisplayName("Test LU factorization reconstructs the original matrix")
    public void testLUFactorization_ReconstructsOriginalMatrix() {
        float[][] matrix = {
                {4.0f, 3.0f},
                {6.0f, 3.0f}
        };
        int size = 2;

        var lu = Matrix.factorizeLU(matrix, size);
        float[][] reconstructed = Matrix.matrixMultiplication(lu.get(0), lu.get(1), size);

        for (int i = 0; i < size; i++) {
            assertArrayEquals(matrix[i], reconstructed[i], 1e-4f, "L * U must reconstruct the original matrix.");
        }
    }

    @Test
    @DisplayName("Test LU factorization on a 1x1 matrix")
    public void testLUFactorization_SizeOneMatrix() {
        float[][] matrix = {{7.0f}};

        var lu = Matrix.factorizeLU(matrix, 1);

        assertArrayEquals(new float[]{1.0f}, lu.get(0)[0], "L must be the identity matrix for size 1.");
        assertArrayEquals(new float[]{7.0f}, lu.get(1)[0], "U must match the original matrix for size 1.");
    }

    @Test
    @DisplayName("Test LU factorization throws on zero pivot without pivoting")
    public void testLUFactorization_ZeroPivotThrows() {
        float[][] matrix = {
                {0.0f, 1.0f},
                {2.0f, 3.0f}
        };

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
                () -> Matrix.factorizeLU(matrix, 2));

        assertTrue(exception.getMessage().contains("pivot"), "The error message should mention the pivot.");
    }

    @Test
    @DisplayName("Test inverseOnlyOneColumn with a simple 3x3 matrix and column index 0")
    public void testInverseOnlyOneColumn_SimpleCase() {
        float[][] matrix = {
                {1.0f, 0.0f, 0.0f},
                {4.0f, 1.0f, 1.0f},
                {7.0f, 0.0f, 0.0f}
        };
        int size = 3;
        int column = 0;

        float[][] expectedInverse = {
                {1.0f, 0.0f, 0.0f},
                {-4.0f, 1.0f, 0.0f},
                {-7.0f, 0.0f, 1.0f}
        };

        float[][] result = Matrix.inverseOnlyOneColumn(matrix, size, column);

        for (int i = 0; i < size; i++) {
            assertArrayEquals(expectedInverse[i], result[i], "Inverse matrix is not correct.");
        }
    }
}
