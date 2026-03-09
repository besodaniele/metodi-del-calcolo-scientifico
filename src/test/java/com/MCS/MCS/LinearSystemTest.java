package com.MCS.MCS;

import com.MCS.MCS.LinearAlgebra.Matrix;
import com.MCS.MCS.LinearSystems.LinearSystems;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

public class LinearSystemTest {

    private static final float EPSILON = 1e-5f;

    // ========== VALIDATION TESTS ==========

    @Test
    @DisplayName("Test null matrix throws exception")
    public void testNullMatrix() {
        float[] rhs = {1, 2, 3};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(null, rhs, 3);
        });
    }

    @Test
    @DisplayName("Test null right-hand side throws exception")
    public void testNullRightHandSide() {
        float[][] matrix = {{1, 0}, {2, 1}};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, null, 2);
        });
    }

    @Test
    @DisplayName("Test negative size throws exception")
    public void testNegativeSize() {
        float[][] matrix = {{1, 0}, {2, 1}};
        float[] rhs = {1, 2};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, -1);
        });
    }

    @Test
    @DisplayName("Test zero size throws exception")
    public void testZeroSize() {
        float[][] matrix = {{1, 0}, {2, 1}};
        float[] rhs = {1, 2};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 0);
        });
    }

    @Test
    @DisplayName("Test mismatched matrix dimensions throws exception")
    public void testMismatchedMatrixDimensions() {
        float[][] matrix = {{1, 0}, {2, 1}};  // 2x2 matrix
        float[] rhs = {1, 2, 3};  // size 3 vector
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 3);
        });
    }

    @Test
    @DisplayName("Test mismatched vector dimensions throws exception")
    public void testMismatchedVectorDimensions() {
        float[][] matrix = {{1, 0}, {2, 1}};
        float[] rhs = {1};  // size 1 vector but matrix is 2x2
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 2);
        });
    }

    @Test
    @DisplayName("Test matrix with null row throws exception")
    public void testMatrixWithNullRow() {
        float[][] matrix = {{1, 0}, null};  // Second row is null
        float[] rhs = {1, 2};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 2);
        });
    }

    @Test
    @DisplayName("Test matrix with incorrect row length throws exception")
    public void testMatrixWithIncorrectRowLength() {
        float[][] matrix = {{1, 0}, {2, 1, 3}};  // Second row has 3 elements
        float[] rhs = {1, 2};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 2);
        });
    }

    @Test
    @DisplayName("Test non-lower-triangular matrix throws exception")
    public void testNonLowerTriangular() {
        float[][] matrix = {
            {1, 2, 0},  // Element [0][1] = 2 should be 0 for lower triangular
            {2, 1, 0},
            {3, 4, 1}
        };
        float[] rhs = {1, 2, 3};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 3);
        });
    }

    @Test
    @DisplayName("Test non-upper-triangular matrix throws exception")
    public void testNonUpperTriangular() {
        float[][] matrix = {
            {1, 2, 3},
            {4, 1, 2},  // Element [1][0] = 4 should be 0 for upper triangular
            {0, 0, 1}
        };
        float[] rhs = {1, 2, 3};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveUpperTriangular(matrix, rhs, 3);
        });
    }

    @Test
    @DisplayName("Test matrix with NaN throws exception")
    public void testMatrixWithNaN() {
        float[][] matrix = {
            {1, 0, 0},
            {2, Float.NaN, 0},  // NaN value
            {3, 4, 1}
        };
        float[] rhs = {1, 2, 3};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 3);
        });
    }

    @Test
    @DisplayName("Test matrix with Infinity throws exception")
    public void testMatrixWithInfinity() {
        float[][] matrix = {
            {1, 0, 0},
            {2, Float.POSITIVE_INFINITY, 0},  // Infinity value
            {3, 4, 1}
        };
        float[] rhs = {1, 2, 3};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 3);
        });
    }

    @Test
    @DisplayName("Test vector with NaN throws exception")
    public void testVectorWithNaN() {
        float[][] matrix = {{1, 0}, {2, 1}};
        float[] rhs = {1, Float.NaN};  // NaN value
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 2);
        });
    }

    @Test
    @DisplayName("Test vector with Infinity throws exception")
    public void testVectorWithInfinity() {
        float[][] matrix = {{1, 0}, {2, 1}};
        float[] rhs = {Float.NEGATIVE_INFINITY, 2};  // Infinity value
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, 2);
        });
    }

    @Test
    @DisplayName("Test upper triangular with zero diagonal throws exception")
    public void testUpperTriangularZeroDiagonal() {
        float[][] matrix = {
            {2, 3, 4},
            {0, 0, 6},  // Zero diagonal at row 1
            {0, 0, 7}
        };
        float[] rhs = {4, 18, 25};
        int size = 3;

        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveUpperTriangular(matrix, rhs, size);
        });
    }

    @Test
    @DisplayName("Test checkResolution with null matrix throws exception")
    public void testCheckResolutionNullMatrix() {
        float[] solution = {1, 2, 3};
        float[] rhs = {1, 2, 3};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.checkResolution(null, solution, rhs, 3);
        });
    }

    @Test
    @DisplayName("Test checkResolution with null solution throws exception")
    public void testCheckResolutionNullSolution() {
        float[][] matrix = {{1, 0}, {2, 1}};
        float[] rhs = {1, 2};
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.checkResolution(matrix, null, rhs, 2);
        });
    }

    // ========== FUNCTIONAL TESTS ==========

    @Test
    @DisplayName("Test lower triangular system - simple 3x3")
    public void testResolveLowerTriangularSimple() {
        // System:
        // 2x1 = 4
        // 3x1 + 4x2 = 18
        // 1x1 + 2x2 + 5x3 = 25
        float[][] matrix = {
            {2, 0, 0},
            {3, 4, 0},
            {1, 2, 5}
        };
        float[] rhs = {4, 18, 25};
        int size = 3;

        float[] solution = LinearSystems.resolveLowerTriangular(matrix, rhs, size);

        // Expected solution: x1 = 2, x2 = 3, x3 = 3.4
        assertArrayEquals(new float[]{2, 3, 3.4f}, solution, EPSILON);
        assertTrue(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }

    @Test
    @DisplayName("Test lower triangular system - 2x2")
    public void testResolveLowerTriangular2x2() {
        // System:
        // 3x1 = 6
        // 2x1 + 5x2 = 14
        float[][] matrix = {
            {3, 0},
            {2, 5}
        };
        float[] rhs = {6, 14};
        int size = 2;

        float[] solution = LinearSystems.resolveLowerTriangular(matrix, rhs, size);

        // Expected solution: x1 = 2, x2 = 2
        assertArrayEquals(new float[]{2, 2}, solution, EPSILON);
        assertTrue(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }

    @Test
    @DisplayName("Test upper triangular system - simple 3x3")
    public void testResolveUpperTriangularSimple() {
        // System:
        // 2x1 + 3x2 + 4x3 = 20
        // 5x2 + 6x3 = 28
        // 7x3 = 14
        float[][] matrix = {
            {2, 3, 4},
            {0, 5, 6},
            {0, 0, 7}
        };
        float[] rhs = {20, 28, 14};
        int size = 3;

        float[] solution = LinearSystems.resolveUpperTriangular(matrix, rhs, size);

        // Expected solution: x3 = 2, x2 = (28 - 6*2)/5 = 16/5 = 3.2, x1 = (20 - 3*3.2 - 4*2)/2 = (20 - 9.6 - 8)/2 = 1.2
        assertTrue(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }

    @Test
    @DisplayName("Test upper triangular system - 2x2")
    public void testResolveUpperTriangular2x2() {
        // System:
        // 4x1 + 2x2 = 14
        // 3x2 = 9
        float[][] matrix = {
            {4, 2},
            {0, 3}
        };
        float[] rhs = {14, 9};
        int size = 2;

        float[] solution = LinearSystems.resolveUpperTriangular(matrix, rhs, size);

        // Expected solution: x2 = 3, x1 = (14 - 2*3)/4 = 2
        assertArrayEquals(new float[]{2, 3}, solution, EPSILON);
        assertTrue(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }

    @Test
    @DisplayName("Test lower triangular with identity matrix")
    public void testLowerTriangularIdentity() {
        float[][] matrix = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        };
        float[] rhs = {5, 7, 3};
        int size = 3;

        float[] solution = LinearSystems.resolveLowerTriangular(matrix, rhs, size);

        assertArrayEquals(rhs, solution, EPSILON);
        assertTrue(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }

    @Test
    @DisplayName("Test upper triangular with identity matrix")
    public void testUpperTriangularIdentity() {
        float[][] matrix = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        };
        float[] rhs = {5, 7, 3};
        int size = 3;

        float[] solution = LinearSystems.resolveUpperTriangular(matrix, rhs, size);

        assertArrayEquals(rhs, solution, EPSILON);
        assertTrue(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }

    @Test
    @DisplayName("Test lower triangular with zero diagonal - should throw exception")
    public void testLowerTriangularZeroDiagonal() {
        float[][] matrix = {
            {2, 0, 0},
            {3, 0, 0},  // Zero diagonal at row 1
            {1, 2, 5}
        };
        float[] rhs = {4, 18, 25};
        int size = 3;

        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystems.resolveLowerTriangular(matrix, rhs, size);
        });
    }

    @Test
    @DisplayName("Test matrix-vector multiplication")
    public void testMatrixVectorMultiplication() {
        float[][] matrix = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        float[] vector = {1, 2, 3};
        int size = 3;

        float[] result = Matrix.matrixVectorMultiplication(matrix, vector, size);

        // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3, 7*1 + 8*2 + 9*3] = [14, 32, 50]
        assertArrayEquals(new float[]{14, 32, 50}, result, EPSILON);
    }

    @Test
    @DisplayName("Test vector subtraction")
    public void testVectorSubtraction() {
        float[] vector1 = {10, 20, 30};
        float[] vector2 = {3, 5, 7};
        int size = 3;

        float[] result = Matrix.vectorSubtraction(vector1, vector2, size);

        assertArrayEquals(new float[]{7, 15, 23}, result, EPSILON);
    }

    @Test
    @DisplayName("Test check resolution with exact solution")
    public void testCheckResolutionExact() {
        float[][] matrix = {
            {2, 0, 0},
            {3, 4, 0},
            {1, 2, 5}
        };
        float[] solution = {2, 3, 3.4f};  // Correct solution
        float[] rhs = {4, 18, 25};
        int size = 3;

        assertTrue(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }

    @Test
    @DisplayName("Test check resolution with incorrect solution")
    public void testCheckResolutionIncorrect() {
        float[][] matrix = {
            {2, 0, 0},
            {3, 4, 0},
            {1, 2, 5}
        };
        float[] solution = {1, 1, 1};  // Wrong solution
        float[] rhs = {4, 18, 25};
        int size = 3;

        assertFalse(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }

    @Test
    @DisplayName("Test larger lower triangular system - 4x4")
    public void testLowerTriangular4x4() {
        float[][] matrix = {
            {1, 0, 0, 0},
            {2, 1, 0, 0},
            {3, 2, 1, 0},
            {4, 3, 2, 1}
        };
        float[] rhs = {1, 4, 10, 20};
        int size = 4;

        float[] solution = LinearSystems.resolveLowerTriangular(matrix, rhs, size);

        assertTrue(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }

    @Test
    @DisplayName("Test larger upper triangular system - 4x4")
    public void testUpperTriangular4x4() {
        float[][] matrix = {
            {1, 2, 3, 4},
            {0, 1, 2, 3},
            {0, 0, 1, 2},
            {0, 0, 0, 1}
        };
        float[] rhs = {20, 10, 4, 1};
        int size = 4;

        float[] solution = LinearSystems.resolveUpperTriangular(matrix, rhs, size);

        assertTrue(LinearSystems.checkResolution(matrix, solution, rhs, size));
    }
}

