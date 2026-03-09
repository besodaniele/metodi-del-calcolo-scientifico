package com.MCS.MCS.LinearSystems;

public class LinearSystems {

    /**
     * Validates that the matrix is not null and has correct dimensions
     * @throws IllegalArgumentException if validation fails
     */
    private static void validateMatrix(float[][] matrix, int size) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }
        if (size <= 0) {
            throw new IllegalArgumentException("Size must be positive, got: " + size);
        }
        if (matrix.length != size) {
            throw new IllegalArgumentException("Matrix row count (" + matrix.length + ") does not match size (" + size + ")");
        }
        for (int i = 0; i < size; i++) {
            if (matrix[i] == null) {
                throw new IllegalArgumentException("Matrix row " + i + " is null");
            }
            if (matrix[i].length != size) {
                throw new IllegalArgumentException("Matrix row " + i + " has length " + matrix[i].length + ", expected " + size);
            }
        }
    }

    /**
     * Validates that the vector is not null and has correct size
     * @throws IllegalArgumentException if validation fails
     */
    private static void validateVector(float[] vector, int size, String vectorName) {
        if (vector == null) {
            throw new IllegalArgumentException(vectorName + " cannot be null");
        }
        if (size <= 0) {
            throw new IllegalArgumentException("Size must be positive, got: " + size);
        }
        if (vector.length != size) {
            throw new IllegalArgumentException(vectorName + " length (" + vector.length + ") does not match size (" + size + ")");
        }
    }

    /**
     * Validates that the matrix is lower triangular (all elements above diagonal are zero)
     * @throws IllegalArgumentException if matrix is not lower triangular
     */
    private static void validateLowerTriangular(float[][] matrix, int size) {
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                if (Math.abs(matrix[i][j]) > 1e-10f) {
                    throw new IllegalArgumentException(
                        "Matrix is not lower triangular: element at [" + i + "][" + j + "] = " + matrix[i][j] + " (should be 0)"
                    );
                }
            }
        }
    }

    /**
     * Validates that the matrix is upper triangular (all elements below diagonal are zero)
     * @throws IllegalArgumentException if matrix is not upper triangular
     */
    private static void validateUpperTriangular(float[][] matrix, int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < i; j++) {
                if (Math.abs(matrix[i][j]) > 1e-10f) {
                    throw new IllegalArgumentException(
                        "Matrix is not upper triangular: element at [" + i + "][" + j + "] = " + matrix[i][j] + " (should be 0)"
                    );
                }
            }
        }
    }

    /**
     * Validates that all diagonal elements are non-zero
     * @throws IllegalArgumentException if any diagonal element is zero or near-zero
     */
    private static void validateNonZeroDiagonal(float[][] matrix, int size) {
        for (int i = 0; i < size; i++) {
            if (Math.abs(matrix[i][i]) < 1e-8f) {
                throw new IllegalArgumentException("Zero or near-zero diagonal at row " + i + ": " + matrix[i][i]);
            }
        }
    }

    /**
     * Validates that the vector contains only finite values (no NaN or Infinity)
     * @throws IllegalArgumentException if any element is NaN or Infinity
     */
    private static void validateFiniteValues(float[] vector, String vectorName) {
        for (int i = 0; i < vector.length; i++) {
            if (Float.isNaN(vector[i])) {
                throw new IllegalArgumentException(vectorName + " contains NaN at index " + i);
            }
            if (Float.isInfinite(vector[i])) {
                throw new IllegalArgumentException(vectorName + " contains Infinity at index " + i);
            }
        }
    }

    /**
     * Validates that the matrix contains only finite values (no NaN or Infinity)
     * @throws IllegalArgumentException if any element is NaN or Infinity
     */
    private static void validateFiniteMatrixValues(float[][] matrix, int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (Float.isNaN(matrix[i][j])) {
                    throw new IllegalArgumentException("Matrix contains NaN at [" + i + "][" + j + "]");
                }
                if (Float.isInfinite(matrix[i][j])) {
                    throw new IllegalArgumentException("Matrix contains Infinity at [" + i + "][" + j + "]");
                }
            }
        }
    }

    public static float [] resolveLowerTriangular(float [][] matrix, float [] rightHandSide, int size) {
        // Validate inputs
        validateMatrix(matrix, size);
        validateVector(rightHandSide, size, "Right-hand side");
        validateFiniteMatrixValues(matrix, size);
        validateFiniteValues(rightHandSide, "Right-hand side");
        validateLowerTriangular(matrix, size);
        validateNonZeroDiagonal(matrix, size);

        float [] solution = new float[size];
        for (int i = 0; i < size; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += matrix[i][j] * solution[j];
            }

            solution[i] = (float) ((rightHandSide[i] - sum) / matrix[i][i]);
        }
        return solution;
    }
    public static  float [] resolveUpperTriangular(float [][] matrix, float [] rightHandSide, int size) {
        // Validate inputs
        validateMatrix(matrix, size);
        validateVector(rightHandSide, size, "Right-hand side");
        validateFiniteMatrixValues(matrix, size);
        validateFiniteValues(rightHandSide, "Right-hand side");
        validateUpperTriangular(matrix, size);
        validateNonZeroDiagonal(matrix, size);

        float [] solution = new float[size];
        for (int i = size - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < size; j++) {
                sum += matrix[i][j] * solution[j];
            }

            solution[i] = (float) ((rightHandSide[i] - sum) / matrix[i][i]);
        }
        return solution;
    }
    public static  boolean checkResolution(float [][] matrix, float [] solution, float [] rightHandSide, int size) {
        // Validate inputs
        validateMatrix(matrix, size);
        validateVector(solution, size, "Solution");
        validateVector(rightHandSide, size, "Right-hand side");
        validateFiniteMatrixValues(matrix, size);
        validateFiniteValues(solution, "Solution");
        validateFiniteValues(rightHandSide, "Right-hand side");

        double maxResidual = 0.0;
        double maxScale = 0.0;

        for (int i = 0; i < size; i++) {
            double ax = 0.0;
            double rowAbsAxBound = 0.0;
            for (int j = 0; j < size; j++) {
                double aij = matrix[i][j];
                double xj = solution[j];
                ax += aij * xj;
                rowAbsAxBound += Math.abs(aij) * Math.abs(xj);
            }

            double residual = Math.abs(ax - rightHandSide[i]);
            double scale = rowAbsAxBound + Math.abs(rightHandSide[i]);

            if (residual > maxResidual) {
                maxResidual = residual;
            }
            if (scale > maxScale) {
                maxScale = scale;
            }
        }

        if (maxScale == 0.0) {
            return maxResidual == 0.0;
        }

        double relativeResidual = maxResidual / maxScale;
        return relativeResidual <= 1e-5;
    }
}
