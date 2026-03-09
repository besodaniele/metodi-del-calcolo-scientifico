package com.MCS.MCS.LinearSystems;

/**
 * Utility class for solving triangular linear systems and verifying solutions.
 * <p>
 * This class provides methods to solve lower triangular and upper triangular systems
 * using forward and backward substitution respectively, along with a method to
 * validate the correctness of solutions.
 * </p>
 *
 * @author MCS
 * @version 1.0
 */
public class LinearSystems {

    /**
     * Validates that the matrix is not null and has correct dimensions.
     * <p>
     * Ensures the matrix is square with dimensions matching the specified size,
     * and that no rows are null.
     * </p>
     *
     * @param matrix The matrix to validate, represented as a 2D array of floats
     * @param size   The expected size of the matrix (number of rows and columns)
     * @throws IllegalArgumentException if the matrix is null, size is non-positive,
     *                                  row count doesn't match size, any row is null,
     *                                  or any row length doesn't match size
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
     * Validates that the vector is not null and has the correct size.
     *
     * @param vector     The vector to validate, represented as an array of floats
     * @param size       The expected length of the vector
     * @param vectorName The name of the vector (for error messages)
     * @throws IllegalArgumentException if the vector is null, size is non-positive,
     *                                  or vector length doesn't match size
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
     * Validates that the matrix is lower triangular.
     * <p>
     * A lower triangular matrix has all elements above the main diagonal equal to zero.
     * This method checks that all elements a[i][j] where j &gt; i are zero (within a tolerance of 1e-10).
     * </p>
     *
     * @param matrix The matrix to validate
     * @param size   The size of the matrix
     * @throws IllegalArgumentException if any element above the diagonal is non-zero
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
     * Validates that the matrix is upper triangular.
     * <p>
     * An upper triangular matrix has all elements below the main diagonal equal to zero.
     * This method checks that all elements a[i][j] where j &lt; i are zero (within a tolerance of 1e-10).
     * </p>
     *
     * @param matrix The matrix to validate
     * @param size   The size of the matrix
     * @throws IllegalArgumentException if any element below the diagonal is non-zero
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
     * Validates that all diagonal elements are non-zero.
     * <p>
     * Non-zero diagonal elements are required for triangular system solvers to avoid
     * division by zero. Elements with absolute value less than 1e-8 are considered zero.
     * </p>
     *
     * @param matrix The matrix to validate
     * @param size   The size of the matrix
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
     * Validates that the vector contains only finite values (no NaN or Infinity).
     *
     * @param vector     The vector to validate
     * @param vectorName The name of the vector (for error messages)
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
     * Validates that the matrix contains only finite values (no NaN or Infinity).
     *
     * @param matrix The matrix to validate
     * @param size   The size of the matrix
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

    /**
     * Solves a lower triangular linear system Lx = b using forward substitution.
     * <p>
     * Given a lower triangular matrix L and a right-hand side vector b, this method
     * computes the solution vector x such that Lx = b. The algorithm proceeds from
     * the first equation to the last, substituting previously computed values.
     * </p>
     * <p>
     * The algorithm works as follows: for each row i (from 0 to size-1):
     * <pre>
     * x[i] = (b[i] - sum(L[i][j] * x[j] for j = 0 to i-1)) / L[i][i]
     * </pre>
     * </p>
     * <p>
     * Time complexity: O(n²) where n is the size of the system.<br>
     * Space complexity: O(n) for the solution vector.
     * </p>
     *
     * @param matrix       The lower triangular coefficient matrix L (size × size)
     * @param rightHandSide The right-hand side vector b (length = size)
     * @param size         The dimension of the linear system
     * @return The solution vector x (length = size)
     * @throws IllegalArgumentException if matrix is null, not lower triangular, has zero diagonal,
     *                                  contains non-finite values, or dimensions are invalid
     */
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

    /**
     * Solves an upper triangular linear system Ux = b using backward substitution.
     * <p>
     * Given an upper triangular matrix U and a right-hand side vector b, this method
     * computes the solution vector x such that Ux = b. The algorithm proceeds from
     * the last equation to the first, substituting previously computed values.
     * </p>
     * <p>
     * The algorithm works as follows: for each row i (from size-1 to 0):
     * <pre>
     * x[i] = (b[i] - sum(U[i][j] * x[j] for j = i+1 to size-1)) / U[i][i]
     * </pre>
     * </p>
     * <p>
     * Time complexity: O(n²) where n is the size of the system.<br>
     * Space complexity: O(n) for the solution vector.
     * </p>
     *
     * @param matrix       The upper triangular coefficient matrix U (size × size)
     * @param rightHandSide The right-hand side vector b (length = size)
     * @param size         The dimension of the linear system
     * @return The solution vector x (length = size)
     * @throws IllegalArgumentException if matrix is null, not upper triangular, has zero diagonal,
     *                                  contains non-finite values, or dimensions are invalid
     */
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

    /**
     * Verifies the accuracy of a solution to a linear system Ax = b.
     * <p>
     * This method checks whether the computed solution x satisfies the equation Ax = b
     * within a relative error tolerance. It computes the relative residual:
     * <pre>
     * relative_residual = max_i(|Ax - b|_i) / max_i(sum_j(|A[i][j]| * |x[j]|) + |b[i]|)
     * </pre>
     * and returns true if it is less than or equal to 1e-5.
     * </p>
     * <p>
     * This validation method is numerically stable as it uses:
     * <ul>
     *   <li>Double precision arithmetic for intermediate calculations</li>
     *   <li>Relative error scaling to handle systems of varying magnitudes</li>
     *   <li>Row-wise error bounds to account for accumulated floating-point errors</li>
     * </ul>
     * </p>
     * <p>
     * Time complexity: O(n²) where n is the size of the system.<br>
     * Space complexity: O(1) (constant space).
     * </p>
     *
     * @param matrix       The coefficient matrix A (size × size)
     * @param solution     The solution vector x to verify (length = size)
     * @param rightHandSide The right-hand side vector b (length = size)
     * @param size         The dimension of the linear system
     * @return true if the solution is accurate within relative tolerance of 1e-5, false otherwise
     * @throws IllegalArgumentException if any input is null, contains non-finite values,
     *                                  or dimensions are invalid
     */
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
