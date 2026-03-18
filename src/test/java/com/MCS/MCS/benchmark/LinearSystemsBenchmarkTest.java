package com.MCS.MCS.benchmark;

import com.MCS.MCS.LinearSystems.LinearSystems;
import org.openjdk.jmh.annotations.*;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.concurrent.TimeUnit;


@SpringBootTest
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class LinearSystemsBenchmarkTest extends AbstractBenchmark {
    float [][] positiveDefinitiveSparseMatrix;
    float [][] positiveDefiniteMatrix;
    float [][] genericMatrix;
    float [] b;
    int size;


    @Setup
    public void setup() {
        // Initialize any resources needed for the benchmark
        genericMatrix = new float[][]{
                {3, -1, 4, 2, 5, -2, 6, 1, -3, 2},
                {2, 5, -2, 3, 1, 4, -1, 6, 2, -4},
                {1, 3, 6, -2, 4, 2, 5, -1, 3, 1},
                {4, 2, 1, 5, -3, 6, 2, 3, -2, 4},
                {5, -2, 3, 1, 6, 2, 4, -3, 1, 5},
                {2, 4, -1, 6, 3, 5, -2, 1, 4, 2},
                {6, 1, 5, 2, 4, -1, 3, 2, 6, -2},
                {1, 6, 2, 4, -2, 3, 1, 5, 2, 6},
                {3, 2, 4, -1, 5, 1, 2, 6, 4, 3},
                {2, 3, 1, 5, 2, 6, 4, 3, 1, 5}
        };
        positiveDefiniteMatrix= new float[][]{
                {15, 2, 3, 1, 4, 2, 1, 3, 2, 1},
                {2, 14, 2, 3, 1, 4, 2, 1, 3, 2},
                {3, 2, 16, 2, 3, 1, 4, 2, 1, 3},
                {1, 3, 2, 15, 2, 3, 1, 4, 2, 1},
                {4, 1, 3, 2, 17, 2, 3, 1, 4, 2},
                {2, 4, 1, 3, 2, 16, 2, 3, 1, 4},
                {1, 2, 4, 1, 3, 2, 15, 2, 3, 1},
                {3, 1, 2, 4, 1, 3, 2, 16, 2, 3},
                {2, 3, 1, 2, 4, 1, 3, 2, 15, 2},
                {1, 2, 3, 1, 2, 4, 1, 3, 2, 14}
        };
        /*
        positiveDefinitiveSparseMatrix = new float[][]{
                {4, 0, 0},
                {0, 3, 0},
                {0, 0, 5}
        };
        */

        size = 10;
        b = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    }

    @Benchmark
    public void benchmarkDefinitePositivePLU() {
        // Code to solve a linear system using the generic matrix using PLU decomposition
        LinearSystems.resolveGenericPLU(positiveDefiniteMatrix,b,size);

    }
    @Benchmark
    public void benchmarkDefinitePositiveCholesky() {
        // Code to solve a linear system using the generic matrix using PLU decomposition
        LinearSystems.resolveCholesky(positiveDefiniteMatrix,b,size);

    }
}
