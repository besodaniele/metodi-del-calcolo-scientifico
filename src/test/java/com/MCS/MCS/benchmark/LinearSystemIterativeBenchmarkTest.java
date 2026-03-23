package com.MCS.MCS.benchmark;

import com.MCS.MCS.LinearSystems.LinearSystems;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.concurrent.TimeUnit;

@SpringBootTest
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class LinearSystemIterativeBenchmarkTest extends AbstractBenchmark{
    float [][] positiveDefinitiveSparseMatrix;
    float [][] positiveDefiniteMatrix;
    float [][] genericMatrix;
    float [][] diagonalDominantMatrix;
    float [] b;
    int size;

    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.EVENTS)
    public static class IterationCounters {
        public int iterations;

        @Setup(Level.Invocation)
        public void reset() {
            iterations = 0;
        }
    }


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
        //
        diagonalDominantMatrix = new float[][]{
                {50, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                {1, 51, 1, 2, 3, 4, 5, 6, 7, 8},
                {2, 1, 52, 1, 2, 3, 4, 5, 6, 7},
                {3, 2, 1, 53, 1, 2, 3, 4, 5, 6},
                {4, 3, 2, 1, 54, 1, 2, 3, 4, 5},
                {5, 4, 3, 2, 1, 55, 1, 2, 3, 4},
                {6, 5, 4, 3, 2, 1 ,56 ,1 ,2 ,3},
                {7 ,6 ,5 ,4 ,3 ,2 ,1 ,57 ,1 ,2},
                {8 ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,38 ,1},
                {9 ,8 ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,49}
        };
        size = 10;
        b = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    }
    @Benchmark
    public void benchmarkDiagonalDominantJacobi(IterationCounters counters, Blackhole blackhole) {
        // Code to solve a linear system using the generic matrix using PLU decomposition
        var result = LinearSystems.resolveJacobi(diagonalDominantMatrix,b,size);
        counters.iterations = result.getSecond();
        blackhole.consume(result.getFirst());

    }
    @Benchmark
    public void benchmarkDiagonalDominantGaussSeidel(IterationCounters counters, Blackhole blackhole) {
        // Code to solve a linear system using the generic matrix using PLU decomposition
        var result = LinearSystems.resolveGaussSeidel(diagonalDominantMatrix,b,size);
        counters.iterations = result.getSecond();
        blackhole.consume(result.getFirst());

    }
}
