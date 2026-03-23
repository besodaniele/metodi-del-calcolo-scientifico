package com.MCS.MCS.benchmark;

import org.junit.jupiter.api.Test;
import org.openjdk.jmh.results.format.ResultFormatType;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

abstract public class AbstractBenchmark {

    private final static Integer MEASUREMENT_ITERATIONS = 3;
    private final static Integer WARMUP_ITERATIONS = 3;
    private static final String JMH_REPORT_PATH = "./report/report.json";
    private static final String ITERATIONS_PER_RUN_REPORT_PATH = "./report/iterations-per-run.csv";

    @Test
    public void executeJmhRunner() throws RunnerException {
        Options opt = new OptionsBuilder()
                // set the class name regex for benchmarks to search for to the current class
                .include("\\." + this.getClass().getSimpleName() + "\\.")
                .warmupIterations(WARMUP_ITERATIONS)
                .measurementIterations(MEASUREMENT_ITERATIONS)
                // do not use forking or the benchmark methods will not see references stored within its class
                .forks(0)
                // do not use multiple threads
                .threads(1)
                .shouldDoGC(true)
                .shouldFailOnError(true)
                .resultFormat(ResultFormatType.JSON)
                .result(JMH_REPORT_PATH)
                .shouldFailOnError(true)
                .jvmArgs("-server")
                .build();

        new Runner(opt).run();
        BenchmarkReportPostProcessor.writeIterationsPerRunCsv(JMH_REPORT_PATH, ITERATIONS_PER_RUN_REPORT_PATH);
    }
}