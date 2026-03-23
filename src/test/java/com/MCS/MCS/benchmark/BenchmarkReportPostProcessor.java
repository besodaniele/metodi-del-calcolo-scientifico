package com.MCS.MCS.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public final class BenchmarkReportPostProcessor {

    private BenchmarkReportPostProcessor() {
    }

    public static void writeIterationsPerRunCsv(String jsonReportPath, String csvOutputPath) {
        Path inputPath = Paths.get(jsonReportPath);
        if (!Files.exists(inputPath)) {
            return;
        }

        ObjectMapper mapper = new ObjectMapper();
        try {
            JsonNode root = mapper.readTree(inputPath.toFile());
            if (!root.isArray()) {
                return;
            }

            List<String> lines = new ArrayList<>();
            lines.add("benchmark,run,iterations,time_us_per_op");

            for (JsonNode benchmarkNode : root) {
                String benchmarkName = benchmarkNode.path("benchmark").asText("unknown");
                JsonNode iterationsForkData = benchmarkNode.path("secondaryMetrics").path("iterations").path("rawData");
                JsonNode timeForkData = benchmarkNode.path("primaryMetric").path("rawData");

                if (!iterationsForkData.isArray() || iterationsForkData.isEmpty()) {
                    continue;
                }

                JsonNode iterationsRuns = iterationsForkData.get(0);
                JsonNode timeRuns = (timeForkData.isArray() && !timeForkData.isEmpty()) ? timeForkData.get(0) : null;

                for (int runIndex = 0; runIndex < iterationsRuns.size(); runIndex++) {
                    double iterations = iterationsRuns.get(runIndex).asDouble(Double.NaN);
                    double time = (timeRuns != null && runIndex < timeRuns.size())
                            ? timeRuns.get(runIndex).asDouble(Double.NaN)
                            : Double.NaN;

                    lines.add(csvEscape(benchmarkName)
                            + "," + (runIndex + 1)
                            + "," + iterations
                            + "," + time);
                }
            }

            Path outputPath = Paths.get(csvOutputPath);
            if (outputPath.getParent() != null) {
                Files.createDirectories(outputPath.getParent());
            }
            Files.write(outputPath, lines, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Unable to build iterations-per-run CSV report", e);
        }
    }

    private static String csvEscape(String value) {
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return '"' + value.replace("\"", "\"\"") + '"';
        }
        return value;
    }
}

