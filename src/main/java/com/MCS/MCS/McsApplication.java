package com.MCS.MCS;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Main application class for the MCS (Methods for Computational Science) project.
 * <p>
 * This Spring Boot application provides implementations of numerical methods
 * for linear algebra and linear systems, including triangular system solvers
 * and solution verification.
 *
 * @author MCS
 * @version 0.0.1-SNAPSHOT
 * @since 1.0
 */
@SpringBootApplication
public class McsApplication {

	/**
	 * Entry point for the Spring Boot application.
	 * <p>
	 * Initializes and starts the Spring application context.
	 *
	 * @param args Command-line arguments passed to the application
	 */
	public static void main(String[] args) {
		SpringApplication.run(McsApplication.class, args);
	}

}
