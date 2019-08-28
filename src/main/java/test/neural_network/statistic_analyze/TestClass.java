package test.neural_network.statistic_analyze;

import java.io.IOException;
import java.util.List;
import test.neural_network.statistic_analyze.StaticAnalyzer.NumberStatDefinition;


public class TestClass {

	public static void main(String[] args) {
		StaticAnalyzer analyze = new StaticAnalyzer("/home/legioner/MNIST/");
		try {
			List<NumberStatDefinition> result = analyze.analyze();
			System.out.println("\n");
			result.forEach(r -> {
				System.out.println(String.format("number: %d have dx = %.6f; dy = %.6f; dx_max = %.6f; dx_min = %.6f; cp = %.6f", 
								r.getLabel(), r.getdX(), r.getdY(), r.getdX_max(), r.getdX_min(), r.getColorPercentage()));
			});
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}

	
}
