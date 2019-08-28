package test.neural_network.statistic_analyze;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import test.neural_network.statistic_analyze.StaticAnalyzer.NumberStatDefinition;

public class TestRecognation {

	private static class NumberData {
		private Integer label;
		private Double colorPercent;
		private Double dX;
		private Double dY;
		public NumberData(Integer label, Double colorPercent, Double dX, Double dY) {
			super();
			this.label = label;
			this.colorPercent = colorPercent;
			this.dX = dX;
			this.dY = dY;
		}
		public Integer getLabel() {
			return label;
		}
		public Double getdX() {
			return dX;
		}
		public Double getdY() {
			return dY;
		}
		public Double getColorPercent() {
			return colorPercent;
		}
	}
	
	private final static List<NumberData> datas = new ArrayList<NumberData>() {
		{
			add(new NumberData(0, 0.509680, 0.486962, 0.470855));
			add(new NumberData(1, 0.493206, 0.480739, 0.432105));
			add(new NumberData(2, 0.445153, 0.519413, 0.469194));
			add(new NumberData(3, 0.461883, 0.473083, 0.523363));
			add(new NumberData(4, 0.403952, 0.449353, 0.477140));
			add(new NumberData(5, 0.412135, 0.460978, 0.438639));
			add(new NumberData(6, 0.479513, 0.569268, 0.431180));
			add(new NumberData(7, 0.384241, 0.364948, 0.515890));
			add(new NumberData(8, 0.512770, 0.458450, 0.458450));
			add(new NumberData(9, 0.464017, 0.390049, 0.490041));
		}
	};
	
	private final static double relevant_percent = 0.25;
	
	private final static String fileToTest = "/home/legioner/MNIST/7/15425.png";
	private final static String filesToTest = "/home/legioner/MNIST/9/";	
	
	public static void main(String[] args) throws IOException {
		
		StaticAnalyzer analyze = new StaticAnalyzer(null);
//		NumberStatDefinition test = analyze.analyzeFile(new File(fileToTest), null);
//		
//		System.out.println(fileToTest + " determinates to " + getNumber(test));

		
		int all = 0, good = 0, bad = 0;
		File[] files = (new File(filesToTest)).listFiles();
		for (File f : files) {
			NumberStatDefinition test = analyze.analyzeFile(f, null, null, null, null);
//			System.out.println(getNumber(test));
			all++;
			if (getNumber(test) == 9) good++;
			else bad++; 
		}
		System.out.println("from " + all + " files " + good + " determinated right, " + bad + " determinated incorrect. (" + 
									 (100 * good/all) + "%)");		
	}
	
	private static int getNumber(NumberStatDefinition test) {
		Map<Double, Integer> selected = datas.stream()
				.collect(Collectors
						.toMap(nd -> Math.sqrt((test.getdX() - nd.getdX()) * (test.getdX() - nd.getdX()) +
								(test.getdY() - nd.getdY()) * (test.getdY() - nd.getdY()) + 
								(test.getColorPercentage() - nd.getColorPercent()) * (test.getColorPercentage() - nd.getColorPercent())), 
						nd -> nd.getLabel()));
		Optional<Double> minDiff = selected.keySet().stream().sorted().findFirst();
		return selected.get(minDiff.get());
	}

}
