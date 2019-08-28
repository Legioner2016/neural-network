package test.neural_network.statistic_analyze;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import test.neural_network.statistic_analyze.StaticAnalyzer.NumberStatDefinition;


public class StaticRecognation {
	
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
		/**
		 * 
		 */
		private static final long serialVersionUID = 144804038304472363L;

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
	
	private StaticAnalyzer analyze;
	
	public StaticRecognation() {
		analyze = new StaticAnalyzer(null);
	}
	
	public int recognize(int[] img_)  {
		NumberStatDefinition test;
		Map<Double, Integer> selected;
		Optional<Double> minDiff;
		try {
			test = analyze.analyzeFile(null, null, img_, 28, 28);
			selected = datas.stream()
					.collect(Collectors
							.toMap(nd -> Math.sqrt((test.getdX() - nd.getdX()) * (test.getdX() - nd.getdX()) +
									(test.getdY() - nd.getdY()) * (test.getdY() - nd.getdY()) + 
									(test.getColorPercentage() - nd.getColorPercent()) * (test.getColorPercentage() - nd.getColorPercent())), 
							nd -> nd.getLabel()));
			minDiff = selected.keySet().stream().sorted().findFirst();
		}
		catch (IOException e) {
			e.printStackTrace();
			return -1;
		}		
		return selected.get(minDiff.get());
	}

}
