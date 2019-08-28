package test.neural_network.statistic_analyze;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

/**
 * Статистический анализ MNIST
 * (Для сравнения эффективности с нейросетями. Задуман как нечуствительный к сдвигу и масштабу (и с поиском по неподготовленному изображению))
 * А получилось - как всегда.. неожиданно
 * 
 * @author a.palkin
 *
 */
public class StaticAnalyzer {

	private String MNIST_BASE_PATH = "/home/legioner/MNIST/";
	private final int image_height = 28;
	private final int image_width = 28;
	
	
	private final int grain_size_percent_x = 2;
	private final int grain_size_percent_y = 2;
	private final int min_block_size = 2;
	private final static int color_similarity_percent = 25;
	
	public enum Avarege_type {
		avg,
		quiadratic,
		max,
		min
	}
	
	private final Avarege_type avg_type = Avarege_type.quiadratic; 
	
	public StaticAnalyzer(String basePath) {
		MNIST_BASE_PATH = basePath;
	}
	
	public static class NumberStatDefinition {
		private Integer label;
		private Double colorPercentage = 0.0d;
		private Double dX = 0.0d;
		private Double dY = 0.0d;
		private Double dX_min = Double.MAX_VALUE;
		private Double dY_min = Double.MAX_VALUE;
		private Double dX_max = 0.0d;
		private Double dY_max = 0.0d;
		public NumberStatDefinition() {}
		public Integer getLabel() {
			return label;
		}
		public void setLabel(Integer label) {
			this.label = label;
		}
		public Double getColorPercentage() {
			return colorPercentage;
		}
		public void setColorPercentage(Double colorPercentage) {
			this.colorPercentage = colorPercentage;
		}
		public Double getdX() {
			return dX;
		}
		public void setdX(Double dX) {
			this.dX = dX;
		}
		public Double getdY() {
			return dY;
		}
		public void setdY(Double dY) {
			this.dY = dY;
		}
		public Double getdX_min() {
			return dX_min;
		}
		public void setdX_min(Double dX_min) {
			this.dX_min = dX_min;
		}
		public Double getdY_min() {
			return dY_min;
		}
		public void setdY_min(Double dY_min) {
			this.dY_min = dY_min;
		}
		public Double getdX_max() {
			return dX_max;
		}
		public void setdX_max(Double dX_max) {
			this.dX_max = dX_max;
		}
		public Double getdY_max() {
			return dY_max;
		}
		public void setdY_max(Double dY_max) {
			this.dY_max = dY_max;
		}
		
	}
	
	public static class BlockData {
		private int x;
		private int y;
		private int avg_color;
		public BlockData(int x, int y, int avg_color) {
			super();
			this.x = x;
			this.y = y;
			this.avg_color = avg_color;
		}
		public int getX() {
			return x;
		}
		public void setX(int x) {
			this.x = x;
		}
		public int getY() {
			return y;
		}
		public void setY(int y) {
			this.y = y;
		}
		public int getAvg_color() {
			return avg_color;
		}
		public void setAvg_color(int avg_color) {
			this.avg_color = avg_color;
		}
	}
	
	public static class ColorSimilarity {
		private int color;
		public ColorSimilarity(int color) {
			this.color = color;
		}
		public int getColor() {
			return this.color;
		}
		@Override
		public boolean equals(Object obj) {
			if (!(obj instanceof ColorSimilarity)) return false;
			if (Math.abs(color  - ((ColorSimilarity)obj).getColor()) > color_similarity_percent) return false;
			return true;
		}
		@Override
		public int hashCode() {
			return new Integer(0).hashCode();
		}
	}
	
	public List<NumberStatDefinition> analyze() throws IOException {
		List<NumberStatDefinition> result = new ArrayList<>();
		File basePath = new File(MNIST_BASE_PATH);
		File[] folders = basePath.listFiles();
		for (File fNumber : folders) {
			if (!fNumber.isDirectory()) continue;
			System.out.println("Starting work statistic on number " + fNumber.getName());
			File[] files = fNumber.listFiles();
			NumberStatDefinition numberClass =  new NumberStatDefinition();
			numberClass.setLabel(Integer.parseInt(fNumber.getName()));
			int fileCount = 0;
			for (File f : files) {
				NumberStatDefinition number = analyzeFile(f, numberClass.getLabel(), null, null, null);
				fileCount++;
				numberClass.setdX(numberClass.getdX() + number.getdX());
				numberClass.setdY(numberClass.getdY() + number.getdY());
				numberClass.setColorPercentage(numberClass.getColorPercentage() + number.getColorPercentage());
				if (numberClass.getdX_min() > number.getdX()) numberClass.setdX_min(number.getdX());
				if (numberClass.getdX_max() < number.getdX()) numberClass.setdX_max(number.getdX());
				if (numberClass.getdY_min() > number.getdY()) numberClass.setdY_min(number.getdY());
				if (numberClass.getdY_max() < number.getdY()) numberClass.setdY_max(number.getdY());
			}
			numberClass.setdX(numberClass.getdX() / ((double)fileCount));
			numberClass.setdY(numberClass.getdY() / ((double)fileCount));
			numberClass.setColorPercentage(numberClass.getColorPercentage() / ((double)fileCount));
			result.add(numberClass);
			System.out.println("statistic on number " + fNumber.getName() + " done");
		}
		return result;
	}
	
	
	public NumberStatDefinition analyzeFile(File f, Integer label, int[] image, Integer w, Integer h) throws IOException {
		BufferedImage img = null;
		int width = w;
		int height = h;
		int[] imageData = image;  
		if (image == null) {
			img = ImageIO.read(f); 
			width = img.getWidth();
			height = img.getHeight();
			imageData = new int[width * height];
			img.getRaster().getPixels(0, 0, width, height, imageData);
		}
		//Центр масс считаем так:
		//1. Находим фоновый цвет (максимальное число пикселей с таким цветом)
		//2. Находим верхную левую точку
		//3. Находим правую нижнюю точку
		//4. Считаем центр масс как сумма (координата * вес (разница цвета с фоном)) / сумма разностей весов
		//
		//Результаты - надо признать - не впечатляют (максимальный и минимальные значения в 1,5 раза отличаются от среднего,
		//так что определение по сравнению со средним дает никакую точность). 
		//Думаю метод был бы пригоден для печатных цифр, но никак не для рукописных
		//Вероятно нужен фильтр (дополнительная нормальзация изображений) 
		Map<ColorSimilarity, List<BlockData>> map = new HashMap<>();
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				ColorSimilarity block = new ColorSimilarity(imageData[x * width + y]);
				List<BlockData> data = null;
				if ((data = map.get(block)) == null) {
					data = new ArrayList<>();
					map.put(block, data);
				}
				data.add(new BlockData(x, y, block.getColor()));
			}
		}
		List<Integer> sizes = map.values().stream().map(List::size).sorted().collect(Collectors.toList());
		ColorSimilarity background =  map.keySet().stream()
													.filter(c -> map.get(c).size() == sizes.get(sizes.size() - 1))
													.findFirst().get();
		map.remove(background);
		int[] left_top = {width, height};
		int[] right_bottom = {0, 0};
		map.forEach((s, b) -> {
			b.forEach(d -> {
				if (left_top[0] > d.x) left_top[0] = d.x;
				if (left_top[1] > d.y) left_top[1] = d.y;
				if (right_bottom[0] < d.x) right_bottom[0] = d.x;
				if (right_bottom[1] < d.y) right_bottom[1] = d.y;
			});
		});
		double[] sums = {0, 0, 0};
//		map.forEach((s, b) -> {
//			b.stream().filter(d -> d.avg_color < color_similarity_percent).forEach(d -> {
//				sums[0] = sums[0] + ((double)(255 - d.avg_color))/255d; //Вообще - тут надо использовать background (пока не придумал - как)
//				sums[1] = sums[1] + (((double)(255 - d.avg_color)) * (d.x - left_top[0]) / 255d);
//				sums[2] = sums[2] + (((double)(255 - d.avg_color)) * (d.y - left_top[1]) / 255d);
//			});
//		});
		map.forEach((s, b) -> {
			b.forEach(d -> {
				sums[0] = sums[0] + 1d; //Вообще - тут надо использовать background (пока не придумал - как)
				sums[1] = sums[1] + (d.x - left_top[0]);
				sums[2] = sums[2] + (d.y - left_top[1]);
			});
		});
		double center_x = sums[1] / sums[0];
		double center_y = sums[2] / sums[0];
		double diff_y = (double)(right_bottom[1] - left_top[1] + 1);
		double diff_x = (double)(right_bottom[0] - left_top[0] + 1);
		NumberStatDefinition definition = new NumberStatDefinition();
		definition.setLabel(label);
		definition.setdX(center_x / diff_x);
		definition.setdY(center_y / diff_y);
		definition.setColorPercentage(sums[0] / (diff_x * diff_y));
		return definition;
	}
	
}
 