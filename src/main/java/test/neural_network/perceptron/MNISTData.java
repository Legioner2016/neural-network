package test.neural_network.perceptron;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

/**
 * Класс, представляющий данные одного изображения MNIST для обработки
 * 
 * @author legioner
 *
 */
public class MNISTData {

	private int width;
	private int height;
	private boolean[] imageData;
	private Integer result; //Какое значение предполагается
	
	public MNISTData(File file) {
		String temp = file.getParent();
		temp = temp.substring(temp.lastIndexOf("/") + 1);
		result = Integer.parseInt(temp);
		BufferedImage bi;
		try {
			bi = ImageIO.read(file);
			this.width = bi.getWidth();
			this.height = bi.getHeight();
			this.imageData = new boolean[this.width * this.height];
			Arrays.fill(imageData, false);
			double[] tmp = new double[this.width * this.height];
			bi.getRaster().getPixels(0, 0, this.width, this.height, tmp);
			for (int i = 0; i < this.imageData.length; i++) tmp[i] = (255d - tmp[i])/255d;
			int skip_y = 0;
			for (int i = 0; i < tmp.length; i++) {
				if (tmp[i] > 0) {
					skip_y = (i / width); 	
					break;
				}
			}
			int skip_x = 0;
			outer: for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					if (tmp[j * width + i] > 0) {
						skip_x = i; 		
						break outer;
					}
				}
			}
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					if (x >= skip_x && y >= skip_y) {
						this.imageData[(y - skip_y) * width + (x - skip_x)] = tmp[y * width + x] > 0.2; 			
					}
				}
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public boolean[] getInputs() {
		return this.imageData;
	}
	
	public Integer getResult() {
		return this.result;
	}

	
}
