package test.neural_network.dl4j_moved;

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
	private float[] array;
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
			float[] imageData = new float[this.width * this.height];
			array = new float[this.width * this.height];
			Arrays.fill(array, 0f);
			int[] tmp = new int[this.width * this.height];
			bi.getRaster().getPixels(0, 0, this.width, this.height, tmp);
			for (int j = 0; j < tmp.length; j++) {
					float v = 255 - (tmp[j] & 0xFF); //byte is loaded as signed -> convert to unsigned
					imageData[j] = v / 255.0f;
			}
			int skip_y = 0;
			for (int i = 0; i < tmp.length; i++) {
				if (imageData[i] > 0) {
					skip_y = (i / width); 	
					break;
				}
			}
			int skip_x = 0;
			outer: for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					if (imageData[j * width + i] > 0) {
						skip_x = i; 		
						break outer;
					}
				}
			}
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					if (x >= skip_x && y >= skip_y) {
						array[(y - skip_y) * width + (x - skip_x)] = imageData[y * width + x]; 			
					}
				}
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public float[] getInputs() {
		return this.array;
	}
	
	public Integer getResult() {
		return this.result;
	}

	
}
