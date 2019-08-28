package test.neural_network.self_write_direct_connected;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

import org.ejml.simple.SimpleMatrix;

/**
 * Класс, представляющий данные одного изображения MNIST для обработки
 * 
 * @author legioner
 *
 */
public class MNISTData {

	private int width;
	private int height;
	private double[] imageData;
	private String fileName; //На всякий случай (вообще-то на случай отладки)
	private Integer fileNumber; //Имя файла числом
	private Integer result; //Какое значение предполагается
	
	public MNISTData(File file) {
		fileName = file.getName();
		if (fileName.endsWith(".png")) fileNumber = Integer.parseInt(fileName.substring(0, fileName.length() - ".png".length()));
		else fileNumber = Integer.parseInt(fileName);
		String temp = file.getParent();
		temp = temp.substring(temp.lastIndexOf("/") + 1);
		result = Integer.parseInt(temp);
		BufferedImage bi;
		try {
			bi = ImageIO.read(file);
			this.width = bi.getWidth();
			this.height = bi.getHeight();
			double[] tmp = new double[this.width * this.height];
			bi.getRaster().getPixels(0, 0, this.width, this.height, tmp);
			for (int i = 0; i < tmp.length; i++) tmp[i] = (255d - tmp[i])/255d;
			this.imageData = tmp;
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public SimpleMatrix getOutputMatrix() {
		double[] values = new double[10];
		Arrays.fill(values, 0d);
		SimpleMatrix result = new SimpleMatrix(10, 1, true, values);
		result.set(this.result, 0, 1d);
		return result;
	}

	public SimpleMatrix getInputMatrix() {
		return new SimpleMatrix(imageData.length, 1, true, imageData);
	}
	
	public Integer getResult() {
		return this.result;
	}

	
}
