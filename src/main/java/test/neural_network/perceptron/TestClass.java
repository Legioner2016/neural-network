package test.neural_network.perceptron;

import java.io.File;
import java.io.FileNotFoundException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;


public class TestClass {

	private final static String paramsFile = "perceptronMNIST1_%d.net";
	private static final String MNIST_path = "/home/legioner/MNIST/";

	public static void main(String[] args) {
		try {
			//прочитать параметры нейросети
			ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
			Perceptron[] perceprtons = new Perceptron[10];
			for (int i = 0; i < 10; i++) {
				URL fileUrl = classLoader.getResource(String.format(paramsFile, i));
				perceprtons[i] = Perceptron.loadFromFile(new File(fileUrl.getFile()));
			}
			System.out.println("perceptrons have been loaded");

			//Подготовить тренировочные данные
			System.out.println("preparing training data");
			List<MNISTData> trainingSet = new ArrayList<>(60000);
			for (int i = 0; i < 10; i++) {
				String tempPath = MNIST_path + i + "/";
				File tempFile = new File(tempPath);
				File[] files = tempFile.listFiles();
				for (File f : files) trainingSet.add(new MNISTData(f)); 
			}
			System.out.println("training set ready");

			//Проверка
			int good = 0, invalid = 0, unknown = 0;
			for (int i = 0; i < trainingSet.size(); i++) {
				boolean found = false;
				for (int j = 0; j < 10; j++) {
					if (perceprtons[j].feedForward(trainingSet.get(i).getInputs())) {
						if (j == trainingSet.get(i).getResult()) good++;
						else invalid++;
						found = true;
						break;
					}
				}
				if (!found) unknown++;
			}

			System.out.println("All = " + trainingSet.size());
			System.out.println("Goods = " + good);
			System.out.println("Invalid = " + invalid);
			System.out.println("Undefined = " + unknown);

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}


}
