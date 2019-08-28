package test.neural_network.self_write_direct_connected;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.ejml.simple.SimpleMatrix;

public class MNISTNeuroNetTest {
	
	private final static String paramsFile = "paramsMNIST5.net"; 
	private static final String MNIST_path = "/home/legioner/MNIST/";
	private static NeuroNetwork network;

	public static void main(String[] args) throws IOException {
		//прочитать параметры нейросети
		network = NeuroNetwork.loadFromResourceFile(paramsFile);
		try {
			//Подготовить тренировочные данные
			System.out.println("preparing training data");
			//Забыл сам у себя спросить - а почему я тестирую на тренировочных, а не на проверочных данных?
			List<MNISTData> trainingSet = new ArrayList<>(60000);
			for (int i = 0; i < 10; i++) {
				String tempPath = MNIST_path + i + "/";
				File tempFile = new File(tempPath);
				File[] files = tempFile.listFiles();
				for (File f : files) trainingSet.add(new MNISTData(f)); 
			}
			System.out.println("training set ready");
			
			//Проверка
			int good = 0, invalid = 0, unknown2 = 0, unknown1 = 0;
			for (int i = 0; i < trainingSet.size(); i++) {
				SimpleMatrix resultActivation = network.feedForward(trainingSet.get(i).getInputMatrix());
				boolean found = false;
				for (int j = 0; j < resultActivation.numRows(); j++) {
					if (resultActivation.get(j , 0) > 0.5f) {
						found = true;
						if (trainingSet.get(i).getResult() == j) {
							good++;
						}
						else invalid++;
						break;
					}
				}
				if (!found) unknown1++;
			}
			
			System.out.println("All = " + trainingSet.size());
			System.out.println("Goods = " + good);
			System.out.println("Invalid = " + invalid);
			System.out.println("Undefined = " + (unknown2 +  unknown1));

		} catch (Exception e) {
			e.printStackTrace();
		}


	}
	
}