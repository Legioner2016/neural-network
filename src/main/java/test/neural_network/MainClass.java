package test.neural_network;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import javax.swing.JFrame;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import test.neural_network.mnist_ui_test.MainFrame;
import test.neural_network.perceptron.Perceptron;
import test.neural_network.self_write_direct_connected.NeuroNetwork;
import test.neural_network.statistic_analyze.StaticRecognation;


public class MainClass {

	private static final String modelFile = "lenetmnist.zip";
	private final static String paramsFile = "perceptronMNIST1_%d.net";
	private final static String paramsFile_2 = "paramsMNIST5.net";
	
	public static void main(String[] args) throws IOException {
		
		ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
		URL fileUrl = classLoader.getResource(modelFile);
		
		MultiLayerNetwork  model = MultiLayerNetwork.load(new File(fileUrl.getFile()), false);
		StaticRecognation stat = new StaticRecognation();
		NeuroNetwork network = NeuroNetwork.loadFromResourceFile(paramsFile_2);
		
		Perceptron[] perceprtons = new Perceptron[10];
		for (int i = 0; i < 10; i++) {
			fileUrl = classLoader.getResource(String.format(paramsFile, i));
			perceprtons[i] = Perceptron.loadFromFile(new File(fileUrl.getFile()));
		}
		
		MainFrame frame = new MainFrame(model, stat, perceprtons, network);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	}

}
