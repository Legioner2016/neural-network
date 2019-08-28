package test.neural_network.dl4j_origin;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestNetworkUsage {

	private final static String tempModelFile = "/opt/cms/network/lenetmnist.zip";
	private final static String testMNISTFile = "/home/legioner/MNIST/6/15084.png";

	public static void main(String[] args) {

		
		MultiLayerNetwork model = null;
		try {
			model = MultiLayerNetwork.load(new File(tempModelFile), false);	
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		 
		
		List<INDArray> activations = model.feedForward(prepareInputs());
		INDArray output = activations.get(activations.size() - 1);

		System.out.println(output.toString());

	}

	private static INDArray prepareInputs() {

		float[] featureVec = new float[28 * 28];
		BufferedImage bi;
		try {
			bi = ImageIO.read(new File(testMNISTFile));
			int[] img = new int[28 * 28];
			bi.getRaster().getPixels(0, 0, 28, 28, img);

			for (int j = 0; j < img.length; j++) {
				float v = 255 - (img[j] & 0xFF); //byte is loaded as signed -> convert to unsigned
				featureVec[j] = v / 255.0f;
			}


		} catch (IOException e) {
			e.printStackTrace();
		}

		INDArray features = Nd4j.create(featureVec);
		return features;

	}

}
