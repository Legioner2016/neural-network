package test.neural_network.dl4j_moved;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestNetworkUsage {

	private final static String tempModelFile = "/opt/cms/network/lenetmnist_moved.zip";
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
			float[] imageData = new float[28 * 28];
			
			bi.getRaster().getPixels(0, 0, 28, 28, img);

			for (int j = 0; j < img.length; j++) {
				float v = 255 - (img[j] & 0xFF); //byte is loaded as signed -> convert to unsigned
				imageData[j] = v / 255.0f;
				featureVec[j] = 0f;
			}
			
			int skip_y = 0;
			for (int i = 0; i < imageData.length; i++) {
				if (imageData[i] > 0) {
					skip_y = (i / 28); 	
					break;
				}
			}
			int skip_x = 0;
			outer: for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					if (imageData[j * 28 + i] > 0) {
						skip_x = i; 		
						break outer;
					}
				}
			}
			for (int x = 0; x < 28; x++) {
				for (int y = 0; y < 28; y++) {
					if (x >= skip_x && y >= skip_y) {
						featureVec[(y - skip_y) * 28 + (x - skip_x)] = imageData[y * 28 + x]; 			
					}
				}
			}
			


		} catch (IOException e) {
			e.printStackTrace();
		}

		INDArray features = Nd4j.create(featureVec);
		return features;

	}

}
