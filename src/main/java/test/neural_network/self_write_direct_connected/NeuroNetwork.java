package test.neural_network.self_write_direct_connected;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

public class NeuroNetwork {

	private List<MatrixBiasesAndWeights> network;
	private int[] layers_sizes;
	private int layers_sizes_length;

	public NeuroNetwork(int[] layers_sizes, int layers_sizes_length) {
		this.layers_sizes = layers_sizes; 
		this.layers_sizes_length = layers_sizes_length;
		
		network = new ArrayList<>();
		for (int i = 1; i < layers_sizes.length; i++) {
			network.add(new MatrixBiasesAndWeights(layers_sizes[i], layers_sizes[i - 1], true, true));
		}
	}
	
	private NeuroNetwork(int[] layers_sizes, int layers_sizes_length, List<MatrixBiasesAndWeights> network) {
		this.network = network;
		this.layers_sizes = layers_sizes;
		this.layers_sizes_length = layers_sizes_length;
	}



	public List<MatrixBiasesAndWeights> getNetwork() {
		return network;
	}

	public int[] getLayers_sizes() {
		return layers_sizes;
	}

	public int getLayers_sizes_length() {
		return layers_sizes_length;
	}
	
	public static  NeuroNetwork loadFromResourceFile(String paramsFileName) {
		ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
		URL fileUrl = classLoader.getResource(paramsFileName);
		int linePos = 0;
		List<String> lines = null;
		try {
			lines =  Files.readAllLines(Paths.get((new File(fileUrl.getFile())).toURI()));
			String line = lines.get(linePos++);
			int layers_sizes_length = Integer.parseInt(line);
			int[] layers_sizes = new int[layers_sizes_length];
			line = lines.get(linePos++);
			String[] data = line.split("\t");
			for (int i = 0 ; i < layers_sizes_length; i++) {
				layers_sizes[i] = Integer.parseInt(data[i]);
			}
			line = lines.get(linePos++);
			line = lines.get(linePos++);
			line = lines.get(linePos++);
			List<MatrixBiasesAndWeights> network = new ArrayList<>();
			for (int i = 1; i < layers_sizes.length; i++) {
				network.add(new MatrixBiasesAndWeights(true, layers_sizes[i], layers_sizes[i - 1]));
			}
			for (MatrixBiasesAndWeights matrix : network) {
				for (int k = 0; k < matrix.getWeights().numRows(); k++) {
					String lineW = lines.get(linePos++);
					String[] dataW = lineW.split("\t");
					for (int u = 0; u < matrix.getWeights().numCols(); u++) {
						matrix.getWeights().set(k, u, new Float(dataW[u]));
					}					
				}
				for (int k = 0; k < matrix.getBiases().numRows(); k++) {
					String lineB = lines.get(linePos++);
					String[] dataB = lineB.split("\t");
					for (int u = 0; u < matrix.getBiases().numCols(); u++) {
						matrix.getBiases().set(k, u, new Float(dataB[u]));
					}					
				}
			}
			
			return new NeuroNetwork(layers_sizes, layers_sizes_length, network);
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e1) {
			e1.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null;
	
	}
	
	
	public SimpleMatrix feedForward(SimpleMatrix inputColumn) {
		SimpleMatrix result = inputColumn;
		for (int i = 1; i < layers_sizes_length; i++) {
			result = network.get(i - 1).getWeights().mult(result).plus(network.get(i - 1).getBiases());
			for (int j = 0; j < result.numRows(); j++) {
				result.set(j, 0, sigmoidD(result.get(j, 0)));
			}
		}
		return result;
	}
	
	private double sigmoidD(double value) {
		return (1d/(1d + (float)Math.exp(-value)));
	}

	
	

}
