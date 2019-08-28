package test.neural_network.self_write_direct_connected;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class MatrixBiasesAndWeights {
	private SimpleMatrix biases;
	private SimpleMatrix weights;
	private SimpleMatrix sigmaargument;
	private SimpleMatrix activations;
	private SimpleMatrix deltaBiases;
	private SimpleMatrix deltaWeights;
	public MatrixBiasesAndWeights (int rows, int columns, boolean random, boolean learning) {
		if (learning) {
			activations = new SimpleMatrix(rows, 1);
			sigmaargument = new SimpleMatrix(rows, 1);
			deltaBiases = new SimpleMatrix(rows, 1);
			deltaWeights = new SimpleMatrix(rows, columns);
		} 
		if (random) {
			biases = SimpleMatrix.random(rows, 1, -1d, 1d, new Random());
			weights = SimpleMatrix.random(rows, columns, -1d, 1d, new Random());
		} else {
			biases = new SimpleMatrix(rows, 1);
			weights = new SimpleMatrix(rows, columns);
		}
	}
	public MatrixBiasesAndWeights (boolean zero, int rows, int columns) {
		if (zero) {
			double[] values = new double[rows];
			Arrays.fill(values, 0d);
			biases = new SimpleMatrix(rows, 1, true, values);
			double[] values_ = new double[rows * columns];
			Arrays.fill(values_, 0d);
			weights = new SimpleMatrix(rows, columns, true, values_);
		} else {
			biases = new SimpleMatrix(rows, 1);
			weights = new SimpleMatrix(rows, columns);
		}
	}
	public SimpleMatrix getBiases() {
		return biases;
	}
	public void setBiases(SimpleMatrix biases) {
		this.biases = biases;
	}
	public SimpleMatrix getWeights() {
		return weights;
	}
	public void setWeights(SimpleMatrix weights) {
		this.weights = weights;
	}
	public SimpleMatrix getSigmaargument() {
		return sigmaargument;
	}
	public void setSigmaargument(SimpleMatrix sigmaargument) {
		this.sigmaargument = sigmaargument;
	}
	public SimpleMatrix getActivations() {
		return activations;
	}
	public void setActivations(SimpleMatrix activations) {
		this.activations = activations;
	}
	public SimpleMatrix getDeltaBiases() {
		return deltaBiases;
	}
	public SimpleMatrix getDeltaWeights() {
		return deltaWeights;
	}
	public void setDeltaBiases(SimpleMatrix deltaBiases) {
		this.deltaBiases = deltaBiases;
	}
	public void setDeltaWeights(SimpleMatrix deltaWeights) {
		this.deltaWeights = deltaWeights;
	}
	public void zeroDeltaMatrix() {
		deltaBiases.set(0d);
		deltaWeights.set(0d);
	}
	public List<String> getWeightsStringRepresentation() {
		List<String> result = new ArrayList<>();
		for (int i = 0; i < weights.numRows(); i++) {
			String row = "";
			for (int j = 0; j < weights.numCols(); j++) {
				row += weights.get(i, j);
				if (j < weights.numCols() - 1) row += "\t"; 
			}
			result.add(row);
		}
		return result;
	}
	public List<String> getBiasesStringRepresentation() {
		List<String> result = new ArrayList<>();
		for (int i = 0; i < biases.numRows(); i++) {
			String row = "";
			for (int j = 0; j < biases.numCols(); j++) {
				row += biases.get(i, j);
				if (j < biases.numCols() - 1) row += "\t"; 
			}
			result.add(row);
		}
		return result;
	}
}
