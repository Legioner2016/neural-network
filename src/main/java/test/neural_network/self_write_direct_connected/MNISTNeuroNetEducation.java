package test.neural_network.self_write_direct_connected;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.ejml.simple.SimpleMatrix;

public class MNISTNeuroNetEducation {
	
	
	private static final int[] layers_sizes = new int[] {784, 24, 10};
	private static List<MatrixBiasesAndWeights> network;
	private static final float learningSpeed = 10.0f;
	private static final int mini_batch_size = 30;
	private static final int epochs_count = 40;
	private static final String MNIST_path = "/home/legioner/MNIST/";
	
	private static Function<MNISTData, SimpleMatrix> costFunction;
	private static final float regularization_parameter = 1.0f;
	private static final float test_set_size = 60000f;
	private static final float regularization_constant = (1f - (learningSpeed * (regularization_parameter / test_set_size)));
	private static BiConsumer<MatrixBiasesAndWeights, MatrixBiasesAndWeights> weightUpdater;
	
	public static void main(String[] args) {
		//Матрицы сдвигов и весов
		//столбцы - связь с предыдущим слоем
		network = new ArrayList<>();
		for (int i = 1; i < layers_sizes.length; i++) {
			network.add(new MatrixBiasesAndWeights(layers_sizes[i], layers_sizes[i - 1], true, true));
		}
		
		//Оптимизация сети - функция стоимости для выходного слоя
		costFunction = MNISTNeuroNetEducation::costFunctionQuadratic;
//		costFunction = MNISTNeuroNet::costFunctionLog;
		//Обновление весов с учетом (или без) регулязации
		weightUpdater = MNISTNeuroNetEducation::updateWeightWithoutRegulization;
//		weightUpdater = MNISTNeuroNet::updateWeightWithRegulization;
		
		//Подготовить тренировочные данные
		System.out.println("preparing training data");
		File file = new File("");
		List<MNISTData> trainingSet = new ArrayList<>(60000);
		for (int i = 0; i < 10; i++) {
			String tempPath = MNIST_path + i + "/";
			File tempFile = new File(tempPath);
			File[] files = tempFile.listFiles();
			for (File f : files) trainingSet.add(new MNISTData(f)); 
		}
		System.out.println("training set ready");

		//обучение
		for (int epoch = 0; epoch < epochs_count; epoch++) {
			Collections.shuffle(trainingSet);
			
			//Пакеты по 30 наборов вход- выход
			int start = 0;
			while (start < trainingSet.size()) {
				int end = start + mini_batch_size;
				if (end > trainingSet.size()) end = trainingSet.size();
				learnOnMiniBatch(trainingSet.subList(start, end));
				start = end;
			}
			
			System.out.println("epoch " + epoch + " finished");
		}
		
		//сохранить параметры нейросети
		int nameAdd = 1;
		String name = "paramsMNIST" + nameAdd + ".net";
		File test = new File(file.getAbsolutePath() + "/" + name);
		while (test.exists()) {
			nameAdd++;
			name = "paramsMNIST" + nameAdd + ".net";
			test = new File(file.getAbsolutePath() + "/" + name);
		}
		try {
			PrintWriter pw = new PrintWriter(test);
			pw.println(layers_sizes.length);
			String neurons = Arrays.stream(layers_sizes).mapToObj(String::valueOf).collect(Collectors.joining("\t"));
			pw.println(neurons);
			pw.println(learningSpeed);
			pw.println(mini_batch_size);
			pw.println(epochs_count);
			network.forEach(m -> {
				m.getWeightsStringRepresentation().forEach(pw::println);
				m.getBiasesStringRepresentation().forEach(pw::println);
			});
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		}
	}
	
	private static double sigmoidD(double value) {
		return (1d/(1d + (float)Math.exp(-value)));
	}
	
	private static double sigmoid_prime_deviriative(double value) {
	    //"""Derivative of the sigmoid function."""
	    return sigmoidD(value)*(1-sigmoidD(value));
	}
	
	private static void backProganation(MNISTData data) {
		for (int i = 1; i < layers_sizes.length; i++) {
			network.get(i - 1).zeroDeltaMatrix();
		}
		feedForwardLearning(data.getInputMatrix());
		//Обратное распространение
		//Для выходного слоя
		SimpleMatrix delta = costFunction.apply(data);
		network.get(layers_sizes.length - 2).setDeltaBiases(delta);
		SimpleMatrix delta_weights = delta.mult(network.get(layers_sizes.length - 3).getActivations().transpose());
		network.get(layers_sizes.length - 2).setDeltaWeights(delta_weights);
		//Для остальных слоев (мы-то помним, что он всего один)
		for (int i = layers_sizes.length - 3; i >= 0; i--) {
			delta = network.get(i + 1).getWeights().transpose().mult(delta);
			for (int j = 0; j < delta.numRows(); j++) {
				double prime_derivative = sigmoid_prime_deviriative(network.get(i).getSigmaargument().get(j ,0)); 
				delta.set(j, 0, delta.get(j, 0) * prime_derivative); 
			}
			network.get(i).setDeltaBiases(delta);
			if (i > 0) delta_weights = delta.mult(network.get(i - 1).getActivations().transpose());
			else delta_weights = delta.mult(data.getInputMatrix().transpose());
			network.get(i).setDeltaWeights(delta_weights);
		}
	} 
	
	private static SimpleMatrix costFunctionLog(MNISTData data) {
//		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))) - python code from book		
		SimpleMatrix activations_log = network.get(layers_sizes.length - 2).getActivations().copy().elementLog();
		SimpleMatrix activations_log_ = network.get(layers_sizes.length - 2).getActivations().copy()
													.scale(-1d).plus(1d).elementLog();
		SimpleMatrix output_minus = data.getOutputMatrix().copy().scale(-1d); 
		SimpleMatrix output_minus_ = data.getOutputMatrix().copy().scale(-1d).plus(1d);
		for (int j = 0; j < output_minus.numRows(); j++) {
			output_minus.set(j, 0, output_minus.get(j, 0) * activations_log.get(j, 0));
			output_minus_.set(j, 0, output_minus_.get(j, 0) * activations_log_.get(j, 0));
		}
		SimpleMatrix delta =  output_minus.minus(output_minus_);
		for (int j = 0; j < delta.numRows(); j++) {
			if (delta.get(j, 0) == Double.NaN) delta.set(j, 0, 0d);
			else if (delta.get(j, 0) == Double.POSITIVE_INFINITY) delta.set(j, 0, Double.MAX_VALUE);
			else if (delta.get(j, 0) == Double.NEGATIVE_INFINITY) delta.set(j, 0, -Double.MAX_VALUE);
		}
		return delta;
	}

	private static SimpleMatrix costFunctionQuadratic(MNISTData data) {
		SimpleMatrix delta = network.get(layers_sizes.length - 2).getActivations().minus(data.getOutputMatrix());
		for (int j = 0; j < delta.numRows(); j++) {
			double sigma_deviriative = sigmoid_prime_deviriative(network.get(layers_sizes.length - 2).getSigmaargument().get(j ,0));
			delta.set(j, 0, delta.get(j, 0) * sigma_deviriative);
		}
		return delta;
	}

	
	private static void learnOnMiniBatch(List<MNISTData> miniBatch) {
		List<MatrixBiasesAndWeights> networkBatch = new ArrayList<>();
		for (int i = 1; i < layers_sizes.length; i++) {
			networkBatch.add(new MatrixBiasesAndWeights(true, layers_sizes[i], layers_sizes[i - 1]));
		}
		for (int b = 0; b < miniBatch.size(); b++) {
			//Вычислить для каждого значения 
			backProganation(miniBatch.get(b));
			//Суммировать все промежуточные результаты по дельтам
			for (int i = 1; i < layers_sizes.length; i++) {
				networkBatch.get(i - 1).setWeights(networkBatch.get(i - 1).getWeights().plus(network.get(i - 1).getDeltaWeights()));
				networkBatch.get(i - 1).setBiases(networkBatch.get(i - 1).getBiases().plus(network.get(i - 1).getDeltaBiases()));
			}
		}
		//Обновить все веса и смещения в сети по резльтатам суммы от мини-пакета
		//(Сначала проведем умножение дельт)
		float teach_speed = learningSpeed / ((float)miniBatch.size());
		for (int i = 1; i < layers_sizes.length; i++) {
			networkBatch.get(i - 1).getBiases().scale(teach_speed);
			networkBatch.get(i - 1).getWeights().scale(teach_speed);
		}
		//Теперь вычесть
		for (int i = 1; i < layers_sizes.length; i++) {
			weightUpdater.accept(network.get(i - 1), networkBatch.get(i - 1));
			network.get(i - 1).setBiases(network.get(i - 1).getBiases().minus(networkBatch.get(i - 1).getBiases()));
		}
	}
	
	private static void updateWeightWithoutRegulization(MatrixBiasesAndWeights layer, MatrixBiasesAndWeights miniBatchLayer) {
		layer.setWeights(layer.getWeights().minus(miniBatchLayer.getWeights()));
	}

	private static void updateWeightWithRegulization(MatrixBiasesAndWeights layer, MatrixBiasesAndWeights miniBatchLayer) {
		layer.setWeights(layer.getWeights().scale(regularization_constant).minus(miniBatchLayer.getWeights()));
	}
	
	/**
	 * Движение вперед при обучении. 
	 * Здесь вычисленные на каждом шаге промежуточные значения (активация и аргумент сигма
	 * функции каждого нейрона сохраняются с целью вычислений обратного распространения)
	 *  
	 * @param inputColumn - матрица (вектор - столбец) входных значений
	 * (ее формирование - отлельный вопрос)
	 */
	public static void feedForwardLearning(SimpleMatrix inputColumn) {
		SimpleMatrix result = inputColumn;
		for (int i = 1; i < layers_sizes.length; i++) {
			network.get(i - 1).setSigmaargument(network.get(i - 1).getWeights().mult(result).plus(network.get(i - 1).getBiases()));
			for (int j = 0; j < network.get(i - 1).getSigmaargument().numRows(); j++) {
				network.get(i - 1).getActivations().set(j, 0, sigmoidD(network.get(i - 1).getSigmaargument().get(j, 0)));
			}
			result = network.get(i - 1).getActivations();
		}
	}

}
