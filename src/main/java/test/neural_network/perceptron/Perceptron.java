package test.neural_network.perceptron;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.IntSummaryStatistics;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

/**
 * Перцептрон по Розенблат-у (как я его понимаю):
 * - скрытый ассоцаитивный слой
 * - каждый элемент ассоциативного слоя обязательно соединен с одним входом (просто индекс в массиве), 
 * и с некоторыми случайно выбранными (до 20)
 * - активирующий элемент - входы все ассоциативные элементы.
 * 
 * Дополнительно:
 * - ассоцитивный слой содержит 60000 элементов (примерно по числу обучающей выборки)
 * - 10 перцептрон-ов (по одному на одну цифру)
 * - обучаются на всей выборке 
 * 
 * @author a.palkin
 *
 */
public class Perceptron {

	private int inputsCount;
	private int layerSize;
	private int maxAdditionalConnections;
	private List<AssationtionLayerItem> layer;
	
	public Perceptron(int inputsCount, int layerSize, int maxAdditionalConnections) {
		this.inputsCount = inputsCount;
		this.layerSize = layerSize;
		this.maxAdditionalConnections = maxAdditionalConnections;
		this.layer = new ArrayList<>(layerSize + 1); //+1 - bias
	}
	
	public void initialyze() {
		float directStep = ((float)layerSize)/((float)inputsCount);
		int current = 0;
		float rangeUp = directStep;
		for (int i = 0; i < layerSize; i++) {
			AssationtionLayerItem a = new AssationtionLayerItem();
			a.weight = 0;
			//a.setActivationLimit(0);
			a.pos = i;
			a.sourcesPos.add(current);
			if (i > rangeUp) {
				current++;
				rangeUp += directStep;
			}			
			int sources = (int)(Math.random() * ((double)maxAdditionalConnections));
			sources++;
			for (int j = 0; j < sources; j++) {
				int pos = (int)(Math.random() * ((double)inputsCount));
				if (pos == inputsCount) pos--;
				pos = j % 2 == 0 ? pos : inputsCount - (pos == 0 ? 1 : pos);
				while (a.sourcesPos.indexOf(pos) != -1) {
					pos = (int)(Math.random() * ((double)inputsCount));
					pos = j % 2 == 0 ? pos : inputsCount - (pos == 0 ? 1 : pos);
				}
				a.sourcesPos.add(pos);
			}
			a.sourcesPos_len = a.sourcesPos.size(); 
			a.sourcesPos_arr = a.sourcesPos.stream().mapToInt(Integer::intValue).toArray();
			a.activationLimit = a.sourcesPos.size() / 2;
			this.layer.add(a);
		}
		//bias
		AssationtionLayerItem a = new AssationtionLayerItem();
		a.weight = 0;
		a.activationLimit = 0;
		a.pos = layerSize;
		this.layer.add(a);
	}
	
	
	public void saveToFile(String fileName) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(new BufferedOutputStream(new FileOutputStream(new File(fileName))));
		pw.println(inputsCount);
		pw.println(layerSize);
		pw.println(maxAdditionalConnections);
		for (int i = 0; i <= layerSize; i++) {
			AssationtionLayerItem a = layer.get(i);
			pw.println(a.pos);
			pw.println(a.weight);
			pw.println(a.activationLimit);
			pw.println(a.sourcesPos.size());
			if (!a.sourcesPos.isEmpty()) {
				a.sourcesPos.forEach(pw::println);				
			}
		}
		pw.close();
	}
	

	public static Perceptron loadFromFile(File file) throws FileNotFoundException {
		Scanner scan = new Scanner(new BufferedInputStream(new FileInputStream(file)));
		String line = scan.nextLine();
		int inputsCount = Integer.parseInt(line);
		line = scan.nextLine();
		int layerSize = Integer.parseInt(line);
		line = scan.nextLine();
		int maxAdditionalConnections = Integer.parseInt(line);
		Perceptron perceptron = new Perceptron(inputsCount, layerSize, maxAdditionalConnections);
		for (int i = 0; i <= layerSize; i++) {
			AssationtionLayerItem a = new AssationtionLayerItem();
			line = scan.nextLine();
			a.pos = Integer.parseInt(line);
			line = scan.nextLine();
			a.weight = Integer.parseInt(line);
			line = scan.nextLine();
			a.activationLimit = Integer.parseInt(line);
			line = scan.nextLine();
			int size = Integer.parseInt(line);
			for (int j = 0; j < size; j++) {
				line = scan.nextLine();
				a.sourcesPos.add(Integer.parseInt(line));	
			}
			a.sourcesPos_len = a.sourcesPos.size(); 
			a.sourcesPos_arr = a.sourcesPos.stream().mapToInt(Integer::intValue).toArray();
			perceptron.layer.add(a);
		}
		scan.close();
		return perceptron;
	}

	
	public boolean feedForward(boolean[] inputs) {
		long sum = 0L;
		for (int i = 0; i < layerSize; i++) {
			int cnt = 0;
			AssationtionLayerItem a = layer.get(i);
			for (int j = 0; j < a.sourcesPos_len; j++) {
				if (inputs[a.sourcesPos_arr[j]]) cnt++;
			}
			if (cnt >= a.activationLimit) {
				a.active = true;
				sum += a.weight;	
			} else {
				a.active = false;
			}
			
//				long cnt = layer.get(i).sourcesPos.stream().filter(j -> inputs[j]).count();
//				if (cnt > layer.get(i).activationLimit) {
//					layer.get(i).active = true;
//					sum += layer.get(i).weight;	
//				} else {
//					layer.get(i).active = false;
//				}
		}
		sum += layer.get(layerSize).weight;
		return sum > 0;
	}
	
	public void updateWeights(boolean[] inputs, boolean positiveSign) {
		for (int i = 0; i < layerSize; i++) {
			if (layer.get(i).active) {
				if (positiveSign ) layer.get(i).weight++;
				else layer.get(i).weight--;
			}
		}
		//Для смещения инвертируем знак
		layer.get(layerSize).weight += (positiveSign ? -1 : 1);
	}
	
	
	public boolean feedForwardThreads(boolean[] inputs, boolean[] activations) {
		long sum = 0L;
		for (int i = 0; i < layerSize; i++) {
			int cnt = 0;
			AssationtionLayerItem a = layer.get(i);
			for (int j = 0; j < a.sourcesPos_len; j++) {
				if (inputs[a.sourcesPos_arr[j]]) cnt++;
			}
			if (cnt >= a.activationLimit) {
				activations[i] = true;
				sum += a.weight;	
			} else {
				activations[i] = false;
			}
		}
		sum += layer.get(layerSize).weight;
		return sum > 0;
	}
	
	public synchronized void updateWeightsThreads(boolean[] inputs, boolean positiveSign, boolean[] activations) {
		for (int i = 0; i < layerSize; i++) {
			if (layer.get(i).active) {
				if (positiveSign ) layer.get(i).weight++;
				else layer.get(i).weight--;
			}
		}
		//Для смещения инвертируем знак
		layer.get(layerSize).weight += (positiveSign ? -1 : 1);
	}
	
}
