package test.neural_network.perceptron;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


public class PerceptronEducation {
	
	private static final String MNIST_path = "/home/legioner/MNIST/";
	private static int education_iteration_limit = 100;
	private final static Perceptron[] perceptrons = new Perceptron[10];
	private final static int elements = 60000;
	private final static List<MNISTData> trainingSet = new ArrayList<>(elements);

	public static void main(String[] args) {
		
		System.out.println("initialization");
		//Создание прецептрона
		//Один перцептрон - одно число распознается
		for (int i = 0; i < 10; i ++) {
			perceptrons[i] = new Perceptron(28 * 28, elements, 20);
			perceptrons[i].initialyze();
		}
		
		//Подготовить тренировочные данные
		System.out.println("preparing training data");
		File file = new File("");
		for (int i = 0; i < 10; i++) {
			String tempPath = MNIST_path + i + "/";
			File tempFile = new File(tempPath);
			File[] files = tempFile.listFiles();
			for (File f : files) trainingSet.add(new MNISTData(f)); 
		}
		System.out.println("training set ready");

		//обучение
//		Collections.shuffle(trainingSet);
//		List<Future<Boolean>> threadResults = new ArrayList<>(3);
//		ExecutorService service = Executors.newFixedThreadPool(3);
//		threadResults.add(service.submit( new EducationThread(0, 20000)));
//		threadResults.add(service.submit( new EducationThread(20000, 40000)));
//		threadResults.add(service.submit( new EducationThread(40000, elements)));
//		for (Future<Boolean> f : threadResults) {
//			try {
//				f.get();
//			}
//			catch (InterruptedException | ExecutionException e) {
//				e.printStackTrace();
//			}
//		}
//		service.shutdown();
		
		for (int i = 9; i < 10; i++) {
			Collections.shuffle(trainingSet);
			Perceptron toTeach = perceptrons[i];
			int currIter = 0;
			boolean perfect = false;
			while (!perfect && currIter < education_iteration_limit) {
				perfect = true;
//				int notGood = 0;
				List<Integer> notGood =  new LinkedList<>(); 
//				for (MNISTData example : trainingSet) {
				for (int e = 0; e < trainingSet.size(); e++) {
					MNISTData example = trainingSet.get(e); 
					if (toTeach.feedForward(example.getInputs()) != (i == example.getResult())) {
//						notGood++;
						notGood.add(e);
						perfect = false;
						//Обновить веса
						boolean plus = false;
						if (i == example.getResult()) {
							plus = true;
						}
						toTeach.updateWeights(example.getInputs(), plus);
					}
				}
				currIter++;
				System.out.println("iteration " + currIter + " finished. " + notGood.size() + " errors");
				
				while (notGood.size() > 0) {
					MNISTData example = trainingSet.get(notGood.get(0));
					if (toTeach.feedForward(example.getInputs()) != (i == example.getResult())) {
						//Обновить веса
						boolean plus = false;
						if (i == example.getResult()) {
							plus = true;
						}
						toTeach.updateWeights(example.getInputs(), plus);
					}
					else {
						notGood.remove(0);
					}
				}
				
			} 
			System.out.println("training for " + i + " finished" + 
								(currIter >= education_iteration_limit ? " education iterations exceded" : ""));
			
			//сохранить параметры нейросети
			Perceptron toSave = perceptrons[i];
			int nameAdd = 1;
			String name = "perceptronMNIST" + nameAdd + "_" + i + ".net";
			File test = new File(file.getAbsolutePath() + "/" + name);
			while (test.exists()) {
				nameAdd++;
				name = "perceptronMNIST" + nameAdd + "_" + i + ".net";
				test = new File(file.getAbsolutePath() + "/" + name);
			}
			try {
				toSave.saveToFile(file.getAbsolutePath() + "/" + name);
			}
			catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			System.out.println("model saved");	
		}
		
	}

	//Пробовал так для увеличения скорости обучения
	//Прирост скорости не значительный
	private static class EducationThread implements Callable<Boolean> {
		private int from;
		private int to;
		private List<MNISTData> localSet;
		public EducationThread(int from, int to) {
			this.from = from;
			this.to = to;
			this.localSet = trainingSet.subList(from, to);
		}
		@Override
		public Boolean call() throws Exception {
			Perceptron toTeach = perceptrons[0];
			int currIter = 0;
			boolean perfect = false;
			boolean[] activations = new boolean[elements];  
			while (!perfect && currIter < education_iteration_limit) {
				perfect = true;
				for (MNISTData example : localSet) {
					if (toTeach.feedForwardThreads(example.getInputs(), activations) != (0 == example.getResult())) {
						perfect = false;
						//Обновить веса
						boolean plus = false;
						if (0 == example.getResult()) {
							plus = true;
						}
						toTeach.updateWeightsThreads(example.getInputs(), plus, activations);
					}
				}
				currIter++;
				if (currIter % 100 == 0) System.out.println("iteration " + currIter);
			} 
			return true;
		}
		
	}
	
}
