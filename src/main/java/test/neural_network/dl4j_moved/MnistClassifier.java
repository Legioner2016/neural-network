package test.neural_network.dl4j_moved;

import org.apache.commons.io.FilenameUtils;
import org.apache.log4j.PropertyConfigurator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Created by agibsonccc on 9/16/15.
 * 
 * Я хотел подвинуть изображение к левому верхнемсу углу (как с перцептроном) - и тем снизить влияние 
 * смещения. Но видимо, не до конца разорбрался с логикой чтения - стало хуже - нет в тесте 
 * (Зато полезно было посмотреть - как подавать свои данные на вход такой сети)   
 */
public class MnistClassifier {

	private static final Logger log = LoggerFactory.getLogger(MnistClassifier.class);
	private final static String tempFolder = "/opt/cms/network/";
	private static final String MNIST_path = "/home/legioner/MNIST/";
	private final static int elements = 60000;

	private static void configureLog4j() {
		
		Properties p = new Properties();
		p.put("log4j.appender.CONSOLE", "org.apache.log4j.ConsoleAppender");
		p.put("log4j.appender.CONSOLE.layout", "org.apache.log4j.PatternLayout");
		p.put("log4j.appender.CONSOLE.layout.ConversionPattern", "%p %m%n");
		p.put("log4j.rootLogger", "INFO,CONSOLE");
		p.put("log4j.logger.MnistClassifier", "INFO");
		
		PropertyConfigurator.configure(p);
	}
	
    public static void main(String[] args) throws Exception {
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 64; // Test batch size
        int nEpochs = 1; // Number of training epochs
        int seed = 123; //

        configureLog4j();
        
        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("preparing training data");
        List<MNISTData> trainingSet = new ArrayList<>(elements);
		for (int i = 0; i < 10; i++) {
			String tempPath = MNIST_path + i + "/";
			File tempFile = new File(tempPath);
			File[] files = tempFile.listFiles();
			for (File f : files) trainingSet.add(new MNISTData(f)); 
		}
		log.info("training set ready");

        
        log.info("Load data....");
        DataSetIterator mnistTrain = new MyDataSetIterator(batchSize, elements, trainingSet);
//        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        /*
            Construct the neural network
         */
        log.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .build();

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)
        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model...");
        model.setListeners(new ScoreIterationListener(10)); //Print score every 10 iterations and evaluate on test set every epoch
        model.fit(mnistTrain, nEpochs);

        String basePart = "lenetmnist_moved";
        String suffix = ".zip";
        String fileName = basePart + suffix; 
        int iter = 0;
        boolean good = false;
        while (!good) {
        	if (iter == 0)  fileName = basePart + suffix;
        	else fileName = basePart + String.valueOf(iter) + suffix;
        	File file = new File(FilenameUtils.concat(tempFolder, fileName));
        	if (!file.exists()) good = true;
        	iter++;
        }
        String path = FilenameUtils.concat(tempFolder, fileName);

        log.info("Saving model to tmp folder: "+path);
        model.save(new File(path), true);

        log.info("****************Example finished********************");
    }

}