package test.neural_network.dl4j_moved;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.MathUtils;

public class MyDataFetcher extends BaseDataFetcher {
    /**
	 * 
	 */
	private static final long serialVersionUID = 2953784084314382283L;

	public static final int NUM_EXAMPLES = 60000;

//    protected transient MnistManager man;
    protected boolean binarize = true;
    protected int[] order;
    protected Random rng;
    protected boolean shuffle;

    protected boolean firstShuffle = true;
    protected final int numExamples;
    
    protected List<MNISTData> trainingSet;


    /**
     * Constructor telling whether to binarize the dataset or not
     * @param binarize whether to binarize the dataset or not
     * @throws IOException
     */
    public MyDataFetcher(boolean binarize, List<MNISTData> trainingSet)  {
        this(binarize, true, System.currentTimeMillis(), NUM_EXAMPLES, trainingSet);
    }

    public MyDataFetcher(boolean binarize, boolean shuffle, long rngSeed, int numExamples, List<MNISTData> trainingSet)  {
    	this.trainingSet = trainingSet;

        totalExamples = NUM_EXAMPLES;

        numOutcomes = 10;
        this.binarize = binarize;
        cursor = 0;
        inputColumns = 784;
        this.shuffle = shuffle;

        order = new int[NUM_EXAMPLES];
        for (int i = 0; i < order.length; i++)
            order[i] = i;
        rng = new Random(rngSeed);
        this.numExamples = numExamples;
        reset(); //Shuffle order
    }

    public MyDataFetcher(List<MNISTData> trainingSet)  {
        this(true, trainingSet);
    }

    @Override
    public void fetch(int numExamples) {
        if (!hasMore()) {
            throw new IllegalStateException("Unable to get more; there are no more images");
        }

        float[][] featureData = new float[numExamples][0];
        float[][] labelData = new float[numExamples][0];

        int actualExamples = 0;
        for (int i = 0; i < numExamples; i++, cursor++) {
            if (!hasMore())
                break;
            
            MNISTData curent_data = trainingSet.get(order[cursor]);

            featureData[actualExamples] = curent_data.getInputs();
            labelData[actualExamples] = new float[numOutcomes];
            labelData[actualExamples][curent_data.getResult()] = 1.0f;

            actualExamples++;
        }

        if (actualExamples < numExamples) {
            featureData = Arrays.copyOfRange(featureData, 0, actualExamples);
            labelData = Arrays.copyOfRange(labelData, 0, actualExamples);
        }

        INDArray features = Nd4j.create(featureData);
        INDArray labels = Nd4j.create(labelData);
        curr = new DataSet(features, labels);
    }

    @Override
    public void reset() {
        cursor = 0;
        curr = null;
        if (shuffle) {
            if(numExamples < NUM_EXAMPLES){
                //Shuffle only first N elements
                if(firstShuffle){
                    MathUtils.shuffleArray(order, rng);
                    firstShuffle = false;
                } else {
                    MathUtils.shuffleArraySubset(order, numExamples, rng);
                }
            } else {
                MathUtils.shuffleArray(order, rng);
            }
        }
    }

    @Override
    public DataSet next() {
        return curr;
    }
    
}