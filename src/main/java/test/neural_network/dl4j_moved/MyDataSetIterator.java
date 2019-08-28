package test.neural_network.dl4j_moved;

import java.util.List;
import java.util.NoSuchElementException;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;

public class MyDataSetIterator implements DataSetIterator {


    private static final long serialVersionUID = -116636792426198949L;
    protected int batch, numExamples;
    protected DataSetFetcher fetcher; 
    protected DataSetPreProcessor preProcessor;


    public MyDataSetIterator(int batch, int numExamples, List<MNISTData> trainingSet) {
        this.batch = batch;
        this.fetcher = new MyDataFetcher(trainingSet);
        this.numExamples = numExamples;
    }

    @Override
    public boolean hasNext() {
        return fetcher.hasMore() && fetcher.cursor() < numExamples;
    }

    @Override
    public DataSet next() {
        if(!hasNext())
            throw new NoSuchElementException("No next element - hasNext() == false");
        int next = Math.min(batch, numExamples - fetcher.cursor());
        fetcher.fetch(next);
        return fetcher.next();
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        return fetcher.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return fetcher.totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        fetcher.reset();
    }

    @Override
    public int batch() {
        return batch;
    }

    /**
     * Set a pre processor
     *
     * @param preProcessor a pre processor to set
     */
    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public DataSet next(int num) {
        fetcher.fetch(num);
        DataSet next = fetcher.next();
        if (preProcessor != null)
            preProcessor.preProcess(next);
        return next;
    }



}
