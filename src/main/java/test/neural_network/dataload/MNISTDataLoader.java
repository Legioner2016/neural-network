package test.neural_network.dataload;

import java.awt.Point;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

import javax.imageio.ImageIO;

/**
 * Java test for 
 * "Neural networks and deep learning"
 * (http://neuralnetworksanddeeplearning.com/chap1.html)
 * book example
 * 
 * 
 * This class is reading the mnist data set and stored handwriter number image for their folders
 * (So visually can be tested that zero images are in folder 0, one image in 1 and so on)
 * @author legioner
 *
 */
public class MNISTDataLoader {

	private static final String defaultPath = "/home/legioner/MNIST/";
	private static final int thread_count = 4;
	private static final int image_file_header = 16;
	private static final int label_file_header = 8;
	
	/**
	 * Read the mnist data from training files and organize it in image groups folders
	 * save as png images 
	 * Read is multi-threaded
	 * 
	 * @param args - path to result image folders. Default is ~/MNIST 
	 * @throws URISyntaxException 
	 * @throws IOException 
	 */
	public static void main(String[] args) throws URISyntaxException, IOException, Exception {
		//Preparing folders 0..9 for images
		String resultImagePath = args.length == 0 ? defaultPath : args[0];
		File resultFolder = new File(resultImagePath);
		if (!resultFolder.exists()) resultFolder.mkdirs();
		IntStream.range(0, 10).forEach(i -> {
			String subDir = resultImagePath + "/" + i + "/"; 
			(new File(subDir)).mkdir();
		});
		
		//Read the file base data (for divide file processing for threads)
		URL urlImages = MNISTDataLoader.class.getClassLoader().getResource("train-images.idx3-ubyte");
		URL urlLabels = MNISTDataLoader.class.getClassLoader().getResource("train-labels.idx1-ubyte");
		InputStream imageIs = new FileInputStream(new File(urlImages.toURI()));
		byte[] b_ = new byte[16];
		int[] b = new int[16];
		if (imageIs.read(b_) < 16) {
			imageIs.close();
			throw new FileNotFoundException("image file is too small");
		}
		imageIs.close();
		for (int i = 0; i < 16; i++) b[i] = (b_[i] & 0xFF); //signed byte to unsigned 
		int imageCount = ((b[4] << 24) | (b[5] << 16) | (b[6] << 8)) + b[7];  
		int imageHeight = ((b[8] << 24) | (b[9] << 16) | (b[10] << 8)) + b[11];
		int imageWidth = ((b[12] << 24) | (b[13] << 16) | (b[14] << 8)) + b[15];
		System.out.println("images count = " + imageCount + ", image height = " + imageHeight + ", image width = " + imageWidth + ".");
		InputStream labelIs = new FileInputStream(new File(urlLabels.toURI()));
		b_ = new byte[8];
		b = new int[8];
		if (labelIs.read(b_) < 8) {
			imageIs.close();
			labelIs.close();
			throw new FileNotFoundException("label file is too small");
		}
		imageIs.close();
		for (int i = 0; i < 8; i++) b[i] = (b_[i] & 0xFF); //signed byte to unsigned
		int labelCount = ((b[4] << 24) | (b[5] << 16) | (b[6] << 8)) + b[7];
		if (labelCount != imageCount) throw new Exception("Image file and label file contains different number of data");
		labelIs.close();
		
		//Create threads to processing images
		int threads = thread_count >  imageCount ? imageCount : thread_count;
		int images_for_one_thread = imageCount / thread_count;
		
		ExecutorService executor = Executors.newFixedThreadPool(threads);
		List<Future<Boolean>> threadResults = new ArrayList<>(threads);
		List<InputStream> inputs = new ArrayList<>(threads * 2);
		int start = 0;
		for (int i = 0; i < threads; i++) {
			int img_count = images_for_one_thread; 
			if (i == threads - 1) img_count = imageCount - start; //Divide get result with some inaccuracy  
			inputs.add(new BufferedInputStream(new FileInputStream(new File(urlImages.toURI()))));
			inputs.add(new BufferedInputStream(new FileInputStream(new File(urlLabels.toURI()))));
			ImageCreatingThread thread = new ImageCreatingThread(inputs.get(inputs.size() - 2),
										inputs.get(inputs.size() - 1),
										resultImagePath, new Point(imageWidth, imageHeight), img_count, start, i);
			threadResults.add(executor.submit(thread));
			start += img_count;
		}
		//Get thread results
		for (Future<Boolean> f : threadResults) f.get();
		executor.shutdown();
		//Close input streams
		for (InputStream is : inputs) is.close();
	}
	
	/**
	 * One thread to generate images from file
	 * 
	 * 
	 * @author legioner
	 *
	 */
	private static class ImageCreatingThread implements Callable<Boolean> {

		private InputStream imageInputStream;
		private InputStream labelInputStream;
		private String baseOutputFolder;
		private Point imageSize;
		private int	imageCount;
		private int iteration = 0;
		private int threadNumber;
		private int imagebytes;
		private int imageOffset;
		private int labelOffset;
		private int imageFrom;
		
		
		
		public ImageCreatingThread(InputStream imageInputStream, InputStream labelInputStream, String baseOutputFolder,
				Point imageSize, int imageCount, int imageFrom, int threadNumber) throws IOException {
			super();
			this.imageInputStream = imageInputStream;
			this.labelInputStream = labelInputStream;
			this.baseOutputFolder = baseOutputFolder;
			this.imageSize = imageSize;
			this.imageCount = imageCount;
			this.imageFrom = imageFrom; 
			this.threadNumber = threadNumber;
			this.imagebytes = imageSize.x * imageSize.y; 
			this.imageOffset = image_file_header + (imageFrom * imagebytes);
			this.labelOffset = label_file_header + imageFrom;
			this.labelInputStream.skip(labelOffset);
			this.imageInputStream.skip(imageOffset);
		}



		@Override
		public Boolean call() throws Exception {
			while (iteration < imageCount) {
				//Read label 
				int label =	labelInputStream.read();
				String fileName = baseOutputFolder + "/" + Integer.toString(label) + "/" + Integer.toString(imageFrom + iteration) + ".png";
				//Read image bytes
				byte[] img = new byte[imagebytes];
				imageInputStream.read(img);
				int[] img_ = new int[imagebytes];
				for (int i = 0; i < imagebytes; i++) img_[i] = img[i] ^ 0xFF; //Convert color to black on white    
				//Create image
				BufferedImage bi = new BufferedImage(imageSize.x, imageSize.y, BufferedImage.TYPE_BYTE_GRAY);
				bi.getRaster().setPixels(0, 0, imageSize.x, imageSize.y, img_);
				//Save image
			    ImageIO.write(bi, "png", new File(fileName));
				//Next iteration
				iteration++;
				if (iteration % 100 == 0) System.out.println("thread " + threadNumber  + " processed " + iteration  + " images.");
			}
			return true;
		}
		
	}

}
