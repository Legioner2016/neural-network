package test.neural_network.mnist_ui_test;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ejml.simple.SimpleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import test.neural_network.perceptron.Perceptron;
import test.neural_network.self_write_direct_connected.NeuroNetwork;
import test.neural_network.statistic_analyze.StaticRecognation;

public class MainFrame extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4911433013492836791L;
	
	private JPanel panel;
	private LeftPanel leftpanel;
	private RightPanel rightpanel;
	private JPanel upperRightpanel;
	private JButton buttonConvert;
	private JButton buttonClear;
	private JButton buttonType;
	private JPanel centralpanel;
	
	private MultiLayerNetwork modelBase;
	private StaticRecognation stat;
	private Perceptron[] perceprtons;
	private NeuroNetwork network;
	
	private EnumComparsionType conversionType = EnumComparsionType.dl4j_base; 
	
	private boolean mouseState = false;
	private int state = 0;
	private Integer result = null;
	
	private Map<Integer, List<Point>> points = new HashMap<>();
	
	public MainFrame(MultiLayerNetwork model, StaticRecognation stat, Perceptron[] perceprtons, NeuroNetwork network) {
		this.modelBase = model;
		this.stat = stat;
		this.perceprtons = perceprtons;
		this.network = network;
		this.setMinimumSize(new Dimension(430, 150));
		panel = new JPanel();
		panel.setMinimumSize(new Dimension(380, 150));
		this.add(panel);
		leftpanel = new LeftPanel();
		leftpanel.setMinimumSize(new Dimension(26 * 4, 26 * 4));
		leftpanel.setPreferredSize(new Dimension(26 * 4, 26 * 4));
		leftpanel.setBorder(BorderFactory.createLineBorder(Color.black));
		panel.add(leftpanel);
		buttonConvert = new JButton("->");
		buttonConvert.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				//Test. Save picture to png
				BufferedImage img = new BufferedImage(28 * 4, 28 * 4, BufferedImage.TYPE_BYTE_GRAY);
				Graphics2D g2 = (Graphics2D) img.getGraphics();
				g2.setColor(Color.WHITE);
				g2.setBackground(Color.WHITE);
				g2.fillRect(0, 0, 28 * 4, 28 * 4);
				g2.setColor(Color.BLACK);
	           g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
               g2.setStroke(new BasicStroke(6));
               points.forEach((i, l) -> {
            	   for (int j = 0; j < l.size(); j++) {
            		   if (j == 0) g2.drawLine(l.get(j).x, l.get(j).y + 8, l.get(j).x, l.get(j).y + 8);
            		   else g2.drawLine(l.get(j - 1).x, l.get(j - 1).y + 8, l.get(j).x, l.get(j).y + 8);
            	   }
               });
               BufferedImage after = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
               AffineTransform at = new AffineTransform();
               at.scale(0.25, 0.25);
               AffineTransformOp scaleOp = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
               after = scaleOp.filter(img, after);
       				float[] imageData = new float[28 * 28];
   					int[] img_ = new int[28 * 28];
   					double[] tmp = new double[28 * 28];
   					boolean[] imageData_ = new boolean[28 * 28];
   					int skip_y = 0;
   					int skip_x = 0;
   					after.getRaster().getPixels(0, 0, 28, 28, img_);
   					upperRightpanel.getGraphics().drawImage(after, 0, 0, 28, 28, Color.WHITE, null);

   					switch (conversionType) {
					case dl4j_base:
	   					for (int j = 0; j < img_.length; j++) {
	   						float v = 255 - (img_[j] & 0xFF); //byte is loaded as signed -> convert to unsigned
	   						imageData[j] = v / 255.0f;
	   					}
						
	       				INDArray features = Nd4j.create(imageData);
	       				List<INDArray> activations = modelBase.feedForward(features);
	       				INDArray output = activations.get(activations.size() - 1);
	       				
	       				result = -1;
	       				for (long i = 0; i < 10; i++) {
	       					if (output.getFloat(i) > 0.5) {
	       						result = (int)i; 
	       						break;
	       					}
	       				}

						break;
					case self_direct:
						after.getRaster().getPixels(0, 0, 28, 28, tmp);
						for (int i = 0; i < tmp.length; i++) tmp[i] = (255d - tmp[i])/255d;
						
						SimpleMatrix resultActivation = network.feedForward(new SimpleMatrix(tmp.length, 1, true, tmp));
						result = -1;
						for (int j = 0; j < resultActivation.numRows(); j++) {
							if (resultActivation.get(j , 0) > 0.5f) {
								result = j;
								break;
							}
						}
						break;
					case statistic:
	   					after.getRaster().getPixels(0, 0, 28, 28, img_);
	   					upperRightpanel.getGraphics().drawImage(after, 0, 0, 28, 28, Color.WHITE, null);
						result = stat.recognize(img_);
						break;
					case perceptron:
	   					after.getRaster().getPixels(0, 0, 28, 28, tmp);
	   					for (int i = 0; i < 784; i++) tmp[i] = (255d - tmp[i])/255d;
	   					upperRightpanel.getGraphics().drawImage(after, 0, 0, 28, 28, Color.WHITE, null);
	   					//Move to left-up corner
	   					for (int i = 0; i < tmp.length; i++) {
	   						if (tmp[i] > 0) {
	   							skip_y = (i / 28); 	
	   							break;
	   						}
	   					}
	   					outer: for (int i = 0; i < 28; i++) {
	   						for (int j = 0; j < 28; j++) {
	   							if (tmp[j * 28 + i] > 0) {
	   								skip_x = i; 		
	   								break outer;
	   							}
	   						}
	   					}
	   					for (int x = 0; x < 28; x++) {
	   						for (int y = 0; y < 28; y++) {
	   							if (x >= skip_x && y >= skip_y) {
	   								imageData_[(y - skip_y) * 28 + (x - skip_x)] = tmp[y * 28 + x] > 0.2; 			
	   							}
	   						}
	   					}
	   					//Recognizing
	   					result = -1;
	   					for (int j = 0; j < 10; j++) {
	   						if (perceprtons[j].feedForward(imageData_)) {
	   							result = j;
	   							break;
	   						}
	   					}

	   					
						break;
					default:
						break;
   					}
   					
					rightpanel.repaint();

			}
		});
		buttonClear = new JButton("X");
		buttonClear.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				mouseState = false;
				state = 0;
				points.clear();
				result = null;
				leftpanel.repaint();
				rightpanel.repaint();
			}
		});
		buttonType = new JButton(getLabelComparsion(conversionType));
		buttonType.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				for (int i = 0; i < EnumComparsionType.values().length; i++) {
					if (conversionType == EnumComparsionType.values()[i]) {
						int next = i == EnumComparsionType.values().length - 1 ? 0 : i  + 1;
						conversionType = EnumComparsionType.values()[next]; 
						break;
					}
				}
				buttonType.setText(getLabelComparsion(conversionType));
			}
		});
		panel.add(buttonConvert);
		panel.add(buttonClear);
		panel.add(buttonType);
		rightpanel = new RightPanel();
		rightpanel.setMinimumSize(new Dimension(28, 28));
		rightpanel.setPreferredSize(new Dimension(28, 28));
		rightpanel.setBorder(BorderFactory.createLineBorder(Color.black));
		panel.add(rightpanel);
		
		upperRightpanel = new JPanel();
		upperRightpanel.setMinimumSize(new Dimension(28, 28));
		upperRightpanel.setPreferredSize(new Dimension(28, 28));
		panel.add(upperRightpanel);
		
//		pack();
	}
	
	private class LeftPanel extends JPanel {

		
		/**
		 * 
		 */
		private static final long serialVersionUID = -5922191471619460616L;

		public LeftPanel() {
			super();
			this.addMouseListener(new java.awt.event.MouseAdapter() {
		        public void mousePressed(java.awt.event.MouseEvent evt) {
		        	state++;
		        	List<Point> newList = new ArrayList<>();
		        	newList.add(new Point(evt.getX(), evt.getY()));
		        	points.put(state, newList);
		        	mouseState = true;
		        }
		        public void mouseReleased(java.awt.event.MouseEvent evt) {
		        	mouseState = false;
		        }
		    });
			this.addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {
		        public void mouseDragged(java.awt.event.MouseEvent evt) {
		            if (mouseState) {
		            	points.get(state).add(new Point(evt.getX(), evt.getY()));	
		            	leftpanel.repaint();
		            }
		        }
		    });
		}
		
		@Override
	    public void paintComponent(Graphics g) {
	           super.paintComponent(g);
	           
	           Graphics2D g2 = (Graphics2D) g;
	           g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
               g2.setStroke(new BasicStroke(6));
               points.forEach((i, l) -> {
            	   for (int j = 0; j < l.size(); j++) {
            		   if (j == 0) g2.drawLine(l.get(j).x, l.get(j).y, l.get(j).x, l.get(j).y);
            		   else g2.drawLine(l.get(j - 1).x, l.get(j - 1).y, l.get(j).x, l.get(j).y);
            	   }
               });
	    }
		
		
	}
	
	
	private class RightPanel extends JPanel {

		
		/**
		 * 
		 */
		private static final long serialVersionUID = 7377581490687120939L;

		public RightPanel() {
			super();
		}
		
		@Override
	    public void paintComponent(Graphics g) {
	           super.paintComponent(g);
	           
	           Graphics2D g2 = (Graphics2D) g;
	           if (result != null) {
	        	   g2.drawString(result.toString(), 3, 15);	        	   
	           }
	    }
	}
	
	private String getLabelComparsion(EnumComparsionType conversionType) {
		switch (conversionType) {
			case self_direct: return "self direct";  //self writed direct connections with move image to left top corner 
			case statistic: return "statistic"; //recognize base on statistic analysis
			case perceptron: return "perceptron"; //recognize with perceptron
			default: return "dl4j default"; //deeplearning for java, convolutional network example for MNIST as is
		}
	}

	
}
