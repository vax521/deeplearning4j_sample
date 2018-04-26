/**
 * 
 */
package com.taikang.deeplearning4j_sample.CNN;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author xingxf03
 * @since 2018年4月26日
 * @Discription TODO
 */
public class NN_MNSIT {
	public static void main(String[] args) throws IOException {
		//超参数定义
				int nChannels = 1;
				int outputNum = 10;
				int batchSize = 64; 
				int nEpoches = 10;
				int iterations = 1;
				int seed = 123;
				System.out.println("Loading data....");
			
				DataSetIterator trainData = new MyMnistDataSetIterator(batchSize, true, 12345);
				DataSetIterator testData = new MyMnistDataSetIterator(batchSize, false, 12345);
				
				
				  // learning rate schedule in the form of <Iteration #, Learning Rate>
		        Map<Integer, Double> lrSchedule = new HashMap<Integer, Double>();
		        lrSchedule.put(0, 0.01);
		        lrSchedule.put(1000, 0.005);
		        lrSchedule.put(3000, 0.001);
		        
		        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		                .seed(seed)
		                .iterations(iterations)
		                .regularization(true).l2(0.0005)
		                .learningRate(.01)
		                .weightInit(WeightInit.XAVIER)
		                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		                .updater(Updater.NESTEROVS).momentum(0.9)
		                .list()
		                .layer(0, new DenseLayer.Builder().activation(Activation.RELU)
                                .nOut(520).build())
		                .layer(1, new DenseLayer.Builder().activation(Activation.RELU)
		                		.nOut(320).build())
		                .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
		                		.nOut(240).build())
		                .layer(3, new DenseLayer.Builder().activation(Activation.RELU)
		                		.nOut(120).build())
		                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
		                		.activation(Activation.SOFTMAX)
		                		.nOut(outputNum)
		                		.build())
		                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
		                .backprop(true).pretrain(false).build();
		        
		        //Build the model 
				System.out.println("Build model....");
		        MultiLayerNetwork  model = new MultiLayerNetwork(conf);
		        model.init();
		        
		        System.out.println("Train the model...");
		        model.setListeners(new ScoreIterationListener(100));
		        //开始训练
		        for (int i = 0; i < nEpoches; i++) {
					model.fit(trainData);
					System.out.println("*** Complete epoch "+(i+1));
					System.out.println("Evaluate the model....");
					Evaluation evaluation = model.evaluate(testData);
					System.out.println(evaluation.stats());
					testData.reset();
				}
		        System.out.println("*** Finish Train ***");		                             
			}        	
}
