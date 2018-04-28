package com.taikang.deeplearning4j_sample.CNN;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.taikang.deeplearning4j_sample.DataLoader.MyMnistDataSetIterator;

import org.nd4j.linalg.activations.Activation;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;


/**
 * @author xingxf03
 * @since 2018年4月26日
 * @Discription Lenet MNIST Example
 */
public class LeNetMNIST {
	public static void main(String[] args) throws Exception {
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
		
		//Build the model 
		System.out.println("Build model....");
	
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
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false).build();
        
        MultiLayerNetwork  model = new MultiLayerNetwork(conf);
        model.init();
        
        System.out.println("Train the model...");
      
        model.setListeners(new ScoreIterationListener(100));
        
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
