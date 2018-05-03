package com.taikang.deeplearning4j_sample.NN;


import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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

import com.taikang.deeplearning4j_sample.DataLoader.MyMnistDataSetIterator;

/**
/**  
* @ClassName: SingleLayerNN  
* @Description: TODO(这里用一句话描述这个类的作用)  
* @author xingxf03  
* @date 2018年4月28日  
*    
*/
@SuppressWarnings("deprecation")
public class SingleLayerNN {
	public static void main(String[] args) throws IOException {
		int nChannels = 1;      //black & white picture, 3 if color image  
		int outputNum = 10;     //number of classification  
		int batchSize = 64;     //mini batch size for sgd  
		int nEpochs = 10;       //total rounds of training  
		int iterations = 1;     //number of iteration in each traning round  
		int seed = 123;         //random seed for initialize weights  
		
		DataSetIterator mnistTrain = new MyMnistDataSetIterator(batchSize, true, 12345);  
		DataSetIterator mnistTest = new MyMnistDataSetIterator(batchSize, false, 12345);  
		
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
				                                .seed(seed)
				                                .iterations(iterations)
				                                .regularization(true).l2(0.0005)
				                                .learningRate(0.001)
				                                .weightInit(WeightInit.XAVIER)
				                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				                                .updater(Updater.NESTEROVS).momentum(0.9)
				                                .list()
				                                .layer(0, new DenseLayer.Builder()
				                                		.nOut(420)
				                                		.activation(Activation.RELU)
				                                        .build())
				                                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
				        		                		.activation(Activation.SOFTMAX)
				        		                		.nOut(outputNum)
				        		                		.build())
												//将输入图片展成28*28的一维向量
								                .setInputType(InputType.convolutionalFlat(28,28,1)) 
								                .backprop(true).pretrain(false)
								                .build();
		
		// 构建模型
		System.out.println("Build model....");
		MultiLayerNetwork model = new MultiLayerNetwork(configuration);
		model.init();

		System.out.println("Train the model...");
		model.setListeners(new ScoreIterationListener(1));
		// 开始训练
		for (int i = 0; i < nEpochs; i++) {
			model.fit(mnistTrain);
			System.out.println("*** Complete epoch " + (i + 1));
			System.out.println("Evaluate the model....");
			// 分析训练情况
			Evaluation evaluation = model.evaluate(mnistTest);
			System.out.println(evaluation.stats());
			mnistTest.reset();
		}
		System.out.println("*** Finish Train ***");
	}

}
