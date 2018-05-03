package com.taikang.deeplearning4j_sample.NN;

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

import com.taikang.deeplearning4j_sample.DataLoader.MyMnistDataSetIterator;

/**
 * @author xingxf03
 * @since 2018年4月26日
 * @Discription TODO
 */
public class MultiLayerNN {
	public static void main(String[] args) throws IOException {
		        //参数定义
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
		                //参数正则化
		                .regularization(true).l2(0.0005)
		                //神经网络学习率
		                .learningRate(.01)
		                //权重初始化
		                .weightInit(WeightInit.XAVIER)
		                //采用SGD随机梯度下降算法
		                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		                //设置权重更新器的动量
		                .updater(Updater.NESTEROVS).momentum(0.9)
		                //将配置复制n次，建立分层的网络结构
		                .list()
		                //配置各层的网络结构及相应的激活函数
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
		                //将输入图片展成28*28的一维向量
		                .setInputType(InputType.convolutionalFlat(28,28,1)) 
		                .backprop(true).pretrain(false).build();
		        
		        //构建模型 
				System.out.println("Build model....");
		        MultiLayerNetwork  model = new MultiLayerNetwork(conf);
		        model.init();
		        
		        System.out.println("Train the model...");
		        model.setListeners(new ScoreIterationListener(1));
		        //开始训练
		        for (int i = 0; i < nEpoches; i++) {
					model.fit(trainData);
					System.out.println("*** Complete epoch "+(i+1));
					System.out.println("Evaluate the model....");
					//分析训练情况
					Evaluation evaluation = model.evaluate(testData);
					System.out.println(evaluation.stats());
					testData.reset();
				}
		        System.out.println("*** Finish Train ***");		                             
			}        	
}
