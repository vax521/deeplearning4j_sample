package com.taikang.deeplearning4j_sample.DataLoader;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author xingxf03
 * @since 2018年4月26日
 * @Discription 数据加载类
 */
@SuppressWarnings("serial")
public class MyMnistDataFetcher extends BaseDataFetcher{

	public static final int NUM_EXAMPLES = 60000;
    public static final int NUM_EXAMPLES_TEST = 10000;
    protected static final String TEMP_ROOT = System.getProperty("user.home");
    //放置数据集的文件夹
    protected static final String MNIST_ROOT = "E:\\LearningMaterials\\dataset\\mnist\\";

    protected transient MnistManager man;
    protected boolean binarize = true;
    protected boolean train;
    protected int[] order;
    protected Random rng;
    protected boolean shuffle;

    /**
     * Constructor telling whether to binarize the dataset or not
     * @param binarize whether to binarize the dataset or not
     * @throws IOException
     */
    public MyMnistDataFetcher(boolean binarize) throws IOException {
        this(binarize,true,true,System.currentTimeMillis());
    }

    public MyMnistDataFetcher(boolean binarize, boolean train, boolean shuffle, long rngSeed) throws IOException {
    	System.out.println("mnistExists()="+mnistExists());
    	if(!mnistExists()) {
            new MnistFetcher().downloadAndUntar();
        }
        String images;
        String labels;
        if(train){
            images = MNIST_ROOT + "train-images.idx3-ubyte";
            labels = MNIST_ROOT + "train-labels.idx1-ubyte";
            totalExamples = NUM_EXAMPLES;
        } else {
            images = MNIST_ROOT + "t10k-images.idx3-ubyte";
            labels = MNIST_ROOT + "t10k-labels.idx1-ubyte";
            totalExamples = NUM_EXAMPLES_TEST;
        }

        try {
            man = new MnistManager(images, labels, train);
        }catch(Exception e) {
        	System.out.println("e.getMessage()="+e.getMessage());
            FileUtils.deleteDirectory(new File(MNIST_ROOT));
            new MnistFetcher().downloadAndUntar();
            man = new MnistManager(images, labels, train);
        }

        numOutcomes = 10;
        this.binarize = binarize;
        cursor = 0;
        inputColumns = man.getImages().getEntryLength();
        this.train = train;
        this.shuffle = shuffle;

        if(train){
            order = new int[NUM_EXAMPLES];
        } else {
            order = new int[NUM_EXAMPLES_TEST];
        }
        for( int i=0; i<order.length; i++ ) order[i] = i;
        rng = new Random(rngSeed);
        reset();    //Shuffle order
    }

    private boolean mnistExists(){
    	 File f = new File(MNIST_ROOT,"train-images-idx3-ubyte.gz");
	        if(!f.exists()) return false;
	        f = new File(MNIST_ROOT,"train-labels-idx1-ubyte.gz");
	        if(!f.exists()) return false;
	        f = new File(MNIST_ROOT,"t10k-images-idx3-ubyte.gz");
	        if(!f.exists()) return false;
	        f = new File(MNIST_ROOT,"t10k-labels-idx1-ubyte.gz");
	        if(!f.exists()) return false;
	        return true;
    }

    public MyMnistDataFetcher() throws IOException {
        this(true);
    }

    public void fetch(int numExamples) {
    	 //每一步的大小，batchSize = 128; // batch size for each epoch
        if(!hasMore()) {
            throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");
        }
        float[][] featureData = new float[numExamples][0];
        float[][] labelData = new float[numExamples][0];

        int actualExamples = 0;
        for( int i=0; i<numExamples; i++, cursor++ ){
            if(!hasMore()) break;

            byte[] img = man.readImageUnsafe(order[cursor]);//读取图像数据
            int label = man.readLabel(order[cursor]);//读取答案，标签

            float[] featureVec = new float[img.length];
            featureData[actualExamples] = featureVec;//存储128个样本中的一个图像数据
            labelData[actualExamples] = new float[10];//初始化十个分类数组为0
            labelData[actualExamples][label] = 1.0f;//第label为答案，置为1

            for( int j=0; j<img.length; j++ ){
                float v = ((int)img[j]) & 0xFF; //byte is loaded as signed -> convert to unsigned
                if(binarize){
                	//二值化
                    if(v > 30.0f) featureVec[j] = 1.0f;
                    else featureVec[j] = 0.0f;
                } else {
                	//非二值化，默认选择这个
                    featureVec[j] = v/255.0f;
                }
            }
            actualExamples++;
        }
        if(actualExamples < numExamples){
            featureData = Arrays.copyOfRange(featureData,0,actualExamples);
            labelData = Arrays.copyOfRange(labelData,0,actualExamples);
        }
        INDArray features = Nd4j.create(featureData);
        INDArray labels = Nd4j.create(labelData);
        curr = new DataSet(features,labels);
    }

    @Override
    public void reset() {
        cursor = 0;
        curr = null;
        if(shuffle) MathUtils.shuffleArray(order, rng);
    }

    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }
}
