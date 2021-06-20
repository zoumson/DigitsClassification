#include "Hog.h"



namespace za {

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::HOGDescriptor hog(
    cv::Size(20,20), //winSize
    cv::Size(8,8), //blocksize
    cv::Size(4,4), //blockStride,
    cv::Size(8,8), //cellSize,
    9,   //nbins,
    1,   //derivAper,
    -1,  //winSigma,
    cv::HOGDescriptor::HistogramNormType::L2Hys, //histogramNormType,
    0.2, //L2HysThresh,
    0,   //gammal correction,
    64,  //nlevels=64
    1
);

///////////////////////////////////////////////////////////////////////////////////////////////////
void CreateTrainTestHOG(std::vector<std::vector<float> > &trainHOG, std::vector<std::vector<float> > &testHOG, 
    std::vector<cv::Mat> &deskewedtrainCells, std::vector<cv::Mat> &deskewedtestCells)
{
    for(int y=0;y<deskewedtrainCells.size();y++)
    {
        std::vector<float> descriptors;
        hog.compute(deskewedtrainCells[y],descriptors);
        trainHOG.push_back(descriptors);
    }

    for(int y=0;y<deskewedtestCells.size();y++){

        std::vector<float> descriptors;
        hog.compute(deskewedtestCells[y],descriptors);
        testHOG.push_back(descriptors);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ConvertVectortoMatrix(std::vector<std::vector<float> > &trainHOG, std::vector<std::vector<float> > &testHOG, cv::Mat &trainMat, cv::Mat &testMat)
{
    int descriptor_size = trainHOG[0].size();

    for(int i = 0;i<trainHOG.size();i++)
    {
        for(int j = 0;j<descriptor_size;j++)
        {
           trainMat.at<float>(i,j) = trainHOG[i][j];
        }
    }
    for(int i = 0;i<testHOG.size();i++)
    {
        for(int j = 0;j<descriptor_size;j++)
        {
            testMat.at<float>(i,j) = testHOG[i][j];
        }
    }
}

}








