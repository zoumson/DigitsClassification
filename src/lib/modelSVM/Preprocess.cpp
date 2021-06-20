// DetectChars.cpp

#include "Preprocess.h"
#include "ConstantsSVM.h"


namespace za {

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat deskew(cv::Mat& img)
{
    cv::Moments m = moments(img);
    if(abs(m.mu02) < 1e-2)
    {
        return img.clone();
    }
    float skew = m.mu11/m.mu02;
    cv::Mat warpMat = (cv::Mat_<float>(2,3) << 1, skew, -0.5*za::SZ*skew, 0, 1, 0);
    cv::Mat imgOut = cv::Mat::zeros(img.rows, img.cols, img.type());
    cv::warpAffine(img, imgOut, warpMat, imgOut.size(), za::affineFlags);

    return imgOut;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void loadTrainTestLabel(cv::String &pathName, std::vector<cv::Mat> &trainCells, 
    std::vector<cv::Mat> &testCells,std::vector<int> &trainLabels, std::vector<int> &testLabels)
{
    cv::Mat img = cv::imread(pathName, cv::IMREAD_GRAYSCALE);
    int ImgCount = 0;
    for(int i = 0; i < img.rows; i = i + za::SZ)
    {
        for(int j = 0; j < img.cols; j = j + za::SZ)
        {
            cv::Mat digitImg = (img.colRange(j,j+za::SZ).rowRange(i,i+za::SZ)).clone();
            if(j < int(0.9*img.cols))
            {
                trainCells.push_back(digitImg);
            }
            else
            {
                testCells.push_back(digitImg);
            }
            ImgCount++;
        }
    }

    std::cout << "Image Count : " << ImgCount <<"\n";
    float digitClassNumber = 0;

    for(int z=0;z<int(0.9*ImgCount);z++)
    {
        if(z % 450 == 0 && z != 0)
        {
            digitClassNumber = digitClassNumber + 1;
        }
        trainLabels.push_back(digitClassNumber);
    }
    digitClassNumber = 0;
    for(int z=0;z<int(0.1*ImgCount);z++)
    {
        if(z % 50 == 0 && z != 0)
        {
            digitClassNumber = digitClassNumber + 1;
        }
        testLabels.push_back(digitClassNumber);
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void CreateDeskewedTrainTest(std::vector<cv::Mat> &deskewedTrainCells,std::vector<cv::Mat> &deskewedTestCells, 
    std::vector<cv::Mat> &trainCells, std::vector<cv::Mat> &testCells)
{

    for(int i=0;i<trainCells.size();i++)
    {

        cv::Mat deskewedImg = deskew(trainCells[i]);
        deskewedTrainCells.push_back(deskewedImg);
    }

    for(int i=0;i<testCells.size();i++)
    {

        cv::Mat deskewedImg = deskew(testCells[i]);
        deskewedTestCells.push_back(deskewedImg);
    }
}


}















