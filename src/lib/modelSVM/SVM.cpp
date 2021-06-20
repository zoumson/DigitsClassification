#include "SVM.h"


namespace za {

///////////////////////////////////////////////////////////////////////////////////////////////////
void getSVMParams(cv::ml::SVM *svm)
{
    std::cout << "Kernel type     : " << svm->getKernelType() << "\n";
    std::cout << "Type            : " << svm->getType() << "\n";
    std::cout << "C               : " << svm->getC() << "\n";
    std::cout << "Degree          : " << svm->getDegree() << "\n";
    std::cout << "Nu              : " << svm->getNu() <<"\n";
    std::cout << "Gamma           : " << svm->getGamma() << "\n";
}
///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Ptr<cv::ml::SVM> svmInit(float C, float gamma)
{
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(cv::ml::SVM::RBF);
  svm->setType(cv::ml::SVM::C_SVC);

  return svm;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void svmTrain(cv::Ptr<cv::ml::SVM> svm, cv::Mat &trainMat, std::vector<int> &trainLabels, cv::String savedFileName)
{
  cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainMat, cv::ml::ROW_SAMPLE, trainLabels);
  svm->train(td);
  svm->save(savedFileName);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void svmPredict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &testResponse, cv::Mat &testMat )
{
  svm->predict(testMat, testResponse);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void SVMevaluate(cv::Mat &testResponse, float &count, float &accuracy, std::vector<int> &testLabels)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    // std::cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
    if(testResponse.at<float>(i,0) == testLabels[i])
      count = count + 1;
  }
  accuracy = (count/testResponse.rows)*100;
}

}



