
#include "Preprocess.h"
#include "ConstantsSVM.h"
#include "Hog.h"
#include "SVM.h"
int main(int argc, char** argv)
{

   cv::String keys =
        "{i image |<none>           | image path}"                                                                                                            
        "{s save |./result/eyeGlassClassifierModel.yml           | save train file name}"                                                                                                            
        "{help h usage ?    |      | show help message}";      
  
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Digits Classification");
    if (parser.has("help")) 
    {
        parser.printMessage();
        return 0;
    }
    
    cv::String pathName = parser.get<cv::String>("image"); 
    cv::String savedFile = parser.get<cv::String>("save"); 


    if (!parser.check()) 
    {
        parser.printErrors();
        return -1;
    }
    std::vector<cv::Mat> trainCells;
    std::vector<cv::Mat> testCells;
    std::vector<int> trainLabels;
    std::vector<int> testLabels;
    za::loadTrainTestLabel(pathName,trainCells,testCells,trainLabels,testLabels);

    std::vector<cv::Mat> deskewedTrainCells;
    std::vector<cv::Mat> deskewedTestCells;
    za::CreateDeskewedTrainTest(deskewedTrainCells,deskewedTestCells,trainCells,testCells);

    std::vector<std::vector<float> > trainHOG;
    std::vector<std::vector<float> > testHOG;
    za::CreateTrainTestHOG(trainHOG,testHOG,deskewedTrainCells,deskewedTestCells);

    int descriptor_size = trainHOG[0].size();
    std::cout << "Descriptor Size : " << descriptor_size << "\n";

    cv::Mat trainMat(trainHOG.size(),descriptor_size, CV_32FC1);
 
    cv::Mat testMat(testHOG.size(),descriptor_size, CV_32FC1);

    za::ConvertVectortoMatrix(trainHOG,testHOG,trainMat,testMat);

    float C = 12.5, gamma = 0.5;

    cv::Mat testResponse;
    cv::Ptr<cv::ml::SVM> model = za::svmInit(C, gamma);
    ///////////  SVM Training  ////////////////
    za::svmTrain(model, trainMat, trainLabels, savedFile);
    ///////////  SVM Testing  ////////////////
    za::svmPredict(model, testResponse, testMat);
    ////////////// Find Accuracy   ///////////
    float count = 0;
    float accuracy = 0 ;
    za::getSVMParams(model);
    za::SVMevaluate(testResponse, count, accuracy, testLabels);
    std::cout << "the accuracy is :" << accuracy <<"\n";
    return 0;
}
