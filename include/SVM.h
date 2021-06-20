
/*
* SVM 
* See COPYRIGHT file at the top of the source tree.
*
* This product includes software developed by the
* STARGUE Project (http://www.stargue.org/).
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the STARGUE License Statement and
* the GNU General Public License along with this program. If not,
* see <http://www.lsstcorp.org/LegalNotices/>.
*/

/**
 * @file SVM.h
 *
 * @brief Hog Feature Extraction for image classification. 
 *
 * @author Adama Zouma
 * 
 * @Contact: stargue49@gmail.com
 *
 */



#ifndef SVM_H
#define SVM_H
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
namespace za {

/* ============================================================================
 * Function Declaration
 * ============================================================================
 */

/**
 * \brief SVM specs.
 *
 * \details Show the current set SVM parameters.
 *
 * \param svm [in][out] svm object, type is opencv manchine learning SVM
 * 
 * \return type is void
 * 
 * 
 */
void getSVMParams(cv::ml::SVM *svm);

/**
 * \brief Initialize SVM.
 *
 * \details Set C and gamma of SVM.
 * 
 * \param C [in] SVM C, type is float
 * \param gamma [in] SVM gamma, type is float
 * 
 * \return updated SVM object
 * 
 */
cv::Ptr<cv::ml::SVM> svmInit(float C, float gamma);

/**
 * \brief Transform Hog vectors to opencv Matrix.
 *
 * \details Each vector is used to build a matrix.
 * 
 * \param svm [in] training vectors, type is vector of vector of float
 * \param trainMat [in] training matrix feature, type is vector of opencv Mat
 * \param trainLabels [in][out] training vector labels, type is vector of int
 * \param savedFileName [in][out] SVM model file storage, type is opencv String
 * 
 * \return type is void
 * 
 */
void svmTrain(cv::Ptr<cv::ml::SVM> svm, cv::Mat &trainMat, std::vector<int> &trainLabels, cv::String savedFileName);

/**
 * \brief SVM prediction.
 *
 * \details Feed input to model and get the response.
 * 
 * \param svm [in] training vectors, type is vector of vector of float
 * \param testResponse [in][out] testing response, type is opencv Mat
 * \param testMat [in][out] testing matrix, type is opencv Mat
 * 
 * \return type is void
 * 
 */
void svmPredict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &testResponse, cv::Mat &testMat );

/**
 * \brief Evaluate SVM model.
 *
 * \details Each vector is used to build a matrix.
 * 
 * \param testResponse [in][out] testing response, type is opencv Mat
 * \param count [in] count, type is float
 * \param accuracy [in][out] SVM model accuracy, type is float
 * \param testLabels [in][out] testing labels, type is vector of int
 * 
 * \return type is void
 * 
 */
void SVMevaluate(cv::Mat &testResponse, float &count, float &accuracy, std::vector<int> &testLabels);


}
#endif	// SVM_H














