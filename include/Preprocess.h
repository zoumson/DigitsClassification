/*
* Preprocess data before training SVM
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
* \see <http://www.lsstcorp.org/LegalNotices/>.
*/

/**
 * @file Preprocess.h
 *
 * @brief Preprocess data before training SVM. 
 *
 * @author Adama Zouma
 * 
 * @Contact: stargue49@gmail.com
 *
 */


#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include "ConstantsSVM.h"
namespace za {
/* ============================================================================
 * Function Declaration
 * ============================================================================
 */

/**
 * \brief Modify an image.
 *
 * \details use opencv moment to modify an image.
 *
 * \param img [in] original image, type is opencv Mat
 *
 * \return type is opencv Mat
 */
cv::Mat deskew(cv::Mat& img);

/**
 * \brief Prepare data to train the model.
 *
 * \details Divide data into training set and testing set.
 *
 * \param pathName [in] MNIST image path name, type is opencv String
 * \param trainCells [in][out] training cells(90%), type is vector of opencv Mat
 * \param testCells [in][out] testing cells(10%), type is vector of opencv Mat
 * \param trainLabels [in][out] 90% of labels belong to training, type is vector of int
 * \param testLabels [in][out] 10% of labels belong to testing, type is vector of int
 * 
 * \return type is void
 * 
 * 
 */
void loadTrainTestLabel(cv::String &pathName, std::vector<cv::Mat> &trainCells, std::vector<cv::Mat> &testCells, std::vector<int> &trainLabels, std::vector<int> &testLabels);

/**
 * \brief Maximize the contrast of an image.
 *
 * \details Maximize the contrast of an image by adding top hat and removing black from it.
 * 
 * \param deskewedTrainCells [in][out] modified training cells(90%), type is vector of opencv Mat
 * \param deskewedTestCells [in][out] modified testing cells(10%), type is vector of opencv Mat
 * \param trainCells [in] training cells(90%), type is vector of opencv Mat
 * \param testCells [in] testing cells(10%), type is vector of opencv Mat
 * 
 * \return type is void
 * 
 */
void CreateDeskewedTrainTest(std::vector<cv::Mat> &deskewedTrainCells,std::vector<cv::Mat> &deskewedTestCells, std::vector<cv::Mat> &trainCells, std::vector<cv::Mat> &testCells);
}

#endif	// PREPROCESS_H