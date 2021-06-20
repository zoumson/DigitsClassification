/*
* Hog 
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
 * @file Hog.h
 *
 * @brief Hog Feature Extraction for image classification. 
 *
 * @author Adama Zouma
 * 
 * @Contact: stargue49@gmail.com
 *
 */



#ifndef HOG_H
#define HOG_H
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
namespace za {
/* ============================================================================
 * Global Constants
 * ============================================================================
 */

extern cv::HOGDescriptor hog;

/* ============================================================================
 * Function Declaration
 * ============================================================================
 */

/**
 * \brief Create Hog feature set.
 *
 * \details Create a multi-dimensional vector using Ho approach.
 *
 * \param trainHOG [in][out] training vectors, type is vector of vector of float
 * \param testHOG [in][out] testing vectors, type is vector of vector of float
 * \param deskewedTrainCells [in] modified training cells(90%), type is vector of opencv Mat
 * \param deskewedTestCells [in] modified testing cells(10%), type is vector of opencv Mat
 * 
 * \return type is void
 * 
 * 
 */
void CreateTrainTestHOG(std::vector<std::vector<float> > &trainHOG, std::vector<std::vector<float> > &testHOG, std::vector<cv::Mat> &deskewedtrainCells, std::vector<cv::Mat> &deskewedtestCells);

/**
 * \brief Transform Hog vectors to opencv Matrix.
 *
 * \details Each vector is used to build a matrix.
 * 
 * \param trainHOG [in] training vectors, type is vector of vector of float
 * \param testHOG [in] testing vectors, type is vector of vector of float
 * \param trainMat [in][out] training matrix, type is vector of opencv Mat
 * \param testMat [in][out] testing matrix, type is vector of opencv Mat
 * 
 * \return type is void
 * 
 */
void ConvertVectortoMatrix(std::vector<std::vector<float> > &trainHOG, std::vector<std::vector<float> > &testHOG, cv::Mat &trainMat, cv::Mat &testMat);


}
#endif	// HOG_H
