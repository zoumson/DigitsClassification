/*
* Train SVM using Hog
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
 * @file ConstantsSVM.h
 *
 * @brief Constants 
 *
 * @author Adama Zouma
 * 
 * @Contact: stargue49@gmail.com
 *
 */

#ifndef CONSTANTS_SVM_H
#define CONSTANTS_SVM_H
#include <opencv2/imgproc.hpp>
namespace za {
/* ============================================================================
 * Global Constants
 * ============================================================================
 */

inline constexpr int SZ = 20;
const float affineFlags = cv::WARP_INVERSE_MAP|cv::INTER_LINEAR;
}
#endif	// CONSTANTS_SVM_H



