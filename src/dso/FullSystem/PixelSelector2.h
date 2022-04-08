/*
 * @Author: Jianheng Liu
 * @Date: 2022-04-02 17:18:27
 * @LastEditors: Jianheng Liu
 * @LastEditTime: 2022-04-03 11:42:16
 * @Description: Description
 */
/**
* This file is part of DSO, written by Jakob Engel.
* It has been modified by Lukas von Stumberg for the inclusion in DM-VIO (http://vision.in.tum.de/dm-vio).
*
* Copyright 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
 
#include "util/NumType.h"

namespace dso
{

enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};


class FrameHessian;

class PixelSelector
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	int makeMaps(
			const FrameHessian* const fh,
			float* map_out, float density, int recursionsLeft=1, bool plot=false, float thFactor=1);

	PixelSelector(int w, int h);
	~PixelSelector();
	int currentPotential; // =3;当前选择像素点的潜力, 就是网格大小, 越大选点越少


	bool allowFast;
	void makeHists(const FrameHessian* const fh);
private:

	Eigen::Vector3i select(const FrameHessian* const fh,
			float* map_out, int pot, float thFactor=1);


	unsigned char* randomPattern;


	int* gradHist;//!< 根号梯度平方根分布直方图, 0是所有像素个数
	float* ths;//!< 平滑之前的阈值,得到每一block的阈值=每一个块梯度平方根中位数+7
	float* thsSmoothed;//!< 平滑后的阈值(梯度平方和)
	int thsStep;
	const FrameHessian* gradHistFrame;

	// = 16;block width, and block height.
	int bW, bH;
	// number of blocks in x and y dimension.
	int nbW, nbH;
};




}

