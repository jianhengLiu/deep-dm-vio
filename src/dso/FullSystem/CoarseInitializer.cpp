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

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/Residuals.h"
#include "util/nanoflann.h"

#include <opencv2/highgui/highgui.hpp>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

CoarseInitializer::CoarseInitializer(int ww, int hh)
    : thisToNext_aff(0, 0)
    , thisToNext(SE3())
{
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        points[lvl] = 0;
        numPoints[lvl] = 0;
    }

    JbBuffer = new Vec10f[ww * hh];
    JbBuffer_new = new Vec10f[ww * hh];

    frameID = -1;
    fixAffine = true;
    printDebug = false;

    wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
    wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
    wM.diagonal()[6] = SCALE_A;
    wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        if (points[lvl] != 0)
            delete[] points[lvl];
    }

    delete[] JbBuffer;
    delete[] JbBuffer_new;
}

/**
 * @brief 纯视觉追踪估计最新帧位姿,至少5帧
 *
 * @param newFrameHessian
 * @param wraps
 * @return true
 * @return false
 */
bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*>& wraps)
{
    newFrame = newFrameHessian;

    //[ ***step 1*** ] 先显示新来的帧
    // 新的一帧, 在跟踪之前显示的
    for (IOWrap::Output3DWrapper* ow : wraps) {
        ow->pushLiveFrame(newFrameHessian);
        ow->pushLiveFeatureFrame(newFrameHessian);
    }

    int maxIterations[] = { 5, 5, 10, 30, 50 };

    alphaK = 2.5 * 2.5; //*freeDebugParam1*freeDebugParam1;
    alphaW = 150 * 150; //*freeDebugParam2*freeDebugParam2;
    regWeight = 0.8; //*freeDebugParam4;
    couplingWeight = 1; //*freeDebugParam5;

    //[ ***step 2*** ] 初始化每个点逆深度为1, 初始化光度参数, 位姿SE3
    if (!snapped) // 应该指的是位移足够大了，不够大就重新优化
    {
        // 初始化
        thisToNext.translation().setZero();
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            int npts = numPoints[lvl];
            Pnt* ptsl = points[lvl];
            for (int i = 0; i < npts; i++) {
                ptsl[i].iR = 1;
                ptsl[i].idepth_new = 1;
                ptsl[i].lastHessian = 0;
            }
        }
    }

    SE3 refToNew_current = thisToNext;

    AffLight refToNew_aff_current = thisToNext_aff;

    // 如果都有仿射系数, 则估计一个初值
    if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
        refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure), 0); // coarse approximation.

    // 从最顶层到最低层(粗到精)优化
    Vec3f latestRes = Vec3f::Zero();
    for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--) {
        //[ ***step 3*** ] 使用计算过的上一层来初始化下一层
        // 顶层未初始化到, reset来完成
        if (lvl < pyrLevelsUsed - 1)
            propagateDown(lvl + 1);

        Mat88f H, Hsc;
        Vec8f b, bsc;
        resetPoints(lvl); // 这里对顶层进行初始化!
        //[ ***step 4*** ] 迭代之前计算能量, Hessian等
        Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
        applyStep(lvl); // 新的能量付给旧的

        float lambda = 0.1; // LM
        float eps = 1e-4;
        int fails = 0;

        if (printDebug) {
            printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
                lvl, 0, lambda,
                "INITIA",
                sqrtf((float)(resOld[0] / resOld[2])),
                sqrtf((float)(resOld[1] / resOld[2])),
                sqrtf((float)(resOld[0] / resOld[2])),
                sqrtf((float)(resOld[1] / resOld[2])),
                (resOld[0] + resOld[1]) / resOld[2],
                (resOld[0] + resOld[1]) / resOld[2],
                0.0f);
            std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() << "\n";
        }

        //[ ***step 5*** ] 迭代求解
        int iteration = 0;
        while (true) {
            //[ ***step 5.1*** ] 计算边缘化后的Hessian矩阵, 以及一些骚操作
            Mat88f Hl = H;
            for (int i = 0; i < 8; i++)
                Hl(i, i) *= (1 + lambda);
            // 舒尔补, 边缘化掉特征点(逆深度)状态; 用来加速求解(维度下降(=8=6(位姿)+2(光参)))
            Hl -= Hsc * (1 / (1 + lambda)); // 因为dd必定是对角线上的, 所以也乘倒数
            Vec8f bl = b - bsc * (1 / (1 + lambda));

            Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
            bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

            //[ ***step 5.2*** ] 求解增量inc
            Vec8f inc;
            SE3 refToNew_new;
            // TODO: 删除光度参数
            if (fixAffine) // 固定光度参数
            {
                // Note as we set the weights of rotation and translation to 1 the wM is just the identity in this case.
                // LDL分解求解线性方程: https://blog.csdn.net/qq_39400324/article/details/123457380
                inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() * (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                inc.tail<2>().setZero(); // 不优化光度参数,所以变化量为0
            } else
                inc = -(wM * (Hl.ldlt().solve(bl)));

            double incNorm = inc.norm();

            //[ ***step 5.3*** ] 更新状态, doStep中更新逆深度
            refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current; // 更新位姿

            // TODO: 删除光度参数
            AffLight refToNew_aff_new = refToNew_aff_current;
            refToNew_aff_new.a += inc[6];
            refToNew_aff_new.b += inc[7];
            doStep(lvl, lambda, inc); //更新逆深度

            //[ ***step 5.4*** ] 计算更新后的能量并且与旧的对比判断是否accept
            Mat88f H_new, Hsc_new;
            Vec8f b_new, bsc_new;
            Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
            Vec3f regEnergy = calcEC(lvl);

            float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
            float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

            bool accept = eTotalOld > eTotalNew;

            if (printDebug) {
                printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
                    lvl, iteration, lambda,
                    (accept ? "ACCEPT" : "REJECT"),
                    sqrtf((float)(resOld[0] / resOld[2])),
                    sqrtf((float)(regEnergy[0] / regEnergy[2])),
                    sqrtf((float)(resOld[1] / resOld[2])),
                    sqrtf((float)(resNew[0] / resNew[2])),
                    sqrtf((float)(regEnergy[1] / regEnergy[2])),
                    sqrtf((float)(resNew[1] / resNew[2])),
                    eTotalOld / resNew[2],
                    eTotalNew / resNew[2],
                    incNorm);
                std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() << "\n";
            }

            //[ ***step 5.5*** ] 接受的话, 更新状态,; 不接受则增大lambda
            if (accept) {
                if (resNew[1] == alphaK * numPoints[lvl])
                    snapped = true;
                H = H_new;
                b = b_new;
                Hsc = Hsc_new;
                bsc = bsc_new;
                resOld = resNew;
                refToNew_aff_current = refToNew_aff_new;
                refToNew_current = refToNew_new;
                applyStep(lvl);
                optReg(lvl);
                lambda *= 0.5;
                fails = 0;
                if (lambda < 0.0001)
                    lambda = 0.0001;
            } else {
                fails++;
                lambda *= 4;
                if (lambda > 10000)
                    lambda = 10000;
            }

            bool quitOpt = false;
            // 迭代停止条件, 收敛/大于最大次数/失败2次以上
            if (!(incNorm > eps) || iteration >= maxIterations[lvl] || fails >= 2) {
                Mat88f H, Hsc;
                Vec8f b, bsc;

                quitOpt = true;
            }

            if (quitOpt)
                break;
            iteration++;
        }
        latestRes = resOld;
    }

    //[ ***step 6*** ] 优化后赋值位姿, 从底层计算上层点的深度
    thisToNext = refToNew_current;
    thisToNext_aff = refToNew_aff_current;

    for (int i = 0; i < pyrLevelsUsed - 1; i++)
        propagateUp(i);

    frameID++;
    if (!snapped)
        snappedAt = 0;

    if (snapped && snappedAt == 0)
        snappedAt = frameID; // 位移足够的帧数

    debugPlot(0, wraps);

    // 位移足够大, 再优化5帧才行
    return snapped && frameID > snappedAt + 5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*>& wraps)
{
    bool needCall = false;
    for (IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if (!needCall)
        return;

    int wl = w[lvl], hl = h[lvl];
    Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

    MinimalImageB3 iRImg(wl, hl);

    for (int i = 0; i < wl * hl; i++)
        iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);

    int npts = numPoints[lvl];

    float nid = 0, sid = 0;
    for (int i = 0; i < npts; i++) {
        Pnt* point = points[lvl] + i;
        if (point->isGood) {
            nid++;
            sid += point->iR;
        }
    }
    float fac = nid / sid;

    for (int i = 0; i < npts; i++) {
        Pnt* point = points[lvl] + i;

        if (!point->isGood)
            iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));

        else
            iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));
    }

    // IOWrap::displayImage("idepth-R", &iRImg, false);
    for (IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}

// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
//* 计算能量函数和Hessian矩阵, 以及舒尔补, sc代表Schur
Vec3f CoarseInitializer::calcResAndGS(
    int lvl, Mat88f& H_out, Vec8f& b_out,
    Mat88f& H_out_sc, Vec8f& b_out_sc,
    const SE3& refToNew, AffLight refToNew_aff,
    bool plot)
{
    int wl = w[lvl], hl = h[lvl];
    // 当前层图像及梯度
    Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
    Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

    //! 旋转矩阵R * 内参矩阵K_inv
    Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
    Vec3f t = refToNew.translation().cast<float>();
    Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b); // 光度参数

    float fxl = fx[lvl];
    float fyl = fy[lvl];
    float cxl = cx[lvl];
    float cyl = cy[lvl];

    Accumulator11 E; // 1*1 的累加器
    acc9.initialize(); // 初始值, 分配空间
    E.initialize();

    int npts = numPoints[lvl];
    Pnt* ptsl = points[lvl];
    for (int i = 0; i < npts; i++) {

        Pnt* point = ptsl + i;

        point->maxstep = 1e10;
        if (!point->isGood) {
            E.updateSingle((float)(point->energy[0])); // 累加
            point->energy_new = point->energy;
            point->isGood_new = false;
            continue;
        }

        VecNRf dp0; // 8*1矩阵, 每个点附近的残差个数为8个(模板)
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        VecNRf dp6;
        VecNRf dp7;
        VecNRf dd;
        VecNRf r;
        JbBuffer_new[i].setZero(); // 10*1 向量

        // sum over all residuals.
        bool isGood = true;
        float energy = 0;
        for (int idx = 0; idx < patternNum; idx++) {
            int dx = patternP[idx][0];
            int dy = patternP[idx][1];

            //! Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
            Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new; // 1.1
            // 归一化坐标 Pj  //1.2
            float u = pt[0] / pt[2];
            float v = pt[1] / pt[2];
            // 像素坐标pj  //1.3
            float Ku = fxl * u + cxl;
            float Kv = fyl * v + cyl;
            // dpi/pz' // 1.4
            float new_idepth = point->idepth_new / pt[2]; // 新一帧上的逆深度

            // 落在边缘附近，深度小于0, 则不好
            if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0)) {
                isGood = false;
                break;
            }

            // 插值得到新图像中的 patch 像素值，(输入3维，输出3维[像素值, x方向梯度, y方向梯度])
            Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
            // Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

            // 参考帧上的 patch 上的像素值, 输出一维像素值
            // float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
            float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

            if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
                isGood = false;
                break;
            }

            // 残差
            // TODO: 替换为特征图残差,同时不再需要光度参数 r2new_aff
            float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
            // Huber权重: 等价的二倍huber函数
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            // robust energy
            energy += hw * residual * residual * (2 - hw);

            // SLAM十四讲ver.2(p218)
            // uj 对 逆深度 di 求导, https://zhuanlan.zhihu.com/p/29177540
            //! 1/Pz * (tx - u*tz), u = px/pz
            float dxdd = (t[0] - t[2] * u) / pt[2];
            //! 1/Pz * (ty - v*tz), u = py/pz
            float dydd = (t[1] - t[2] * v) / pt[2];

            if (hw < 1)
                hw = sqrtf(hw);
            //! hw*dx*fx, hw*dy*fy //2.1
            float dxInterp = hw * hitColor[1] * fxl;
            float dyInterp = hw * hitColor[2] * fyl;
            //* 残差对 j(新状态) 位姿求导,
            // 李代数xi(x,y,z,phi_x,phi_y,phi_z)
            dp0[idx] = new_idepth * dxInterp; // dI/dxi_0:1/Z*hw*dx*fx
            dp1[idx] = new_idepth * dyInterp; // dI/dxi_1
            dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp); // dI/dxi_2
            dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;
            dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
            dp5[idx] = -v * dxInterp + u * dyInterp;
            //* 残差对光度参数求导 // TODO: 不需要
            dp6[idx] = -hw * r2new_aff[0] * rlR;
            dp7[idx] = -hw * 1;
            //* 残差对 i(旧状态) 逆深度求导
            dd[idx] = dxInterp * dxdd + dyInterp * dydd;
            r[idx] = hw * residual;

            float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
            if (maxstep < point->maxstep)
                point->maxstep = maxstep;

            // immediately compute dp*dd' and dd*dd' in JbBuffer1.
            //* 计算Hessian关于逆深度那一行
            // 用来计算舒尔补
            JbBuffer_new[i][0] += dp0[idx] * dd[idx];
            JbBuffer_new[i][1] += dp1[idx] * dd[idx];
            JbBuffer_new[i][2] += dp2[idx] * dd[idx];
            JbBuffer_new[i][3] += dp3[idx] * dd[idx];
            JbBuffer_new[i][4] += dp4[idx] * dd[idx];
            JbBuffer_new[i][5] += dp5[idx] * dd[idx];
            JbBuffer_new[i][6] += dp6[idx] * dd[idx];
            JbBuffer_new[i][7] += dp7[idx] * dd[idx];
            JbBuffer_new[i][8] += r[idx] * dd[idx];
            JbBuffer_new[i][9] += dd[idx] * dd[idx];
        }

        // 如果点的pattern(其中一个像素)超出图像,像素值无穷, 或者残差大于阈值
        if (!isGood || energy > point->outlierTH * 20) {
            E.updateSingle((float)(point->energy[0])); // 上一帧的加进来
            point->isGood_new = false;
            point->energy_new = point->energy; //上一次的给当前次的
            continue;
        }

        // 内点则加进能量函数
        // add into energy.
        E.updateSingle(energy);
        point->isGood_new = true;
        point->energy_new[0] = energy;

        //! 因为使用128位相当于每次加4个数, 因此i+=4, 妙啊!
        // H += J^T * J
        // update Hessian matrix.
        for (int i = 0; i + 3 < patternNum; i += 4)
            acc9.updateSSE(
                _mm_load_ps(((float*)(&dp0)) + i),
                _mm_load_ps(((float*)(&dp1)) + i),
                _mm_load_ps(((float*)(&dp2)) + i),
                _mm_load_ps(((float*)(&dp3)) + i),
                _mm_load_ps(((float*)(&dp4)) + i),
                _mm_load_ps(((float*)(&dp5)) + i),
                _mm_load_ps(((float*)(&dp6)) + i),
                _mm_load_ps(((float*)(&dp7)) + i),
                _mm_load_ps(((float*)(&r)) + i));

        // 加0, 4, 8后面多余的值, 因为SSE2是以128为单位相加, 多余的单独加
        for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
            acc9.updateSingle(
                (float)dp0[i], (float)dp1[i], (float)dp2[i], (float)dp3[i],
                (float)dp4[i], (float)dp5[i], (float)dp6[i], (float)dp7[i],
                (float)r[i]);
    }

    E.finish();
    acc9.finish();

    // calculate alpha energy, and decide if we cap it.
    Accumulator11 EAlpha;
    EAlpha.initialize();
    for (int i = 0; i < npts; i++) {
        Pnt* point = ptsl + i;
        if (!point->isGood_new) {
            E.updateSingle((float)(point->energy[1]));
        } else {
            point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1); // 深度的正则项?
            E.updateSingle((float)(point->energy_new[1]));
        }
    }
    EAlpha.finish();
    float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts); // 平移越大, 越容易初始化成功?

    // printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);

    // compute alpha opt.
    float alphaOpt;
    if (alphaEnergy > alphaK * npts) // 若帧间位移足够,则设置alphaEnergy,后续snapped将变为true
    {
        alphaOpt = 0;
        alphaEnergy = alphaK * npts;
    } else {
        alphaOpt = alphaW;
    }

    // 计算Hessian舒而补
    acc9SC.initialize();
    for (int i = 0; i < npts; i++) {
        Pnt* point = ptsl + i;
        if (!point->isGood_new)
            continue;

        point->lastHessian_new = JbBuffer_new[i][9]; // 对逆深度 dd*dd

        JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1);
        JbBuffer_new[i][9] += alphaOpt;

        if (alphaOpt == 0) {
            JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
            JbBuffer_new[i][9] += couplingWeight;
        }

        JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]); // 取逆是协方差，做权重
        //* 9做权重, 计算的是舒尔补项!
        //! dp*dd*(dd^2)^-1*dd*dp
        acc9SC.updateSingleWeighted(
            (float)JbBuffer_new[i][0], (float)JbBuffer_new[i][1], (float)JbBuffer_new[i][2], (float)JbBuffer_new[i][3],
            (float)JbBuffer_new[i][4], (float)JbBuffer_new[i][5], (float)JbBuffer_new[i][6], (float)JbBuffer_new[i][7],
            (float)JbBuffer_new[i][8], (float)JbBuffer_new[i][9]);
    }
    acc9SC.finish();

    // printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
    H_out = acc9.H.topLeftCorner<8, 8>(); // / acc9.num;  		!dp^T*dp
    b_out = acc9.H.topRightCorner<8, 1>(); // / acc9.num;; 		!dp^T*r
    H_out_sc = acc9SC.H.topLeftCorner<8, 8>(); // / acc9.num; 	!(dp*dd)^T*(dd*dd)^-1*(dd*dp)
    b_out_sc = acc9SC.H.topRightCorner<8, 1>(); // / acc9.num;	!(dp*dd)^T*(dd*dd)^-1*(dp^T*r)

    H_out(0, 0) += alphaOpt * npts;
    H_out(1, 1) += alphaOpt * npts;
    H_out(2, 2) += alphaOpt * npts;

    Vec3f tlog = refToNew.log().head<3>().cast<float>(); // 李代数, 平移部分 (上一次的位姿值)
    b_out[0] += tlog[0] * alphaOpt * npts;
    b_out[1] += tlog[1] * alphaOpt * npts;
    b_out[2] += tlog[2] * alphaOpt * npts;

    // Add zero prior to translation.
    // setting_weightZeroPriorDSOInitY is the squared weight of the prior residual.
    H_out(1, 1) += setting_weightZeroPriorDSOInitY;
    b_out(1) += setting_weightZeroPriorDSOInitY * refToNew.translation().y();

    H_out(0, 0) += setting_weightZeroPriorDSOInitX;
    b_out(0) += setting_weightZeroPriorDSOInitX * refToNew.translation().x();

    return Vec3f(E.A, alphaEnergy, E.num);
}

float CoarseInitializer::rescale()
{
    float factor = 20 * thisToNext.translation().norm();
    //	float factori = 1.0f/factor;
    //	float factori2 = factori*factori;
    //
    //	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
    //	{
    //		int npts = numPoints[lvl];
    //		Pnt* ptsl = points[lvl];
    //		for(int i=0;i<npts;i++)
    //		{
    //			ptsl[i].iR *= factor;
    //			ptsl[i].idepth_new *= factor;
    //			ptsl[i].lastHessian *= factori2;
    //		}
    //	}
    //	thisToNext.translation() *= factori;

    return factor;
}

//* 计算旧的和新的逆深度与iR的差值, 返回旧的差, 新的差, 数目
//? iR到底是啥呢     答：IR是逆深度的均值，尺度收敛到IR
Vec3f CoarseInitializer::calcEC(int lvl)
{
    if (!snapped)
        return Vec3f(0, 0, numPoints[lvl]);
    AccumulatorX<2> E;
    E.initialize();
    int npts = numPoints[lvl];
    for (int i = 0; i < npts; i++) {
        Pnt* point = points[lvl] + i;
        if (!point->isGood_new)
            continue;
        float rOld = (point->idepth - point->iR);
        float rNew = (point->idepth_new - point->iR);
        E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));

        // printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
    }
    E.finish();

    // printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
    return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
}

/**
 * @brief 使用最近点来更新每个点的iR, smooth的感觉
 *
 * @param lvl
 */
void CoarseInitializer::optReg(int lvl)
{
    int npts = numPoints[lvl];
    Pnt* ptsl = points[lvl];
    if (!snapped) {
        return;
    }

    for (int i = 0; i < npts; i++) {
        Pnt* point = ptsl + i;
        if (!point->isGood)
            continue;

        float idnn[10];
        int nnn = 0;
        // 获得当前点周围最近10个点, 质量好的点的iR
        for (int j = 0; j < 10; j++) {
            if (point->neighbours[j] == -1)
                continue;
            Pnt* other = ptsl + point->neighbours[j];
            if (!other->isGood)
                continue;
            idnn[nnn] = other->iR;
            nnn++;
        }

        // 与最近点中位数进行加权获得新的iR
        if (nnn > 2) {
            std::nth_element(idnn, idnn + nnn / 2, idnn + nnn); // 获得中位数
            point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
        }
    }
}

//* 使用归一化积来更新高层逆深度值
void CoarseInitializer::propagateUp(int srcLvl)
{
    assert(srcLvl + 1 < pyrLevelsUsed);
    // set idepth of target

    int nptss = numPoints[srcLvl];
    int nptst = numPoints[srcLvl + 1];
    Pnt* ptss = points[srcLvl];
    Pnt* ptst = points[srcLvl + 1];

    // set to zero.
    for (int i = 0; i < nptst; i++) {
        Pnt* parent = ptst + i;
        parent->iR = 0;
        parent->iRSumNum = 0;
    }
    //* 更新在上一层的parent
    for (int i = 0; i < nptss; i++) {
        Pnt* point = ptss + i;
        if (!point->isGood)
            continue;

        Pnt* parent = ptst + point->parent;
        parent->iR += point->iR * point->lastHessian; //! 均值*信息矩阵 ∑ (sigma*u)
        parent->iRSumNum += point->lastHessian; //! 新的信息矩阵 ∑ sigma
    }

    for (int i = 0; i < nptst; i++) {
        Pnt* parent = ptst + i;
        if (parent->iRSumNum > 0) {
            parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
            //! 高斯归一化积后的均值
            parent->isGood = true;
        }
    }

    optReg(srcLvl + 1); // 使用附近的点来更新IR和逆深度
}

void CoarseInitializer::propagateDown(int srcLvl)
{
    assert(srcLvl > 0);
    // set idepth of target

    int nptst = numPoints[srcLvl - 1]; // 当前层的点数目
    Pnt* ptss = points[srcLvl]; // 当前层+1, 上一层的点集
    Pnt* ptst = points[srcLvl - 1]; // 当前层点集

    for (int i = 0; i < nptst; i++) {
        Pnt* point = ptst + i; // 遍历当前层的点
        Pnt* parent = ptss + point->parent; // 找到当前点的parrent

        if (!parent->isGood || parent->lastHessian < 0.1)
            continue;
        if (!point->isGood) {
            // 当前点不好, 则把父点的值直接给它, 并且置位good
            point->iR = point->idepth = point->idepth_new = parent->iR;
            point->isGood = true;
            point->lastHessian = 0;
        } else {
            // 通过hessian给point和parent加权求得新的iR
            // iR可以看做是深度的值, 使用的高斯归一化积, Hessian是信息矩阵
            float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) / (point->lastHessian * 2 + parent->lastHessian);
            point->iR = point->idepth = point->idepth_new = newiR;
        }
    }
    //? 为什么在这里又更新了iR, 没有更新 idepth
    // 感觉更多的是考虑附近点的平滑效果
    optReg(srcLvl - 1); // 当前层
}

void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
    for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
        int lvlm1 = lvl - 1;
        int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

        Eigen::Vector3f* dINew_l = data[lvl];
        Eigen::Vector3f* dINew_lm = data[lvlm1];

        for (int y = 0; y < hl; y++)
            for (int x = 0; x < wl; x++)
                dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] + dINew_lm[2 * x + 1 + 2 * y * wlm1][0] + dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] + dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

        for (int idx = wl; idx < wl * (hl - 1); idx++) {
            dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
            dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
        }
    }
}

/**
 * @brief 提取候选点
 * TODO: 考虑这个函数的必要性,是否过于复杂了?
 * @param HCalib
 * @param newFrameHessian
 */
void CoarseInitializer::setFirst(CalibHessian* HCalib, FrameHessian* newFrameHessian)
{

    //[ ***step 1*** ] 计算图像每层的内参
    makeK(HCalib);
    firstFrame = newFrameHessian;

    PixelSelector sel(w[0], h[0]);

    float* statusMap = new float[w[0] * h[0]];
    bool* statusMapB = new bool[w[0] * h[0]];

    float densities[] = { 0.03, 0.05, 0.15, 0.5, 1 }; // 不同层取得点密度
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        //[ ***step 2*** ] 针对不同层数选择大梯度像素, 第0层比较复杂1d,
        // 2d,4d大小block来选择3个层次的像素
        sel.currentPotential = 3; // 设置网格大小，3*3大小格
        int npts; // 选择的像素数目
        if (lvl == 0) // 第0层提取特征像素
            npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
        else
            npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);

        if (points[lvl] != nullptr)
            delete[] points[lvl];
        points[lvl] = new Pnt[npts];

        // set idepth map to initially 1 everywhere.
        int wl = w[lvl], hl = h[lvl];
        Pnt* pl = points[lvl];
        int nl = 0;
        //[ ***step 3*** ] 在选出的像素中, 添加点信息
        // 要留出pattern的空间, patternPadding = 2
        for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
            for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
                // if(x==2) printf("y=%d!\n",y);
                //  如果是被选中的像素
                if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0)) {
                    // assert(patternNum==9);
                    pl[nl].u = x + 0.1;
                    pl[nl].v = y + 0.1;
                    pl[nl].idepth = 1;
                    pl[nl].iR = 1;
                    pl[nl].isGood = true;
                    pl[nl].energy.setZero();
                    pl[nl].lastHessian = 0;
                    pl[nl].lastHessian_new = 0;
                    pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

                    // DONE: 这里求的 sumGrad2 是没有用?
                    // 看起来是原本用来定阈值的
                    // { // 论文的八点模板,把八点的梯度平方根加和:patternNum=8
                    //     Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y * w[lvl];
                    //     float sumGrad2 = 0;
                    //     for (int idx = 0; idx < patternNum; idx++) {
                    //         // 模板的相对位置dx,dy
                    //         int dx = patternP[idx][0];
                    //         int dy = patternP[idx][1];
                    //         float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
                    //         sumGrad2 += absgrad;
                    //     }

                    //     //				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
                    //     //				pl[nl].outlierTH = patternNum*gth*gth;
                    //     //
                    // }

                    //! 外点的阈值与pattern的大小有关, 一个像素是12*12
                    //? 这个阈值怎么确定的...
                    pl[nl].outlierTH = patternNum * setting_outlierTH;

                    nl++;
                    assert(nl <= npts);
                }
            }

        numPoints[lvl] = nl;
    }
    delete[] statusMap;
    delete[] statusMapB;

    //[ ***step 4*** ] 计算点的最近邻和父点
    makeNN();

    // 参数初始化
    thisToNext = SE3();
    snapped = false;
    frameID = snappedAt = 0;

    // DONE: 没用的变量
    // {
    //     for (int i = 0; i < pyrLevelsUsed; i++)
    //         dGrads[i].setZero();
    // }
}

//@ 重置点的energy, idepth_new参数
void CoarseInitializer::resetPoints(int lvl)
{
    Pnt* pts = points[lvl];
    int npts = numPoints[lvl];
    for (int i = 0; i < npts; i++) {
        pts[i].energy.setZero();
        pts[i].idepth_new = pts[i].idepth;

        // 如果是最顶层, 则使用周围点平均值来重置
        if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood) {
            float snd = 0, sn = 0;
            for (int neighbour : pts[i].neighbours) {
                if (neighbour == -1 || !pts[neighbour].isGood)
                    continue;
                snd += pts[neighbour].iR;
                sn += 1;
            }

            if (sn > 0) {
                pts[i].isGood = true;
                pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
            }
        }
    }
}
//* 求出状态增量后, 计算被边缘化掉的逆深度, 更新逆深度
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

    const float maxPixelStep = 0.25;
    const float idMaxStep = 1e10;
    Pnt* pts = points[lvl];
    int npts = numPoints[lvl];
    for (int i = 0; i < npts; i++) {
        if (!pts[i].isGood)
            continue;

        // 舒而补计算逆深度增量step
        //! dd*r + (dp*dd)^T*delta_p
        float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
        //! dd * delta_d = dd*r - (dp*dd)^T*delta_p = b
        //! delta_d = b * dd^-1
        float step = -b * JbBuffer[i][9] / (1 + lambda);

        float maxstep = maxPixelStep * pts[i].maxstep; // 逆深度最大只能增加这些
        if (maxstep > idMaxStep)
            maxstep = idMaxStep;

        if (step > maxstep)
            step = maxstep;
        if (step < -maxstep)
            step = -maxstep;

        // 更新得到新的逆深度
        float newIdepth = pts[i].idepth + step;
        if (newIdepth < 1e-3)
            newIdepth = 1e-3;
        if (newIdepth > 50)
            newIdepth = 50;
        pts[i].idepth_new = newIdepth;
    }
}

//* 新的值赋值给旧的 (能量, 点状态, 逆深度, hessian)
void CoarseInitializer::applyStep(int lvl)
{
    Pnt* pts = points[lvl];
    int npts = numPoints[lvl];
    for (int i = 0; i < npts; i++) {
        if (!pts[i].isGood) {
            pts[i].idepth = pts[i].idepth_new = pts[i].iR;
            continue;
        }
        pts[i].energy = pts[i].energy_new;
        pts[i].isGood = pts[i].isGood_new;
        pts[i].idepth = pts[i].idepth_new;
        pts[i].lastHessian = pts[i].lastHessian_new;
    }
    std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

// 读取和构建各层金字塔图像的内参
void CoarseInitializer::makeK(CalibHessian* HCalib)
{
    w[0] = wG[0];
    h[0] = hG[0];

    fx[0] = HCalib->fxl();
    fy[0] = HCalib->fyl();
    cx[0] = HCalib->cxl();
    cy[0] = HCalib->cyl();

    for (int level = 1; level < pyrLevelsUsed; ++level) {
        w[level] = w[0] >> level;
        h[level] = h[0] >> level;
        fx[level] = fx[level - 1] * 0.5;
        fy[level] = fy[level - 1] * 0.5;
        cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
        cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
    }

    for (int level = 0; level < pyrLevelsUsed; ++level) {
        K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
        Ki[level] = K[level].inverse();
        fxi[level] = Ki[level](0, 0);
        fyi[level] = Ki[level](1, 1);
        cxi[level] = Ki[level](0, 2);
        cyi[level] = Ki[level](1, 2);
    }
}

/**
 * @brief k(10)-NearestNeighbor
 * 构建kdtree, 得到每个点同层最近的nn=10个点(neighbours)，和上一层最近的点(parent)
 *
 */
void CoarseInitializer::makeNN()
{
    const float NNDistFactor = 0.05;

    // 第一个参数为distance, 第二个是datasetadaptor, 第三个是维数
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>,
        FLANNPointcloud, 2>
        KDTree;

    // build indices
    FLANNPointcloud pcs[PYR_LEVELS]; // 每层建立一个点云
    KDTree* indexes[PYR_LEVELS]; // 点云建立KDtree
    //* 每层建立一个KDTree索引二维点云
    for (int i = 0; i < pyrLevelsUsed; i++) {
        pcs[i] = FLANNPointcloud(numPoints[i], points[i]); // 二维点点云
        // 参数: 维度, 点数据, 叶节点中最大的点数(越大build快, query慢)
        indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
        indexes[i]->buildIndex();
    }

    const int nn = 10;

    // find NN & parents
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        Pnt* pts = points[lvl];
        int npts = numPoints[lvl];

        int ret_index[nn]; // 搜索到的临近点
        float ret_dist[nn]; // 搜索到点的距离
        nanoflann::KNNResultSet<float, int, int> resultSet(nn);
        nanoflann::KNNResultSet<float, int, int> resultSet1(1);

        for (int i = 0; i < npts; i++) {
            // resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
            resultSet.init(ret_index, ret_dist);
            Vec2f pt = Vec2f(pts[i].u, pts[i].v);
            // 使用建立的KDtree, 来查询最近邻
            indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
            int myidx = 0;
            float sumDF = 0;
            //* 给每个点的neighbours赋值
            for (int k = 0; k < nn; k++) {
                pts[i].neighbours[myidx] = ret_index[k]; // 最近的索引
                float df = expf(-ret_dist[k] * NNDistFactor); // 距离使用指数形式
                sumDF += df; // 距离和
                pts[i].neighboursDist[myidx] = df;
                assert(ret_index[k] >= 0 && ret_index[k] < npts);
                myidx++;
            }
            // 对距离进行归一化
            for (int k = 0; k < nn; k++)
                pts[i].neighboursDist[k] *= 10 / sumDF;

            //* 高一层的图像中找到该点的父节点
            if (lvl < pyrLevelsUsed - 1) {
                resultSet1.init(ret_index, ret_dist);
                pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
                indexes[lvl + 1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

                pts[i].parent = ret_index[0]; // 父节点
                pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor); // 到父节点的距离(在高层中)

                assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
            } else {
                pts[i].parent = -1;
                pts[i].parentDist = -1;
            }
        }
    }

    // done.

    for (int i = 0; i < pyrLevelsUsed; i++)
        delete indexes[i];
}
} // namespace dso
