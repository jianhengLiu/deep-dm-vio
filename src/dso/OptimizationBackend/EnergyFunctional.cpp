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

#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <util/TimeMeasurement.h>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

bool EFAdjointsValid = false; //!< 是否设置状态伴随矩阵
bool EFIndicesValid = false; //!< 是否设置frame, point, res的ID
bool EFDeltaValid = false; //!< 是否设置状态增量值

//@ 计算adHost(F), adTarget(F)
void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
{

    if (adHost != 0)
        delete[] adHost;
    if (adTarget != 0)
        delete[] adTarget;
    adHost = new Mat88[nFrames * nFrames];
    adTarget = new Mat88[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++) // 主帧host
        for (int t = 0; t < nFrames; t++) // 目标帧target
        {
            FrameHessian* host = frames[h]->data;
            FrameHessian* target = frames[t]->data;

            SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

            Mat88 AH = Mat88::Identity();
            Mat88 AT = Mat88::Identity();

            AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
            AT.topLeftCorner<6, 6>() = Mat66::Identity();

            Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
            AT(6, 6) = -affLL[0];
            AH(6, 6) = affLL[0];
            AT(7, 7) = -1;
            AH(7, 7) = affLL[0];

            AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
            AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
            AH.block<1, 8>(6, 0) *= SCALE_A;
            AH.block<1, 8>(7, 0) *= SCALE_B;
            AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
            AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
            AT.block<1, 8>(6, 0) *= SCALE_A;
            AT.block<1, 8>(7, 0) *= SCALE_B;

            adHost[h + t * nFrames] = AH;
            adTarget[h + t * nFrames] = AT;
        }
    cPrior = VecC::Constant(setting_initialCalibHessian);

    if (adHostF != 0)
        delete[] adHostF;
    if (adTargetF != 0)
        delete[] adTargetF;
    adHostF = new Mat88f[nFrames * nFrames];
    adTargetF = new Mat88f[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++)
        for (int t = 0; t < nFrames; t++) {
            adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
            adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
        }

    cPriorF = cPrior.cast<float>();

    EFAdjointsValid = true;
}

//@ 计算adHost(F), adTarget(F)
void EnergyFunctional::setDeepAdjointsF(CalibHessian* Hcalib)
{
    // if (adHost != 0)
    //     delete[] adHost;
    // if (adTarget != 0)
    //     delete[] adTarget;
    // adHost = new Mat88[nFrames * nFrames];
    // adTarget = new Mat88[nFrames * nFrames];
    if (adDeepHost != 0)
        delete[] adDeepHost;
    if (adDeepTarget != 0)
        delete[] adDeepTarget;
    adDeepHost = new Mat66[nFrames * nFrames];
    adDeepTarget = new Mat66[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++) // 主帧host
        for (int t = 0; t < nFrames; t++) // 目标帧target
        {
            FrameHessian* host = frames[h]->data;
            FrameHessian* target = frames[t]->data;

            SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

            // Mat88 AH = Mat88::Identity();
            // Mat88 AT = Mat88::Identity();
            Mat66 AH = Mat66::Identity();
            Mat66 AT = Mat66::Identity();

            // AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
            // AT.topLeftCorner<6, 6>() = Mat66::Identity();
            AH = -hostToTarget.Adj().transpose();
            AT = Mat66::Identity();

            // Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
            // AT(6, 6) = -affLL[0];
            // AH(6, 6) = affLL[0];
            // AT(7, 7) = -1;
            // AH(7, 7) = affLL[0];

            // AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
            // AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
            // AH.block<1, 8>(6, 0) *= SCALE_A;
            // AH.block<1, 8>(7, 0) *= SCALE_B;
            // AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
            // AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
            // AT.block<1, 8>(6, 0) *= SCALE_A;
            // AT.block<1, 8>(7, 0) *= SCALE_B;
            AH.block<3, 6>(0, 0) *= SCALE_XI_TRANS;
            AH.block<3, 6>(3, 0) *= SCALE_XI_ROT;
            AT.block<3, 6>(0, 0) *= SCALE_XI_TRANS;
            AT.block<3, 6>(3, 0) *= SCALE_XI_ROT;

            // adHost[h + t * nFrames] = AH;
            // adTarget[h + t * nFrames] = AT;
            adDeepHost[h + t * nFrames] = AH;
            adDeepTarget[h + t * nFrames] = AT;
        }
    cPrior = VecC::Constant(setting_initialCalibHessian); // 常数矩阵

    // if (adHostF != 0)
    //     delete[] adHostF;
    // if (adTargetF != 0)
    //     delete[] adTargetF;
    // adHostF = new Mat88f[nFrames * nFrames];
    // adTargetF = new Mat88f[nFrames * nFrames];
    if (adDeepHostF != 0)
        delete[] adDeepHostF;
    if (adDeepTargetF != 0)
        delete[] adDeepTargetF;
    adDeepHostF = new Mat66f[nFrames * nFrames];
    adDeepTargetF = new Mat66f[nFrames * nFrames];

    // for (int h = 0; h < nFrames; h++)
    //     for (int t = 0; t < nFrames; t++) {
    //         adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
    //         adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
    //     }
    for (int h = 0; h < nFrames; h++)
        for (int t = 0; t < nFrames; t++) {
            adDeepHostF[h + t * nFrames] = adDeepHost[h + t * nFrames].cast<float>();
            adDeepTargetF[h + t * nFrames] = adDeepTarget[h + t * nFrames].cast<float>();
        }

    cPriorF = cPrior.cast<float>();

    EFAdjointsValid = true;
}

EnergyFunctional::EnergyFunctional(dmvio::BAGTSAMIntegration& gtsamIntegration)
    : gtsamIntegration(gtsamIntegration)
{
    adHost = 0;
    adTarget = 0;

    red = 0;

    adHostF = 0;
    adTargetF = 0;
    adHTdeltaF = 0;

    nFrames = nResiduals = nPoints = 0;

    HM = MatXX::Zero(CPARS, CPARS);
    HMForGTSAM = MatXX::Zero(CPARS, CPARS);
    bM = VecX::Zero(CPARS);
    bMForGTSAM = VecX::Zero(CPARS);

    accSSE_top_L = new AccumulatedTopHessianSSE();
    accSSE_top_A = new AccumulatedTopHessianSSE();
    accSSE_bot = new AccumulatedSCHessianSSE();

    resInA = resInL = resInM = 0;
    currentLambda = 0;
}
EnergyFunctional::~EnergyFunctional()
{
    for (EFFrame* f : frames) {
        for (EFPoint* p : f->points) {
            for (EFResidual* r : p->residualsAll) {
                r->data->efResidual = 0;
                delete r;
            }
            p->data->efPoint = 0;
            delete p;
        }
        f->data->efFrame = 0;
        delete f;
    }

    if (adHost != 0)
        delete[] adHost;
    if (adTarget != 0)
        delete[] adTarget;

    if (adHostF != 0)
        delete[] adHostF;
    if (adTargetF != 0)
        delete[] adTargetF;
    if (adHTdeltaF != 0)
        delete[] adHTdeltaF;

    delete accSSE_top_L;
    delete accSSE_top_A;
    delete accSSE_bot;
}

//@ 计算各种状态的相对量的增量
void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
{
    if (adHTdeltaF != 0)
        delete[] adHTdeltaF;
    adHTdeltaF = new Mat18f[nFrames * nFrames];
    for (int h = 0; h < nFrames; h++)
        for (int t = 0; t < nFrames; t++) {
            int idx = h + t * nFrames;
            //! delta_th = Adj * delta_t or delta_th = Adj * delta_h
            // 加一起应该是, 两帧之间位姿变换的增量, 因为h变一点, t变一点
            adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
                + frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
        }

    cDeltaF = HCalib->value_minus_value_zero.cast<float>(); // 相机内参增量
    for (EFFrame* f : frames) {
        f->delta = f->data->get_state_minus_stateZero().head<8>(); // 帧位姿增量
        f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>(); // 先验增量

        for (EFPoint* p : f->points)
            p->deltaF = p->data->idepth - p->data->idepth_zero; // 逆深度的增量
    }

    EFDeltaValid = true;
}

//@ 计算各种状态的相对量的增量
void EnergyFunctional::setDeepDeltaF(CalibHessian* HCalib)
{
    if (adHTdeltaF != 0)
        delete[] adHTdeltaF;
    adHTdeltaF = new Mat18f[nFrames * nFrames];
    for (int h = 0; h < nFrames; h++)
        for (int t = 0; t < nFrames; t++) {
            int idx = h + t * nFrames;
            //! delta_th = Adj * delta_t or delta_th = Adj * delta_h
            // 加一起应该是, 两帧之间位姿变换的增量, 因为h变一点, t变一点
            adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
                + frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
        }

    cDeltaF = HCalib->value_minus_value_zero.cast<float>(); // 相机内参增量
    for (EFFrame* f : frames) {
        f->delta = f->data->get_state_minus_stateZero().head<8>(); // 帧位姿增量
        f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>(); // 先验增量

        for (EFPoint* p : f->points)
            p->deltaF = p->data->idepth - p->data->idepth_zero; // 逆深度的增量
    }

    EFDeltaValid = true;
}

// accumulates & shifts L.
void EnergyFunctional::accumulateAF_MT(MatXX& H, VecX& b, bool MT)
{
    if (MT) {
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames, _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
                        accSSE_top_A, &allPoints, this, _1, _2, _3, _4),
            0, allPoints.size(), 50);
        accSSE_top_A->stitchDoubleMT(red, H, b, this, false, true);
        resInA = accSSE_top_A->nres[0];
    } else {
        accSSE_top_A->setZero(nFrames);
        for (EFFrame* f : frames)
            for (EFPoint* p : f->points)
                accSSE_top_A->addPoint<0>(p, this);
        accSSE_top_A->stitchDoubleMT(red, H, b, this, false, false);
        resInA = accSSE_top_A->nres[0];
    }
}

// accumulates & shifts L.
void EnergyFunctional::accumulateLF_MT(MatXX& H, VecX& b, bool MT)
{
    if (MT) {
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames, _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
                        accSSE_top_L, &allPoints, this, _1, _2, _3, _4),
            0, allPoints.size(), 50);
        accSSE_top_L->stitchDoubleMT(red, H, b, this, true, true);
        resInL = accSSE_top_L->nres[0];
    } else {
        accSSE_top_L->setZero(nFrames);
        for (EFFrame* f : frames)
            for (EFPoint* p : f->points)
                accSSE_top_L->addPoint<1>(p, this);
        accSSE_top_L->stitchDoubleMT(red, H, b, this, true, false);
        resInL = accSSE_top_L->nres[0];
    }
}

void EnergyFunctional::accumulateSCF_MT(MatXX& H, VecX& b, bool MT)
{
    if (MT) {
        red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames, _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
                        accSSE_bot, &allPoints, true, _1, _2, _3, _4),
            0, allPoints.size(), 50);
        accSSE_bot->stitchDoubleMT(red, H, b, this, true);
    } else {
        accSSE_bot->setZero(nFrames);
        for (EFFrame* f : frames)
            for (EFPoint* p : f->points)
                accSSE_bot->addPoint(p, true);
        accSSE_bot->stitchDoubleMT(red, H, b, this, false);
    }
}

//@ 计算相机内参和位姿, 光度的增量
void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
{
    assert(x.size() == CPARS + nFrames * 8);

    VecXf xF = x.cast<float>();
    HCalib->step = -x.head<CPARS>(); // 相机内参, 这次的增量

    Mat18f* xAd = new Mat18f[nFrames * nFrames];
    VecCf cstep = xF.head<CPARS>();
    for (EFFrame* h : frames) {
        h->data->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idx); // 帧位姿和光度求解的增量
        h->data->step.tail<2>().setZero();

        //* 绝对位姿增量变相对的
        for (EFFrame* t : frames)
            xAd[nFrames * h->idx + t->idx] = xF.segment<8>(CPARS + 8 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx]
                + xF.segment<8>(CPARS + 8 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
    }

    //* 计算点的逆深度增量
    if (MT)
        red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt,
                        this, cstep, xAd, _1, _2, _3, _4),
            0, allPoints.size(), 50);
    else
        resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

    delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(
    const VecCf& xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
{
    for (int k = min; k < max; k++) {
        EFPoint* p = allPoints[k];

        int ngoodres = 0;
        for (EFResidual* r : p->residualsAll)
            if (r->isActive())
                ngoodres++;
        if (ngoodres == 0) {
            p->data->step = 0;
            continue;
        }
        float b = p->bdSumF;
        b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

        for (EFResidual* r : p->residualsAll) {
            if (!r->isActive())
                continue;
            b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
        }

        p->data->step = -b * p->HdiF;
        assert(std::isfinite(p->data->step));
    }
}

double EnergyFunctional::calcMEnergyF(bool useNewValues)
{

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    VecX delta = getStitchedDeltaF();

    double firstVal = delta.dot(2 * bM + HM * delta);

    if (setting_useGTSAMIntegration) {
        if (!useNewValues) {
            gtsamIntegration.updateBAValues(frames);
        }
        double secondVal = gtsamIntegration.getBAEnergy(useNewValues) + delta.dot(2 * bMForGTSAM + HMForGTSAM * delta);
        return secondVal;
    }

    return firstVal;
}

void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
{

    Accumulator11 E;
    E.initialize();
    VecCf dc = cDeltaF;

    for (int i = min; i < max; i++) {
        EFPoint* p = allPoints[i];
        float dd = p->deltaF;

        for (EFResidual* r : p->residualsAll) {
            if (!r->isLinearized || !r->isActive())
                continue;

            Mat18f dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
            RawResidualJacobian* rJ = r->J;

            // compute Jp*delta
            float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>())
                + rJ->Jpdc[0].dot(dc)
                + rJ->Jpdd[0] * dd;

            float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>())
                + rJ->Jpdc[1].dot(dc)
                + rJ->Jpdd[1] * dd;

            __m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
            __m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
            __m128 delta_a = _mm_set1_ps((float)(dp[6]));
            __m128 delta_b = _mm_set1_ps((float)(dp[7]));

            for (int i = 0; i + 3 < patternNum; i += 4) {
                // PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
                __m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx)) + i), Jp_delta_x);
                Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx + 1)) + i), Jp_delta_y));
                Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF)) + i), delta_a));
                Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF + 1)) + i), delta_b));

                __m128 r0 = _mm_load_ps(((float*)&r->res_toZeroF) + i);
                r0 = _mm_add_ps(r0, r0);
                r0 = _mm_add_ps(r0, Jdelta);
                Jdelta = _mm_mul_ps(Jdelta, r0);
                E.updateSSENoShift(Jdelta);
            }
            for (int i = ((patternNum >> 2) << 2); i < patternNum; i++) {
                float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 + rJ->JIdx[1][i] * Jp_delta_y_1 + rJ->JabF[0][i] * dp[6] + rJ->JabF[1][i] * dp[7];
                E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
            }
        }
        E.updateSingle(p->deltaF * p->deltaF * p->priorF);
    }
    E.finish();
    (*stats)[0] += E.A;
}

double EnergyFunctional::calcLEnergyF_MT()
{
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    double E = 0;
    for (EFFrame* f : frames) {
        E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);
    }
    E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

    red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt,
                    this, _1, _2, _3, _4),
        0, allPoints.size(), 50);

    return E + red->stats[0];
}

EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
{
    EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
    efr->idxInAll = r->point->efPoint->residualsAll.size();
    r->point->efPoint->residualsAll.push_back(efr);

    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

    nResiduals++;
    r->efResidual = efr;
    return efr;
}
//@ 向能量函数中增加一帧, 进行的操作: 改变正规方程, 重新排ID, 共视关系
EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
{
    // 建立优化用的能量函数帧. 并加进能量函数frames中
    EFFrame* eff = new EFFrame(fh);
    eff->idx = frames.size();
    frames.push_back(eff);

    nFrames++;
    fh->efFrame = eff; // FrameHessian 指向能量函数帧

    assert(HM.cols() == 8 * nFrames + CPARS - 8); // 边缘化掉一帧, 缺8个
    // 一个帧8个参数 + 相机内参
    bM.conservativeResize(8 * nFrames + CPARS);
    bMForGTSAM.conservativeResize(8 * nFrames + CPARS);
    HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
    HMForGTSAM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
    bM.tail<8>().setZero();
    bMForGTSAM.tail<8>().setZero();
    HM.rightCols<8>().setZero();
    HM.bottomRows<8>().setZero();
    HMForGTSAM.rightCols<8>().setZero();
    HMForGTSAM.bottomRows<8>().setZero();

    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    setAdjointsF(Hcalib); // 设置伴随矩阵
    makeIDX(); // 设置ID

    for (EFFrame* fh2 : frames) {
        // 前32位是host帧的历史ID, 后32位是Target的历史ID
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0, 0);
        if (fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0, 0);
    }

    return eff;
}

//@ 向能量函数中增加一帧, 进行的操作: 改变正规方程, 重新排ID, 共视关系
EFFrame* EnergyFunctional::insertDeepFrame(FrameHessian* fh, CalibHessian* Hcalib)
{
    // 建立优化用的能量函数帧. 并加进能量函数frames中
    EFFrame* eff = new EFFrame(fh);
    eff->idx = frames.size();
    frames.push_back(eff);

    nFrames++;
    fh->efFrame = eff; // FrameHessian 指向能量函数帧

    // assert(HM.cols() == 8 * nFrames + CPARS - 8); // 边缘化掉一帧, 缺8个
    // // 一个帧8个参数 + 相机内参
    // bM.conservativeResize(8 * nFrames + CPARS);
    // bMForGTSAM.conservativeResize(8 * nFrames + CPARS);
    // HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
    // HMForGTSAM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
    // bM.tail<8>().setZero();
    // bMForGTSAM.tail<8>().setZero();
    // HM.rightCols<8>().setZero();
    // HM.bottomRows<8>().setZero();
    // HMForGTSAM.rightCols<8>().setZero();
    // HMForGTSAM.bottomRows<8>().setZero();
    assert(HM.cols() == 6 * nFrames + CPARS - 6); // 边缘化掉一帧, 缺8个
    // 一个帧6个参数 + 相机内参
    bM.conservativeResize(6 * nFrames + CPARS);
    bMForGTSAM.conservativeResize(6 * nFrames + CPARS);
    HM.conservativeResize(6 * nFrames + CPARS, 6 * nFrames + CPARS);
    HMForGTSAM.conservativeResize(6 * nFrames + CPARS, 6 * nFrames + CPARS);
    bM.tail<6>().setZero();
    bMForGTSAM.tail<6>().setZero();
    HM.rightCols<6>().setZero();
    HM.bottomRows<6>().setZero();
    HMForGTSAM.rightCols<6>().setZero();
    HMForGTSAM.bottomRows<6>().setZero();

    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    // setAdjointsF(Hcalib); // 设置伴随矩阵
    setDeepAdjointsF(Hcalib); // 设置伴随矩阵
    makeIDX(); // 设置ID

    for (EFFrame* fh2 : frames) {
        // 前32位是host帧的历史ID, 后32位是Target的历史ID
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0, 0);
        if (fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0, 0);
    }

    return eff;
}

EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
{
    EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
    efp->idxInPoints = ph->host->efFrame->points.size();
    ph->host->efFrame->points.push_back(efp);

    nPoints++;
    ph->efPoint = efp;

    EFIndicesValid = false;

    return efp;
}

void EnergyFunctional::dropResidual(EFResidual* r)
{
    EFPoint* p = r->point;
    assert(r == p->residualsAll[r->idxInAll]);

    p->residualsAll[r->idxInAll] = p->residualsAll.back();
    p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
    p->residualsAll.pop_back();

    if (r->isActive())
        r->host->data->shell->statistics_goodResOnThis++;
    else
        r->host->data->shell->statistics_outlierResOnThis++;

    connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
    nResiduals--;
    r->data->efResidual = 0;
    delete r;
}

void EnergyFunctional::marginalizeFrame(EFFrame* fh)
{
    dmvio::TimeMeasurement timeMeasurement("EF-marginalizeFrame");
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    assert((int)fh->points.size() == 0);
    int ndim = nFrames * 8 + CPARS - 8; // new dimension
    int odim = nFrames * 8 + CPARS; // old dimension

    //	VecX eigenvaluesPre = HM.eigenvalues().real();
    //	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
    //

    if (setting_useGTSAMIntegration) {
        // When adding additional factors with GTSAM they need to be accounted for during keyframe marginalization.
        // Hence we move the whole keyframe marginalization to the GTSAMIntegration.
        dmvio::TimeMeasurement innerMeas("MainMarginalization");
        assert(odim == (int)HM.rows());
        assert(odim == (int)HM.cols());
        assert(odim == (int)bM.size());

        // Adds H and b from the last points to the graph. Needs the current evaluation point for each frames.
        gtsamIntegration.addMarginalizedPointsBA(HMForGTSAM, bMForGTSAM, frames);

        Vec8 priorH;
        Vec8 priorB;
        priorH = fh->prior;
        priorB = fh->prior.cwiseProduct(fh->delta_prior);

        gtsamIntegration.addPriorBA(fh, priorH, priorB);

        // Marginalizes out the frame. Adds the symbols of this frame and then calls marginalize out.
        gtsamIntegration.marginalizeBAFrame(fh);

        HMForGTSAM.resize(ndim, ndim);
        bMForGTSAM.resize(ndim);

        HMForGTSAM.setZero();
        bMForGTSAM.setZero();
    }

    //    if(!setting_useGTSAMIntegration) // enable to remove the redundant visual only marginalization.
    if (true) {
        dmvio::TimeMeasurement measVis("VisualMarginalization");
        if ((int)fh->idx != (int)frames.size() - 1) {
            int io = fh->idx * 8 + CPARS; // index of frame to move to end
            int ntail = 8 * (nFrames - fh->idx - 1);
            assert((io + 8 + ntail) == nFrames * 8 + CPARS);

            Vec8 bTmp = bM.segment<8>(io);
            VecX tailTMP = bM.tail(ntail);
            bM.segment(io, ntail) = tailTMP;
            bM.tail<8>() = bTmp;

            MatXX HtmpCol = HM.block(0, io, odim, 8);
            MatXX rightColsTmp = HM.rightCols(ntail);
            HM.block(0, io, odim, ntail) = rightColsTmp;
            HM.rightCols(8) = HtmpCol;

            MatXX HtmpRow = HM.block(io, 0, 8, odim);
            MatXX botRowsTmp = HM.bottomRows(ntail);
            HM.block(io, 0, ntail, odim) = botRowsTmp;
            HM.bottomRows(8) = HtmpRow;
        }

        HM.bottomRightCorner<8, 8>().diagonal() += fh->prior;
        bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);

        //	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";

        VecX SVec = (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
        VecX SVecI = SVec.cwiseInverse();

        //	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() << "\n\n";
        //	std::cout << std::setprecision(16) << "SVecI: " << SVecI.transpose() << "\n\n";

        // scale!
        MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
        VecX bMScaled = SVecI.asDiagonal() * bM;

        // invert bottom part!
        Mat88 hpi = HMScaled.bottomRightCorner<8, 8>();
        hpi = 0.5f * (hpi + hpi);
        hpi = hpi.inverse();
        hpi = 0.5f * (hpi + hpi);

        // schur-complement!
        MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi;
        HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8, ndim);
        bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

        // unscale!
        HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
        bMScaled = SVec.asDiagonal() * bMScaled;

        // set.
        HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
        bM = bMScaled.head(ndim);

        // With the imu-integration this cannot be used, because there are other variables that have to be considered.
    } else {
        // Just update the dimensions without actually marginalizing, as these are not used in practice.
        HM = HM.topLeftCorner(ndim, ndim);
        bM = bM.head(ndim);
    }

    // remove from vector, without changing the order!
    for (unsigned int i = fh->idx; i + 1 < frames.size(); i++) {
        frames[i] = frames[i + 1];
        frames[i]->idx = i;
    }
    frames.pop_back();
    nFrames--;
    fh->data->efFrame = 0;

    assert((int)frames.size() * 8 + CPARS == (int)HM.rows());
    assert((int)frames.size() * 8 + CPARS == (int)HM.cols());
    assert((int)frames.size() * 8 + CPARS == (int)bM.size());
    assert((int)frames.size() == (int)nFrames);

    //	VecX eigenvaluesPost = HM.eigenvalues().real();
    //	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

    //	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

    //	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
    //	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    makeIDX();
}

//@ 边缘化掉一个点
void EnergyFunctional::marginalizePointsF()
{
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    //[ ***step 1*** ] 记录被边缘化的点
    allPointsToMarg.clear();
    for (EFFrame* f : frames) {
        for (auto p : f->points) {
            if (p->stateFlag == EFPointStatus::PS_MARGINALIZE) {
                p->priorF *= setting_idepthFixPriorMargFac;
                for (EFResidual* r : p->residualsAll)
                    if (r->isActive()) // 边缘化残差计数
                        connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
                allPointsToMarg.push_back(p);
            }
        }
    }

    //[ ***step 2*** ] 计算该点相连的残差构成的H, b, HSC, bSC
    accSSE_bot->setZero(nFrames);
    accSSE_top_A->setZero(nFrames);
    for (EFPoint* p : allPointsToMarg) {
        accSSE_top_A->addPoint<2>(p, this); // 这个点的残差, 计算 H b
        accSSE_bot->addPoint(p, false); // 边缘化部分
        removePoint(p);
    }
    MatXX M, Msc;
    VecX Mb, Mbsc;
    accSSE_top_A->stitchDouble(M, Mb, this, false, false); // 不加先验, 在后面加了
    accSSE_bot->stitchDouble(Msc, Mbsc, this);

    resInM += accSSE_top_A->nres[0];

    MatXX H = M - Msc;
    VecX b = Mb - Mbsc;

    //[ ***step 3*** ] 处理零空间
    // 减去零空间部分
    if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
        // have a look if prior is there.
        bool haveFirstFrame = false;
        for (EFFrame* f : frames)
            if (f->frameID == 0)
                haveFirstFrame = true;

        if (!haveFirstFrame)
            orthogonalize(&b, &H);
    }

    //! 给边缘化的量加了个权重，不准确的线性化
    HM += setting_margWeightFac * H; //* 所以边缘化的部分直接加在HM bM了
    bM += setting_margWeightFac * b;

    HMForGTSAM += setting_margWeightFac * H;
    bMForGTSAM += setting_margWeightFac * b;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
        orthogonalize(&bM, &HM);

    EFIndicesValid = false;
    makeIDX(); // 梳理ID
}

//@ 直接丢掉点, 不边缘化
void EnergyFunctional::dropPointsF()
{

    for (EFFrame* f : frames) {
        for (int i = 0; i < (int)f->points.size(); i++) {
            EFPoint* p = f->points[i];
            if (p->stateFlag == EFPointStatus::PS_DROP) {
                removePoint(p);
                i--;
            }
        }
    }

    EFIndicesValid = false;
    makeIDX();
}

void EnergyFunctional::removePoint(EFPoint* p)
{
    for (EFResidual* r : p->residualsAll)
        dropResidual(r);

    EFFrame* h = p->host;
    h->points[p->idxInPoints] = h->points.back();
    h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
    h->points.pop_back();

    nPoints--;
    p->data->efPoint = 0;

    EFIndicesValid = false;

    delete p;
}

//@ 计算零空间矩阵伪逆, 从 H 和 b 中减去零空间, 相当于设相应的Jacob为0
// https://blog.csdn.net/wubaobao1993/article/details/105106301/
void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
{
    //	VecX eigenvaluesPre = H.eigenvalues().real();
    //	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
    //	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";

    // decide to which nullspaces to orthogonalize.
    std::vector<VecX> ns;
    ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
    ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
    //	if(setting_affineOptModeA <= 0)
    //		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
    //	if(setting_affineOptModeB <= 0)
    //		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());

    // make Nullspaces matrix
    //! 7自由度不可观
    MatXX N(ns[0].rows(), ns.size());
    for (unsigned int i = 0; i < ns.size(); i++)
        N.col(i) = ns[i].normalized();

    //* 求伪逆
    // compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
    Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

    VecX SNN = svdNN.singularValues();
    double minSv = 1e10, maxSv = 0;
    for (int i = 0; i < SNN.size(); i++) {
        if (SNN[i] < minSv)
            minSv = SNN[i];
        if (SNN[i] > maxSv)
            maxSv = SNN[i];
    }
    // 比最大奇异值小setting_solverModeDelta(e-5)倍, 则认为是0
    for (int i = 0; i < SNN.size(); i++) {
        if (SNN[i] > setting_solverModeDelta * maxSv)
            SNN[i] = 1.0 / SNN[i];
        else
            SNN[i] = 0;
    } // 求逆

    MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); // [dim] x 9.
    //! Npi.transpose()是N的伪逆
    MatXX NNpiT = N * Npi.transpose(); // [dim] x [dim].
    MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose()); // = N * (N' * N)^-1 * N'.

    if (b != 0)
        *b -= NNpiTS * *b;
    if (H != 0)
        *H -= NNpiTS * *H * NNpiTS;

    //	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

    //	VecX eigenvaluesPost = H.eigenvalues().real();
    //	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
    //	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";
}

//@ 计算正规方程, 并求解
void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
{
    if (setting_solverMode & SOLVER_USE_GN) // 不同的位控制不同的模式
        lambda = 0;
    if (setting_solverMode & SOLVER_FIX_LAMBDA)
        lambda = 1e-5;

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    //[ ***step 1*** ] 先计算正规方程, 涉及边缘化, 先验, 舒尔补等
    MatXX HL_top, HA_top, H_sc;
    VecX bL_top, bA_top, bM_top, b_sc;

    //* 针对新的残差, 使用的当前残差, 没有逆深度的部分
    accumulateAF_MT(HA_top, bA_top, multiThreading);

    //* 边缘化fix的残差, 有边缘化对的, 使用的res_toZeroF减去线性化部分, 加上先验, 没有逆深度的部分
    // bug: 这里根本就没有点参与了, 只有先验信息, 因为边缘化的和删除的点都不在了
    //! 这里唯一的作用就是 把 p相关的置零
    accumulateLF_MT(HL_top, bL_top, multiThreading);

    //* 关于逆深度的Schur部分
    accumulateSCF_MT(H_sc, b_sc, multiThreading);

    //* 由于固定线性化点, 每次迭代更新残差
    bM_top = (bM + HM * getStitchedDeltaF());
    VecX bMGTSAM_top = (bMForGTSAM + HMForGTSAM * getStitchedDeltaF());

    MatXX HFinal_top;
    VecX bFinal_top;
    //[ ***step 2*** ] 如果是设置求解正交系统, 则把相对应的零空间部分Jacobian设置为0, 否则正常计算schur
    if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) {
        // have a look if prior is there.
        bool haveFirstFrame = false;
        for (EFFrame* f : frames)
            if (f->frameID == 0)
                haveFirstFrame = true;

        MatXX HT_act = HL_top + HA_top - H_sc;
        VecX bT_act = bL_top + bA_top - b_sc;

        //! 包含第一帧则不减去零空间
        //! 不包含第一帧, 因为要固定第一帧, 和第一帧统一, 减去零空间, 防止在零空间乱飘
        if (!haveFirstFrame)
            orthogonalize(&bT_act, &HT_act);

        HFinal_top = HT_act + HM;
        bFinal_top = bT_act + bM_top;

        lastHS = HFinal_top;
        lastbS = bFinal_top;

        // LM
        //* 这个阻尼也是加在Schur complement计算之后的
        for (int i = 0; i < 8 * nFrames + CPARS; i++)
            HFinal_top(i, i) *= (1 + lambda);

    } else {

        HFinal_top = HL_top + HM + HA_top;
        bFinal_top = bL_top + bM_top + bA_top - b_sc;

        lastHS = HFinal_top - H_sc;
        lastbS = bFinal_top;

        //* 而这个就是阻尼加在了整个Hessian上
        //? 为什么呢, 是因为减去了零空间么  ??
        for (int i = 0; i < 8 * nFrames + CPARS; i++)
            HFinal_top(i, i) *= (1 + lambda);
        HFinal_top -= H_sc * (1.0f / (1 + lambda)); // 因为Schur里面有个对角线的逆, 所以是倒数
    }

    //[ ***step 3*** ] 使用SVD求解, 或者ldlt直接求解
    VecX x;
    if (setting_solverMode & SOLVER_SVD) {
        //* 为数值稳定进行缩放
        VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
        MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
        VecX bFinalScaled = SVecI.asDiagonal() * bFinal_top;
        //! Hx=b --->  U∑V^T*x = b
        Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

        VecX S = svd.singularValues(); // 奇异值
        double minSv = 1e10, maxSv = 0;
        for (int i = 0; i < S.size(); i++) {
            if (S[i] < minSv)
                minSv = S[i];
            if (S[i] > maxSv)
                maxSv = S[i];
        }

        //! Hx=b --->  U∑V^T*x = b  --->  ∑V^T*x = U^T*b
        VecX Ub = svd.matrixU().transpose() * bFinalScaled;
        int setZero = 0;
        for (int i = 0; i < Ub.size(); i++) {
            if (S[i] < setting_solverModeDelta * maxSv) //* 奇异值小的设置为0
            {
                Ub[i] = 0;
                setZero++;
            }

            if ((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size() - 7)) //* 留出7个不可观的, 零空间
            {
                Ub[i] = 0;
                setZero++;
            }
            //! V^T*x = ∑^-1*U^T*b
            else
                Ub[i] /= S[i];
        }
        //! x = V*∑^-1*U^T*b   把scaled的乘回来
        x = SVecI.asDiagonal() * svd.matrixV() * Ub;

    } else {
        VecX myX;
        if (setting_useGTSAMIntegration) {
            // Instead of directly solving the system we instead pass it to the GTSAMIntegration which will add more
            // factors and then solve it for us. This is mathematically correct as long as the new residuals are
            // independent of the DSO residuals (which usually they are) and as long as they don't depend on the
            // points (as otherwise the Schur-complement trick doesn't work like this anymore).
            MatXX HPassed = HL_top + HMForGTSAM + HA_top;
            for (int i = 0; i < 8 * nFrames + CPARS; i++)
                HPassed(i, i) *= (1 + lambda);
            HPassed -= H_sc * (1.0f / (1 + lambda));
            x = gtsamIntegration.computeBAUpdate(HPassed, bL_top + bMGTSAM_top + bA_top - b_sc, lambda,
                frames, HL_top + HMForGTSAM + HA_top - H_sc);
        } else {
            VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
            MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
            x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);
        }
        // Important: x is -step !
    }

    //[ ***step 4*** ] 如果设置的是直接对解进行处理, 直接去掉解x中的零空间
    if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
        VecX xOld = x;
        orthogonalize(&x, 0);
    }

    lastX = x;

    //[ ***step 5*** ] 分别求出各个待求量的增量值
    // resubstituteF(x, HCalib);
    currentLambda = lambda;
    resubstituteF_MT(x, HCalib, multiThreading);
    currentLambda = 0;
}
void EnergyFunctional::makeIDX()
{
    for (unsigned int idx = 0; idx < frames.size(); idx++)
        frames[idx]->idx = idx;

    allPoints.clear();

    for (EFFrame* f : frames)
        for (EFPoint* p : f->points) {
            allPoints.push_back(p);
            for (EFResidual* r : p->residualsAll) {
                r->hostIDX = r->host->idx;
                r->targetIDX = r->target->idx;
            }
        }

    EFIndicesValid = true;
}

VecX EnergyFunctional::getStitchedDeltaF() const
{
    VecX d = VecX(CPARS + nFrames * 8);
    d.head<CPARS>() = cDeltaF.cast<double>();
    for (int h = 0; h < nFrames; h++)
        d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
    return d;
}

}
