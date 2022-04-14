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
#define MAX_ACTIVE_FRAMES 100

#include "util/globalCalib.h"
#include "vector"

#include "FullSystem/Residuals.h"
#include "util/ImageAndExposure.h"
#include "util/NumType.h"
#include <fstream>
#include <iostream>

namespace dso {

inline Vec2 affFromTo(const Vec2& from, const Vec2& to) // contains affine parameters as XtoWorld.
{
    return Vec2(from[0] / to[0], (from[1] - to[1]) / to[0]);
}

struct FrameHessian;
struct PointHessian;

class ImmaturePoint;
class FrameShell;

class EFFrame;
class EFPoint;

#define SCALE_IDEPTH 1.0f // scales internal value to idepth.
#define SCALE_XI_ROT 1.0f
// #define SCALE_XI_TRANS 0.5f
#define SCALE_XI_TRANS 1.0f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f

#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)
#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)
#define SCALE_A_INVERSE (1.0f / SCALE_A)
#define SCALE_B_INVERSE (1.0f / SCALE_B)

struct FrameFramePrecalc {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // static values
    static int instanceCounter;
    FrameHessian* host; // defines row
    FrameHessian* target; // defines column

    // precalc values
    Mat33f PRE_RTll;
    Mat33f PRE_KRKiTll;
    Mat33f PRE_RKiTll;
    Mat33f PRE_RTll_0;

    Vec2f PRE_aff_mode;
    float PRE_b0_mode;

    Vec3f PRE_tTll;
    Vec3f PRE_KtTll;
    Vec3f PRE_tTll_0;

    float distanceLL;

    inline ~FrameFramePrecalc() { }
    inline FrameFramePrecalc() { host = target = 0; }
    void set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib);
    void setDeep(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib);
};

/**
 * @brief 保存单帧信息，存在较多冗余信息，因此每一次都会主动delete
 * 图像信息，梯度
 * 相机位姿+相机光度Hessian
 *
 */
struct FrameHessian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EFFrame* efFrame; //!< 帧的能量函数

    // constant info & pre-calculated values
    // DepthImageWrap* frame;
    FrameShell* shell; //!< 帧的"壳", 保存一些不变的,要留下来的量

    //* 图像导数[0]:辐照度  [1]:x方向导数  [2]:y方向导数, （指针表示图像）
    Eigen::Vector3f *dI; // trace, fine tracking. Used for direction select (not for gradient histograms etc.)
    Eigen::Vector3f *dFeatureI; // trace, fine tracking. Used for direction select (not for gradient histograms etc.)
    Eigen::Vector3f* dIp[PYR_LEVELS]; //!< 各金字塔层的图像及导数 // coarse tracking / coarse initializer. NAN in [0] only.
    float* absSquaredGrad[PYR_LEVELS]; //!< 各层的x,y 方向梯度的平方和// only used for pixel select (histograms etc.). no NAN.

    bool addCamPrior;

    int frameID; //!< 所有关键帧的序号 // incremental ID for keyframes only!
    static int instanceCounter; //!< 计数器
    int idx; //!< 激活关键帧的序号(FrameHessian)

    // Photometric Calibration Stuff
    float frameEnergyTH; //!< 阈值// set dynamically depending on tracking residual
    float ab_exposure;

    bool flaggedForMarginalization;

    std::vector<PointHessian*> pointHessians; // contains all ACTIVE points.
    std::vector<PointHessian*> pointHessiansMarginalized; // contains all MARGINALIZED points (= fully marginalized, usually because point went OOB.)
    std::vector<PointHessian*> pointHessiansOut; // contains all OUTLIER points (= discarded.).
    std::vector<ImmaturePoint*> immaturePoints; // contains all OUTLIER points (= discarded.).

    Mat66 nullspaces_pose;
    Mat42 nullspaces_affine;
    Vec6 nullspaces_scale;

    // variable info.
    SE3 worldToCam_evalPT; //!< 在估计的相机位姿
                           // [0-5: 位姿左乘小量. 6-7: a,b 光度仿射系数]
                           //* 这三个是与线性化点的增量, 而光度参数不是增量, state就是值
    Vec10 state_zero; //!< 固定的线性化点的状态增量, 为了计算进行缩放
    Vec10 state_scaled; //!< 乘上比例系数的状态增量, 这个是真正求的值!!!
    Vec10 state; //!< 计算的状态增量// [0-5: worldToCam-leftEps. 6-7: a,b]
    //* step是与上一次优化结果的状态增量, [8 ,9]直接就设置为0了
    Vec10 step; //!< 求解正规方程得到的增量
    Vec10 step_backup; //!< 上一次的增量备份
    Vec10 state_backup; //!< 上一次状态的备份

    //内联提高效率, 返回上面的值
    EIGEN_STRONG_INLINE const SE3& get_worldToCam_evalPT() const { return worldToCam_evalPT; }
    EIGEN_STRONG_INLINE const Vec10& get_state_zero() const { return state_zero; } // the first 6 parameters of state_zero seem to be always 0 (as this part ist represented by the worldToCam_evalPT. The last two parameters on the other hand are not zero.
    EIGEN_STRONG_INLINE const Vec10& get_state() const { return state; }
    EIGEN_STRONG_INLINE const Vec10& get_state_scaled() const { return state_scaled; }
    EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const { return get_state() - get_state_zero(); } // x小量可以直接减

    // precalc values
    SE3 PRE_worldToCam; //!< 预计算的, 位姿状态增量更新到位姿上
    SE3 PRE_camToWorld;
    std::vector<FrameFramePrecalc, Eigen::aligned_allocator<FrameFramePrecalc>> targetPrecalc; //!< 对于其它帧的预运算值
    MinimalImageB3* debugImage;

    inline Vec6 w2c_leftEps() const { return get_state_scaled().head<6>(); }
    inline AffLight aff_g2l() const { return AffLight(get_state_scaled()[6], get_state_scaled()[7]); }
    inline AffLight aff_g2l_0() const { return AffLight(get_state_zero()[6] * SCALE_A, get_state_zero()[7] * SCALE_B); }

    //* 设置FEJ点状态增量
    void setStateZero(const Vec10& state_zero);
    //* 设置增量, 同时复制state和state_scale
    inline void setState(const Vec10& state)
    {

        this->state = state;
        state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
        state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
        state_scaled[6] = SCALE_A * state[6];
        state_scaled[7] = SCALE_B * state[7];
        state_scaled[8] = SCALE_A * state[8];
        state_scaled[9] = SCALE_B * state[9];

        //位姿更新
        PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
        PRE_camToWorld = PRE_worldToCam.inverse();
        // setCurrentNullspace();
    };
    inline void setStateScaled(const Vec10& state_scaled)
    {

        this->state_scaled = state_scaled;
        state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
        state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
        state[6] = SCALE_A_INVERSE * state_scaled[6];
        state[7] = SCALE_B_INVERSE * state_scaled[7];
        state[8] = SCALE_A_INVERSE * state_scaled[8];
        state[9] = SCALE_B_INVERSE * state_scaled[9];

        PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
        PRE_camToWorld = PRE_worldToCam.inverse();
        // setCurrentNullspace();
    };
    inline void setEvalPT(const SE3& worldToCam_evalPT, const Vec10& state)
    {

        this->worldToCam_evalPT = worldToCam_evalPT;
        setState(state);
        setStateZero(state);
    };

    inline void setEvalPT_scaled(const SE3& worldToCam_evalPT, const AffLight& aff_g2l)
    {
        Vec10 initial_state = Vec10::Zero();
        initial_state[6] = aff_g2l.a;
        initial_state[7] = aff_g2l.b;
        this->worldToCam_evalPT = worldToCam_evalPT;
        setStateScaled(initial_state);
        setStateZero(this->get_state());
    };

    void release();

    inline ~FrameHessian()
    {
        assert(efFrame == 0);
        release();
        instanceCounter--;
        for (int i = 0; i < pyrLevelsUsed; i++) {
            delete[] dIp[i];
            delete[] absSquaredGrad[i];
        }

        if (debugImage != 0)
            delete debugImage;
    };
    inline FrameHessian()
    {
        instanceCounter++;
        flaggedForMarginalization = false;
        frameID = -1;
        efFrame = 0;
        frameEnergyTH = 8 * 8 * patternNum;

        debugImage = 0;

        addCamPrior = false;
    };

    void makeImages(float* color, CalibHessian* HCalib);
    void makeDeepImages(float* color, float* feature, CalibHessian* HCalib);

    inline Vec10 getPrior()
    {
        Vec10 p = Vec10::Zero();
        if (frameID == 0) {
            p.head<3>() = Vec3::Constant(setting_initialTransPrior);
            p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
            if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
                p.head<6>().setZero();

            p[6] = setting_initialAffAPrior;
            p[7] = setting_initialAffBPrior;
        } else {
            if (setting_affineOptModeA < 0)
                p[6] = setting_initialAffAPrior;
            else
                p[6] = setting_affineOptModeA;

            if (setting_affineOptModeB < 0)
                p[7] = setting_initialAffBPrior;
            else
                p[7] = setting_affineOptModeB;
        }
        p[8] = setting_initialAffAPrior;
        p[9] = setting_initialAffBPrior;

        if (addCamPrior) {
            p.head<3>() = Vec3::Constant(setting_initialTransPrior);
            p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
            if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
                p.head<6>().setZero();
        }

        return p;
    }

    inline Vec10 getPriorZero()
    {
        return Vec10::Zero();
    }
};

struct CalibHessian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    static int instanceCounter;

    VecC value_zero;
    VecC value_scaled;
    VecCf value_scaledf;
    VecCf value_scaledi;
    VecC value;
    VecC step;
    VecC step_backup;
    VecC value_backup;
    VecC value_minus_value_zero;

    inline ~CalibHessian() { instanceCounter--; }
    inline CalibHessian()
    {

        VecC initial_value = VecC::Zero();
        initial_value[0] = fxG[0];
        initial_value[1] = fyG[0];
        initial_value[2] = cxG[0];
        initial_value[3] = cyG[0];

        setValueScaled(initial_value);
        value_zero = value;
        value_minus_value_zero.setZero();

        instanceCounter++;
        for (int i = 0; i < 256; i++)
            Binv[i] = B[i] = i; // set gamma function to identity
    };

    // normal mode: use the optimized parameters everywhere!
    inline float& fxl() { return value_scaledf[0]; }
    inline float& fyl() { return value_scaledf[1]; }
    inline float& cxl() { return value_scaledf[2]; }
    inline float& cyl() { return value_scaledf[3]; }
    inline float& fxli() { return value_scaledi[0]; }
    inline float& fyli() { return value_scaledi[1]; }
    inline float& cxli() { return value_scaledi[2]; }
    inline float& cyli() { return value_scaledi[3]; }

    inline void setValue(const VecC& value)
    {
        // [0-3: Kl, 4-7: Kr, 8-12: l2r]
        this->value = value;
        value_scaled[0] = SCALE_F * value[0];
        value_scaled[1] = SCALE_F * value[1];
        value_scaled[2] = SCALE_C * value[2];
        value_scaled[3] = SCALE_C * value[3];

        this->value_scaledf = this->value_scaled.cast<float>();
        this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
        this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
        this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
        this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];
        this->value_minus_value_zero = this->value - this->value_zero;
    };

    inline void setValueScaled(const VecC& value_scaled)
    {
        this->value_scaled = value_scaled;
        this->value_scaledf = this->value_scaled.cast<float>();
        value[0] = SCALE_F_INVERSE * value_scaled[0];
        value[1] = SCALE_F_INVERSE * value_scaled[1];
        value[2] = SCALE_C_INVERSE * value_scaled[2];
        value[3] = SCALE_C_INVERSE * value_scaled[3];

        this->value_minus_value_zero = this->value - this->value_zero;
        this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
        this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
        this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
        this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];
    };

    float Binv[256];
    float B[256];

    EIGEN_STRONG_INLINE float getBGradOnly(float color)
    {
        int c = color + 0.5f;
        if (c < 5)
            c = 5;
        if (c > 250)
            c = 250;
        return B[c + 1] - B[c];
    }

    EIGEN_STRONG_INLINE float getBInvGradOnly(float color)
    {
        int c = color + 0.5f;
        if (c < 5)
            c = 5;
        if (c > 250)
            c = 250;
        return Binv[c + 1] - Binv[c];
    }
};

// hessian component associated with one point.
struct PointHessian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    static int instanceCounter;
    EFPoint* efPoint;

    // static values
    float color[MAX_RES_PER_POINT]; // colors in host frame
    float weights[MAX_RES_PER_POINT]; // host-weights for respective residuals.

    float u, v;
    int idx;
    float energyTH;
    FrameHessian* host;
    bool hasDepthPrior;

    float my_type;

    float idepth_scaled;
    float idepth_zero_scaled;
    float idepth_zero;
    float idepth;
    float step;
    float step_backup;
    float idepth_backup;

    float nullspaces_scale;
    float idepth_hessian;
    float maxRelBaseline;
    int numGoodResiduals;

    enum PtStatus { ACTIVE = 0,
        INACTIVE,
        OUTLIER,
        OOB,
        MARGINALIZED };
    PtStatus status;

    inline void setPointStatus(PtStatus s) { status = s; }

    inline void setIdepth(float idepth)
    {
        this->idepth = idepth;
        this->idepth_scaled = SCALE_IDEPTH * idepth;
    }
    inline void setIdepthScaled(float idepth_scaled)
    {
        this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
        this->idepth_scaled = idepth_scaled;
    }
    inline void setIdepthZero(float idepth)
    {
        idepth_zero = idepth;
        idepth_zero_scaled = SCALE_IDEPTH * idepth;
        nullspaces_scale = -(idepth * 1.001 - idepth / 1.001) * 500;
    }

    std::vector<PointFrameResidual*> residuals; // only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
    std::pair<PointFrameResidual*, ResState> lastResiduals[2]; // contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).

    void release();
    PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib);
    inline ~PointHessian()
    {
        assert(efPoint == 0);
        release();
        instanceCounter--;
    }

    inline bool isOOB(const std::vector<FrameHessian*>& toKeep, const std::vector<FrameHessian*>& toMarg) const
    {

        int visInToMarg = 0;
        for (PointFrameResidual* r : residuals) {
            if (r->state_state != ResState::IN)
                continue;
            for (FrameHessian* k : toMarg)
                if (r->target == k)
                    visInToMarg++;
        }
        if ((int)residuals.size() >= setting_minGoodActiveResForMarg && numGoodResiduals > setting_minGoodResForMarg + 10 && (int)residuals.size() - visInToMarg < setting_minGoodActiveResForMarg)
            return true;

        if (lastResiduals[0].second == ResState::OOB)
            return true;
        if (residuals.size() < 2)
            return false;
        if (lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER)
            return true;
        return false;
    }

    inline bool isInlierNew()
    {
        return (int)residuals.size() >= setting_minGoodActiveResForMarg
            && numGoodResiduals >= setting_minGoodResForMarg;
    }
};

}
