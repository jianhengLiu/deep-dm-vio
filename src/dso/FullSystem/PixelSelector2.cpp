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

#include "FullSystem/PixelSelector2.h"

//

#include "FullSystem/HessianBlocks.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

namespace dso {

PixelSelector::PixelSelector(int w, int h)
{
    randomPattern = new unsigned char[w * h];
    std::srand(3141592); // want to be deterministic.
    for (int i = 0; i < w * h; i++)
        randomPattern[i] = rand() & 0xFF; // 随机数, 取低8位

    currentPotential = 3;

    // We create 32 blocks in width dimension, and adjust the number of blocks for the height accordingly.
    // Always use block size of 16.
    bW = 16;
    bH = 16;
    nbW = w / bW;
    nbH = h / bH;
    if (w != bW * nbW || h != bH * nbH) {
        std::cout << "ERROR: Height or width seem to be not divisible by 16!" << std::endl;
        assert(0);
    }

    std::cout << "PixelSelector: Using block sizes: " << bW << ", " << bH << '\n';

    gradHist = new int[100 * (1 + nbW) * (1 + nbH)];
    ths = new float[(nbW) * (nbH) + 100];
    thsSmoothed = new float[(nbW) * (nbH) + 100];

    allowFast = false;
    gradHistFrame = 0;
}

PixelSelector::~PixelSelector()
{
    delete[] randomPattern;
    delete[] gradHist;
    delete[] ths;
    delete[] thsSmoothed;
}

//* 占据 below% 的梯度值作为阈值(below=0.5,即对应中位数作为阈值)
int computeHistQuantil(int* hist, float below)
{
    int th = hist[0] * below + 0.5f; // valid数量/2+0.5;这里below=0.5,对应中位数
    for (int i = 0; i < 90; i++) {
        th -= hist[i + 1];
        if (th < 0)
            return i;
    }
    return 90;
}

/**
 * @brief 计算直方图以及选点的阈值
 *
 * @param fh
 */
void PixelSelector::makeHists(const FrameHessian* const fh)
{
    gradHistFrame = fh;
    float* mapmax0 = fh->absSquaredGrad[0]; // 获取最底层梯度图

    int w = wG[0];
    int h = hG[0];

    // 用bW x bH(16x16)大小网格划分图像
    int w32 = nbW; // width块数
    int h32 = nbH; // height块数
    thsStep = w32; // 网格步长(width块数）

    //遍历每个块,统计每一个块的直方图
    for (int y = 0; y < h32; y++)
        for (int x = 0; x < w32; x++) {
            float* map0 = mapmax0 + bW * x + bH * y * w; //块起点在梯度图的位置
            int* hist0 = gradHist; //直方图储存位置; TODO: gradHist是否可以只用作暂态变量?// + 50*(x+y*w32);
            memset(hist0, 0, sizeof(int) * 50);

            for (int j = 0; j < bH; j++)
                for (int i = 0; i < bW; i++) {
                    // 该格里第(i,j)像素的整个图像坐标
                    int it = i + bW * x;
                    int jt = j + bH * y;
                    // 边缘的不要(对应位置之前没有计算梯度)
                    if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1)
                        continue;
                    int g = sqrtf(map0[i + j * w]); // 梯度平方和开根号
                    // 由于直方图只分了50个区间(memset那里)([0]用于统计valid梯度的数量),根据梯度平方根来统计直方图
                    if (g > 48)
                        g = 48;
                    hist0[g + 1]++;
                    hist0[0]++;
                }

            // 得到每一block的阈值=每一个块梯度平方根中位数+7
            ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd;
        }

    // 使用3*3的窗口求平均值平方来平滑
    for (int y = 0; y < h32; y++)
        for (int x = 0; x < w32; x++) {
            float sum = 0, num = 0;
            if (x > 0) {
                if (y > 0) {
                    num++;
                    sum += ths[x - 1 + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x - 1 + (y + 1) * w32];
                }
                num++;
                sum += ths[x - 1 + (y)*w32];
            }

            if (x < w32 - 1) {
                if (y > 0) {
                    num++;
                    sum += ths[x + 1 + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x + 1 + (y + 1) * w32];
                }
                num++;
                sum += ths[x + 1 + (y)*w32];
            }

            if (y > 0) {
                num++;
                sum += ths[x + (y - 1) * w32];
            }
            if (y < h32 - 1) {
                num++;
                sum += ths[x + (y + 1) * w32];
            }
            num++;
            sum += ths[x + y * w32];

            thsSmoothed[x + y * w32] = (sum / num) * (sum / num);
        }
}

/**
 * @brief
 * 递归makeMaps获得足够侯选点(优先级低层(分辨率高)>高层(分辨率低)),及可视化
 *
 * @param fh 帧Hessian数据结构
 * @param map_out 选出的地图点
 * @param density 当前层需要提取的点数（密度）
 * @param recursionsLeft 最大递归次数
 * @param plot 画图
 * @param thFactor 阈值因子
 * @return int 提取到的点数
 */
int PixelSelector::makeMaps(
    const FrameHessian* const fh,
    float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
    float numHave = 0;
    float numWant = density;
    float quotia;
    int idealPotential = currentPotential;

    // 利用FAST进行特征提取?
    //	if(setting_pixelSelectionUseFast>0 && allowFast)
    //	{
    //		memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
    //		std::vector<cv::KeyPoint> pts;
    //		cv::Mat img8u(hG[0],wG[0],CV_8U);
    //		for(int i=0;i<wG[0]*hG[0];i++)
    //		{
    //			float v = fh->dI[i][0]*0.8;
    //			img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
    //		}
    //		cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
    //		for(unsigned int i=0;i<pts.size();i++)
    //		{
    //			int x = pts[i].pt.x+0.5;
    //			int y = pts[i].pt.y+0.5;
    //			map_out[x+y*wG[0]]=1;
    //			numHave++;
    //		}
    //
    //		printf("FAST selection: got %f / %f!\n", numHave, numWant);
    //		quotia = numWant / numHave;
    //	}
    //	else
    {

        // the number of selected pixels behaves approximately as
        // K / (pot+1)^2, where K is a scene-dependent constant.
        // we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.
        if (fh != gradHistFrame)
            makeHists(fh);

        // select!
        //[ ***step 2*** ] 在当前帧上选择符合条件的像素
        Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor);

        // sub-select!
        numHave = n[0] + n[1] + n[2]; //各层精度点数量和
        quotia = numWant / numHave;

        //[ ***step 3*** ] 计算新的采像素点的, 范围大小, 相当于动态网格了, pot越小取得点越多
        // by default we want to over-sample by 40% just to be sure.
        float K = numHave * (currentPotential + 1) * (currentPotential + 1); // 相当于覆盖的面积, 每一个像素对应一个pot*pot
        idealPotential = sqrtf(K / numWant) - 1; // round down.
        if (idealPotential < 1)
            idealPotential = 1;

        //[ ***step 4*** ] 想要的数目和已经得到的数目, 大于或小于0.25都会重新采样一次
        if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1) {
            // re-sample to get more points!
            //  potential needs to be smaller
            if (idealPotential >= currentPotential)
                idealPotential = currentPotential - 1;

            //		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
            //				100*numHave/(float)(wG[0]*hG[0]),
            //				100*numWant/(float)(wG[0]*hG[0]),
            //				currentPotential,
            //				idealPotential);
            currentPotential = idealPotential;
            return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
        } else if (recursionsLeft > 0 && quotia < 0.25) {
            // re-sample to get less points!

            if (idealPotential <= currentPotential)
                idealPotential = currentPotential + 1;

            //		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
            //				100*numHave/(float)(wG[0]*hG[0]),
            //				100*numWant/(float)(wG[0]*hG[0]),
            //				currentPotential,
            //				idealPotential);
            currentPotential = idealPotential;
            return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
        }
    }

    //[ ***step 5*** ] 现在提取的还是多, 随机删除一些点
    int numHaveSub = numHave;
    if (quotia < 0.95) {
        int wh = wG[0] * hG[0];
        int rn = 0;
        unsigned char charTH = 255 * quotia;
        for (int i = 0; i < wh; i++) {
            if (map_out[i] != 0) {
                if (randomPattern[rn] > charTH) {
                    map_out[i] = 0;
                    numHaveSub--;
                }
                rn++; // TODO: rn = i? 多余的?
            }
        }
    }

    //	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
    //			100*numHave/(float)(wG[0]*hG[0]),
    //			100*numWant/(float)(wG[0]*hG[0]),
    //			currentPotential,
    //			idealPotential,
    //			100*numHaveSub/(float)(wG[0]*hG[0]));
    currentPotential = idealPotential;

    if (plot) {
        int w = wG[0];
        int h = hG[0];

        MinimalImageB3 img(w, h);

        for (int i = 0; i < w * h; i++) {
            float c = fh->dI[i][0] * 0.7;
            if (c > 255)
                c = 255;
            img.at(i) = Vec3b(c, c, c);
        }
        IOWrap::displayImage("Selector Image", &img);

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int i = x + y * w;
                if (map_out[i] == 1)
                    img.setPixelCirc(x, y, Vec3b(0, 255, 0));
                else if (map_out[i] == 2)
                    img.setPixelCirc(x, y, Vec3b(255, 0, 0));
                else if (map_out[i] == 4)
                    img.setPixelCirc(x, y, Vec3b(0, 0, 255));
            }
        IOWrap::displayImage("Selector Pixels", &img);
    }

    return numHaveSub;
}
//? 这个选点到底是不同层上, 还是论文里提到的不同阈值, 不同block???

/********************************
 * @ function:		根据阈值选择不同层上符合要求的像素
 *
 * @ param: 		fh						帧的一些信息
 * @				map_out					选中的像素点及所在层
 * @				pot(currentPotential)	选点的范围大小=3, 一个pot内选一个
 * @				thFactor				阈值因子(乘数)
 *
 * @ note:			返回的是每一层选择的点的个数
 *******************************/
Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
    float* map_out, int pot, float thFactor)
{
    // const 在*左, 指针内容不可改, 在*右指针不可改
    //  等价const Eigen::Vector3f * const
    Eigen::Vector3f const* const map0 = fh->dI;

    // 0, 1, 2层的梯度平方和
    float* mapmax0 = fh->absSquaredGrad[0];
    float* mapmax1 = fh->absSquaredGrad[1];
    float* mapmax2 = fh->absSquaredGrad[2];

    // 不同层的图像大小
    int w = wG[0];
    int w1 = wG[1];
    int w2 = wG[2];
    int h = hG[0];

    //? 这个是为了什么呢,
    //! 随机选这16个对应方向上的梯度和阈值比较
    //! 每个pot里面的方向随机选取的, 防止特征相同, 重复

    // 模都是1
    const Vec2f directions[16] = {
        Vec2f(0, 1.0000),
        Vec2f(0.3827, 0.9239),
        Vec2f(0.1951, 0.9808),
        Vec2f(0.9239, 0.3827),
        Vec2f(0.7071, 0.7071),
        Vec2f(0.3827, -0.9239),
        Vec2f(0.8315, 0.5556),
        Vec2f(0.8315, -0.5556),
        Vec2f(0.5556, -0.8315),
        Vec2f(0.9808, 0.1951),
        Vec2f(0.9239, -0.3827),
        Vec2f(0.7071, -0.7071),
        Vec2f(0.5556, 0.8315),
        Vec2f(0.9808, -0.1951),
        Vec2f(1.0000, 0.0000),
        Vec2f(0.1951, -0.9808)
    };

    memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));

    // 金字塔层阈值的减小倍数=0.75
    float dw1 = setting_gradDownweightPerLevel; //第二层
    float dw2 = dw1 * dw1; // 第三层

    // 第2层1个pot对应第1层4个pot, 第1层1个pot对应第0层的4个pot,
    // 第0层的4个pot里面只要选一个像素, 就不在对应高层的pot里面选了,
    // 但是还会在第0层的每个pot里面选大于阈值的像素
    // 阈值随着层数增加而下降
    // 从顶层向下层遍历, 写的挺有意思!
    int n3 = 0, n2 = 0, n4 = 0;
    //* 第2层中, 每隔pot选一个点遍历
    // |*|*|*| |*|*|*| |*|*|*| |*|*|*| |*|*|*| |*|*|*| |*|*|*| |*|*|*|:原图
    // |*|                             |*|                            :4:mx3 = 4 * pot = 12
    // |*|             |*|             |*|             |*|            :3:mx2 = 6
    // |*|     |*|     |*|     |*|     |*|     |*|     |*|     |*|    :2:mx1 = 3
    // |*|*|*| |*|*|*| |*|*|*| |*|*|*| |*|*|*| |*|*|*| |*|*|*| |*|*|*|:1
    // 把图像按每pot=3划分一个块,找出该块梯度最大的点(可能是为了避免点提取密集,所以每三个一个单元,但是感觉还是会有可能邻接)
    // 依次从0层梯度图进行阈值判断,小于阈值则到更上一层查找
    // 因此可以笼统的概括为找到每一个块里的特征点,并获得其精度等级(最低层精度越高)
    for (int y4 = 0; y4 < h; y4 += (4 * pot))
        for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
            // 该点的邻域(向上取4pot或末尾余数)大小
            int my3 = std::min((4 * pot), h - y4);
            int mx3 = std::min((4 * pot), w - x4);
            int bestIdx4 = -1;
            float bestVal4 = 0;
            // 随机系数
            Vec2f dir4 = directions[randomPattern[n2] & 0xF]; // 取低4位, 0-15, 和directions对应
            //* 上面的领域范围内, 在第1层进行遍历, 每隔pot一个点
            for (int y3 = 0; y3 < my3; y3 += (2 * pot))
                for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
                    int x34 = x3 + x4; // 对应第0层坐标
                    int y34 = y3 + y4;
                    // 继续确定该层上的邻域
                    int my2 = std::min((2 * pot), h - y34);
                    int mx2 = std::min((2 * pot), w - x34);
                    int bestIdx3 = -1;
                    float bestVal3 = 0;
                    Vec2f dir3 = directions[randomPattern[n2] & 0xF];
                    //* 上面的邻域范围内, 变换到第0层, 每隔pot遍历
                    //! 每个pot大小格里面一个大于阈值的最大的像素
                    for (int y2 = 0; y2 < my2; y2 += pot)
                        for (int x2 = 0; x2 < mx2; x2 += pot) {
                            int x234 = x2 + x34;
                            int y234 = y2 + y34;
                            int my1 = std::min(pot, h - y234);
                            int mx1 = std::min(pot, w - x234);
                            int bestIdx2 = -1;
                            float bestVal2 = 0;
                            Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                            //* 第0层中的,pot大小邻域内遍历
                            for (int y1 = 0; y1 < my1; y1 += 1)
                                for (int x1 = 0; x1 < mx1; x1 += 1) {
                                    assert(x1 + x234 < w);
                                    assert(y1 + y234 < h);
                                    int idx = x1 + x234 + w * (y1 + y234); // 像素id
                                    int xf = x1 + x234; // 像素坐标
                                    int yf = y1 + y234;

                                    if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4)
                                        continue;

                                    // 直方图求得阈值, 除以16确定在哪个阈值范围,
                                    //! 可以确定是每个grid, 16格大小
                                    float pixelTH0 = thsSmoothed[xf / bW + (yf / bH) * thsStep];
                                    float pixelTH1 = pixelTH0 * dw1;
                                    float pixelTH2 = pixelTH1 * dw2;

                                    float ag0 = mapmax0[idx]; // 第0层梯度
                                    if (ag0 > pixelTH0 * thFactor) {
                                        Vec2f ag0d = map0[idx].tail<2>(); // 后两位是图像导数
                                        float dirNorm = fabsf((float)(ag0d.dot(dir2))); // 以这个方向上的梯度来判断
                                        if (!setting_selectDirectionDistribution)
                                            dirNorm = ag0;

                                        if (dirNorm > bestVal2) // 在dir2上梯度更大则更新为新的侯选点
                                        {
                                            bestVal2 = dirNorm;
                                            bestIdx2 = idx;
                                            bestIdx3 = -2;
                                            bestIdx4 = -2;
                                        }
                                    }
                                    if (bestIdx3 == -2)
                                        continue; // 有了则不在其它层选点, 但是还会在该pot里选最大的

                                    // 第0层梯度不达标,看其第1层是否达标(提取要求变宽松)
                                    float ag1 = mapmax1[(int)(xf * 0.5f + 0.25f) + (int)(yf * 0.5f + 0.25f) * w1]; // 第1层
                                    if (ag1 > pixelTH1 * thFactor) {
                                        Vec2f ag0d = map0[idx].tail<2>();
                                        float dirNorm = fabsf((float)(ag0d.dot(dir3)));
                                        if (!setting_selectDirectionDistribution)
                                            dirNorm = ag1;

                                        if (dirNorm > bestVal3) {
                                            bestVal3 = dirNorm;
                                            bestIdx3 = idx;
                                            bestIdx4 = -2;
                                        }
                                    }
                                    if (bestIdx4 == -2)
                                        continue;

                                    float ag2 = mapmax2[(int)(xf * 0.25f + 0.125) + (int)(yf * 0.25f + 0.125) * w2]; // 第2层
                                    // thFactor = 2
                                    if (ag2 > pixelTH2 * thFactor) {
                                        Vec2f ag0d = map0[idx].tail<2>();
                                        float dirNorm = fabsf((float)(ag0d.dot(dir4)));
                                        if (!setting_selectDirectionDistribution)
                                            dirNorm = ag2;

                                        if (dirNorm > bestVal4) {
                                            bestVal4 = dirNorm;
                                            bestIdx4 = idx;
                                        }
                                    }
                                }

                            // 第0层的pot循环完, 若有则添加标志
                            if (bestIdx2 > 0) {
                                map_out[bestIdx2] = 1;
                                // 高层pot中有更好的了，满足更严格要求的，就不用满足pixelTH1的了
                                // bug bestVal3没有什么用，因为bestIdx3=-2直接continue了
                                bestVal3 = 1e10; // 第0层找到了, 就不在高层找了
                                n2++; // 计数
                            }
                        }
                    // 第0层没有, 则在第1层选
                    if (bestIdx3 > 0) {
                        map_out[bestIdx3] = 2;
                        bestVal4 = 1e10;
                        n3++;
                    }
                }
            // 第1层没有, 则在第2层选
            if (bestIdx4 > 0) {
                map_out[bestIdx4] = 4;
                n4++;
            }
        }

    return Eigen::Vector3i(n2, n3, n4); // 第0, 1, 2层选点的个数
}

}
