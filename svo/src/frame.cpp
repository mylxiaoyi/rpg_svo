// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <boost/bind.hpp>
#include <fast/fast.h>
#include <stdexcept>
#include <svo/config.h>
#include <svo/feature.h>
#include <svo/feature_detection.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <vikit/math_utils.h>
#include <vikit/performance_monitor.h>
#include <vikit/vision.h>

namespace svo
{

int Frame::frame_counter_ = 0;

Frame::Frame (vk::AbstractCamera* cam, const cv::Mat& img, double timestamp)
: id_ (frame_counter_++), timestamp_ (timestamp), cam_ (cam), key_pts_ (5),
  is_keyframe_ (false), v_kf_ (NULL)
{
    initFrame (img);
}

Frame::Frame (vk::AbstractCamera* cam, const cv::Mat& img, const cv::Mat& depth, double timestamp)
: id_ (frame_counter_++), timestamp_ (timestamp), cam_ (cam), key_pts_ (5),
  is_keyframe_ (false), v_kf_ (NULL)
{
    initFrame (img, depth);
}

Frame::~Frame ()
{
    std::for_each (fts_.begin (), fts_.end (), [&](Feature* i) { delete i; });
}

void Frame::initFrame (const cv::Mat& img)
{
    // check image
    if (img.empty () || img.type () != CV_8UC1 || img.cols != cam_->width () ||
        img.rows != cam_->height ())
        throw std::runtime_error ("Frame: provided image has not the same size "
                                  "as the camera model or image is not "
                                  "grayscale");

    // Set keypoints to NULL
    std::for_each (key_pts_.begin (), key_pts_.end (),
                   [&](Feature* ftr) { ftr = NULL; });

    // Build Image Pyramid
    frame_utils::createImgPyramid (img, max (Config::nPyrLevels (),
                                             Config::kltMaxLevel () + 1),
                                   img_pyr_);
}

void Frame::initFrame (const cv::Mat& img, const cv::Mat& depth)
{
    if (img.empty () || img.type () != CV_8UC1 || img.cols != cam_->width () ||
        img.rows != cam_->height ())
        throw std::runtime_error ("Frame: provided image has not the same size "
                                  "as the camera model or image is not "
                                  "grayscale.");

    if (depth.empty () || (depth.type () != CV_16U && depth.type () != CV_32F) ||
        depth.cols != cam_->width () || depth.rows != cam_->height ())
        throw std::runtime_error ("Frame: provided depth image has not the "
                                  "same size as the camera model or the depth "
                                  "image type is not correct.");

    float depthFactor = 1000.0;
    cv::Mat imDepth = depth;
    if (depth.type () != CV_32F) depth.convertTo (imDepth, CV_32F, 1.0/depthFactor);

    std::for_each (key_pts_.begin (), key_pts_.end (),
                   [&](Feature* ftr) { ftr = NULL; });

    frame_utils::buildImgPyramid (img, max (Config::nPyrLevels (), Config::kltMaxLevel () + 1),
                                  img_pyr_);
    frame_utils::buildImgPyramid (imDepth, max (Config::nPyrLevels (),
                                                Config::kltMaxLevel () + 1),
                                  depth_pyr_);

    // detect features
//    detectFeatures ();
}

void Frame::detectFeatures ()
{
    Features new_features;
    feature_detection::FastDetector detector (img ().cols, img ().rows,
                                              Config::gridSize (), Config::nPyrLevels ());
    detector.setExistingFeatures(this->fts_);
    detector.detect (this, img_pyr_, Config::triangMinCornerScore (), new_features);

    std::cout << "new_features.size = " << new_features.size () << std::endl;
    SE3d T_world_cur = T_f_w_.inverse();
    std::for_each (new_features.begin (), new_features.end (), [&](Feature* ft) {
        double depth = getDepth(depth_pyr_[0], ft->px[0], ft->px[1]);
        if (depth == 0.0)
        {
            delete ft;
            return ;
        }
        assert(depth > 0.0 && !std::isnan(depth) && !std::isinf(depth));
        Vector3d pos = T_world_cur * cam_->cam2world(ft->px[0], ft->px[1], depth);
        Point *new_point = new Point(pos);
        ft->point = new_point;
        this->addFeature(ft);
        new_point->addFrameRef(ft);

    });
    std::cout << "frame.fts.size = " << fts_.size() << std::endl;
}

void Frame::setKeyframe ()
{
    is_keyframe_ = true;
    setKeyPoints ();
}

void Frame::addFeature (Feature* ftr)
{
    fts_.push_back (ftr);
}

void Frame::setKeyPoints ()
{
    for (size_t i = 0; i < 5; ++i)
        if (key_pts_[i] != NULL)
            if (key_pts_[i]->point == NULL) key_pts_[i] = NULL;

    std::for_each (fts_.begin (), fts_.end (), [&](Feature* ftr) {
        if (ftr->point != NULL) checkKeyPoints (ftr);
    });
}

void Frame::checkKeyPoints (Feature* ftr)
{
    const int cu = cam_->width () / 2;
    const int cv = cam_->height () / 2;

    // center pixel
    if (key_pts_[0] == NULL)
        key_pts_[0] = ftr;
    else if (std::max (std::fabs (ftr->px[0] - cu), std::fabs (ftr->px[1] - cv)) <
             std::max (std::fabs (key_pts_[0]->px[0] - cu),
                       std::fabs (key_pts_[0]->px[1] - cv)))
        key_pts_[0] = ftr;

    if (ftr->px[0] >= cu && ftr->px[1] >= cv)
    {
        if (key_pts_[1] == NULL)
            key_pts_[1] = ftr;
        else if ((ftr->px[0] - cu) * (ftr->px[1] - cv) >
                 (key_pts_[1]->px[0] - cu) * (key_pts_[1]->px[1] - cv))
            key_pts_[1] = ftr;
    }
    if (ftr->px[0] >= cu && ftr->px[1] < cv)
    {
        if (key_pts_[2] == NULL)
            key_pts_[2] = ftr;
        else if ((ftr->px[0] - cu) * (ftr->px[1] - cv) >
                 (key_pts_[2]->px[0] - cu) * (key_pts_[2]->px[1] - cv))
            key_pts_[2] = ftr;
    }
    if (ftr->px[0] < cv && ftr->px[1] < cv)
    {
        if (key_pts_[3] == NULL)
            key_pts_[3] = ftr;
        else if ((ftr->px[0] - cu) * (ftr->px[1] - cv) >
                 (key_pts_[3]->px[0] - cu) * (key_pts_[3]->px[1] - cv))
            key_pts_[3] = ftr;
    }
    if (ftr->px[0] < cv && ftr->px[1] >= cv)
    {
        if (key_pts_[4] == NULL)
            key_pts_[4] = ftr;
        else if ((ftr->px[0] - cu) * (ftr->px[1] - cv) >
                 (key_pts_[4]->px[0] - cu) * (key_pts_[4]->px[1] - cv))
            key_pts_[4] = ftr;
    }
}

void Frame::removeKeyPoint (Feature* ftr)
{
    bool found = false;
    std::for_each (key_pts_.begin (), key_pts_.end (), [&](Feature*& i) {
        if (i == ftr)
        {
            i = NULL;
            found = true;
        }
    });
    if (found) setKeyPoints ();
}

bool Frame::isVisible (const Vector3d& xyz_w) const
{
    Vector3d xyz_f = T_f_w_ * xyz_w;
    if (xyz_f.z () < 0.0) return false; // point is behind the camera
    Vector2d px = f2c (xyz_f);
    if (px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width () && px[1] < cam_->height ())
        return true;
    return false;
}

float Frame::getDepth (const cv::Mat& depthImage, float x, float y, bool smoothing, float maxZError, bool estWithNeighborsIfNull)
{
    assert (!depthImage.empty ());
    assert (depthImage.type () == CV_16UC1 || depthImage.type () == CV_32FC1);

    int u = int(x + 0.5f);
    int v = int(y + 0.5f);
    if (u == depthImage.cols && x < float(depthImage.cols))
    {
        u = depthImage.cols - 1;
    }
    if (v == depthImage.rows && y < float(depthImage.rows))
    {
        v = depthImage.rows - 1;
    }

    if (!(u >= 0 && u < depthImage.cols && v >= 0 && v < depthImage.rows))
    {
        SVO_DEBUG_STREAM ("!(x >=0 && x<depthImage.cols && y >=0 && "
                          "y<depthImage.rows) cond failed! returning bad "
                          "point. (x="
                          << x << " (u=" << u << "), y=" << y << " (v=" << v
                          << "), cols=" << depthImage.cols
                          << ", rows=" << depthImage.rows << ")");
        return 0;
    }

    bool isInMM = depthImage.type () == CV_16UC1; // is in mm?

    // Inspired from RGBDFrame::getGaussianMixtureDistribution() method from
    // https://github.com/ccny-ros-pkg/rgbdtools/blob/master/src/rgbd_frame.cpp
    // Window weights:
    //  | 1 | 2 | 1 |
    //  | 2 | 4 | 2 |
    //  | 1 | 2 | 1 |
    int u_start = std::max (u - 1, 0);
    int v_start = std::max (v - 1, 0);
    int u_end = std::min (u + 1, depthImage.cols - 1);
    int v_end = std::min (v + 1, depthImage.rows - 1);

    float depth = 0.0f;
    if (isInMM)
    {
        if (depthImage.at<unsigned short> (v, u) > 0 &&
            depthImage.at<unsigned short> (v, u) <
            std::numeric_limits<unsigned short>::max ())
        {
            depth = float(depthImage.at<unsigned short> (v, u)) * 0.001f;
        }
    }
    else
    {
        depth = depthImage.at<float> (v, u);
    }

    if ((depth == 0.0f || !std::isfinite (depth)) && estWithNeighborsIfNull)
    {
        // all cells no2 must be under the zError to be accepted
        float tmp = 0.0f;
        int count = 0;
        for (int uu = u_start; uu <= u_end; ++uu)
        {
            for (int vv = v_start; vv <= v_end; ++vv)
            {
                if ((uu == u && vv != v) || (uu != u && vv == v))
                {
                    float d = 0.0f;
                    if (isInMM)
                    {
                        if (depthImage.at<unsigned short> (vv, uu) > 0 &&
                            depthImage.at<unsigned short> (vv, uu) <
                            std::numeric_limits<unsigned short>::max ())
                        {
                            d = float(depthImage.at<unsigned short> (vv, uu)) * 0.001f;
                        }
                    }
                    else
                    {
                        d = depthImage.at<float> (vv, uu);
                    }
                    if (d != 0.0f && std::isfinite (d))
                    {
                        if (tmp == 0.0f)
                        {
                            tmp = d;
                            ++count;
                        }
                        else if (std::fabs (d - tmp / float(count)) < maxZError)
                        {
                            tmp += d;
                            ++count;
                        }
                    }
                }
            }
        }
        if (count > 1)
        {
            depth = tmp / float(count);
        }
    }

    if (depth != 0.0f && std::isfinite (depth))
    {
        if (smoothing)
        {
            float sumWeights = 0.0f;
            float sumDepths = 0.0f;
            for (int uu = u_start; uu <= u_end; ++uu)
            {
                for (int vv = v_start; vv <= v_end; ++vv)
                {
                    if (!(uu == u && vv == v))
                    {
                        float d = 0.0f;
                        if (isInMM)
                        {
                            if (depthImage.at<unsigned short> (vv, uu) > 0 &&
                                depthImage.at<unsigned short> (vv, uu) <
                                std::numeric_limits<unsigned short>::max ())
                            {
                                d = float(depthImage.at<unsigned short> (vv, uu)) * 0.001f;
                            }
                        }
                        else
                        {
                            d = depthImage.at<float> (vv, uu);
                        }

                        // ignore if not valid or depth difference is too high
                        if (d != 0.0f && std::isfinite (d) && std::fabs (d - depth) < maxZError)
                        {
                            if (uu == u || vv == v)
                            {
                                sumWeights += 2.0f;
                                d *= 2.0f;
                            }
                            else
                            {
                                sumWeights += 1.0f;
                            }
                            sumDepths += d;
                        }
                    }
                }
            }
            // set window weight to center point
            depth *= 4.0f;
            sumWeights += 4.0f;

            // mean
            depth = (depth + sumDepths) / sumWeights;
        }
    }
    else
    {
        depth = 0;
    }
    return depth;
}


/// Utility functions for the Frame class
namespace frame_utils
{

void createImgPyramid (const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
    pyr.resize (n_levels);
    pyr[0] = img_level_0;
    for (int i = 1; i < n_levels; ++i)
    {
        pyr[i] = cv::Mat (pyr[i - 1].rows / 2, pyr[i - 1].cols / 2, CV_8U);
        vk::halfSample (pyr[i - 1], pyr[i]);
    }
}

void buildImgPyramid (const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
    pyr.resize (n_levels);
    pyr[0] = img_level_0;
    for (int i = 1; i < n_levels; i++)
    {
        cv::pyrDown (pyr[i - 1], pyr[i],
                     cv::Size (pyr[i - 1].cols / 2, pyr[i - 1].rows / 2));
    }
}

bool getSceneDepth (const Frame& frame, double& depth_mean, double& depth_min)
{
    vector<double> depth_vec;
    depth_vec.reserve (frame.fts_.size ());
    depth_min = std::numeric_limits<double>::max ();
    for (auto it = frame.fts_.begin (), ite = frame.fts_.end (); it != ite; ++it)
    {
        if ((*it)->point != NULL)
        {
            const double z = frame.w2f ((*it)->point->pos_).z ();
            depth_vec.push_back (z);
            depth_min = fmin (z, depth_min);
        }
    }
    if (depth_vec.empty ())
    {
        SVO_WARN_STREAM (
        "Cannot set scene depth. Frame has no point-observations!");
        return false;
    }
    depth_mean = vk::getMedian (depth_vec);
    return true;
}

} // namespace frame_utils
} // namespace svo
