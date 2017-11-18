#include <svo/config.h>
#include <svo/depth_filter.h>
#include <svo/feature.h>
#include <svo/frame.h>
#include <svo/frame_handler_rgbd.h>
#include <svo/map.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>

namespace svo
{
FrameHandlerRGBD::FrameHandlerRGBD (vk::AbstractCamera* cam)
: FrameHandlerBase (), cam_ (cam), reprojector_ (cam_, map_), depth_filter_ (NULL)
{
    initialize ();
}

void FrameHandlerRGBD::initialize ()
{
    feature_detection::DetectorPtr feature_detector (
    new feature_detection::FastDetector (cam_->width (), cam_->height (),
                                         Config::gridSize (), Config::nPyrLevels ()));
    DepthFilter::callback_t depth_filter_cb =
    boost::bind (&MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
    depth_filter_ = new DepthFilter (feature_detector, depth_filter_cb);
    //    depth_filter_->startThread ();
}

FrameHandlerRGBD::~FrameHandlerRGBD ()
{
    delete depth_filter_;
}

void FrameHandlerRGBD::addImage (const cv::Mat& mono_img, const cv::Mat& depth_img, double timestamp)
{
    if (!startFrameProcessingCommon (timestamp)) return;

    core_kfs_.clear ();
    overlap_kfs_.clear ();

    // create new frame
    SVO_START_TIMER ("pyramid_creation");
    new_frame_.reset (new Frame (cam_, mono_img.clone (), depth_img.clone (), timestamp));
    SVO_STOP_TIMER ("pyramid_creation");

    // process frame
    UpdateResult res = RESULT_FAILURE;
    if (stage_ == STAGE_FIRST_FRAME)
        res = processFirstFrame ();
    else if (stage_ == STAGE_DEFAULT_FRAME)
        res = processFrame ();
    else if (stage_ == STAGE_RELOCALIZING)
        res = relocalizeFrame (SE3d (Matrix3d::Identity (), Vector3d::Zero ()),
                               map_.getClosestKeyframe (last_frame_));

    // set last frame
    last_frame_ = new_frame_;
    new_frame_.reset ();

    // finish processing
    finishFrameProcessingCommon (last_frame_->id_, res, last_frame_->nObs ());
}

FrameHandlerBase::UpdateResult FrameHandlerRGBD::processFirstFrame ()
{
    new_frame_->T_f_w_ = SE3d (Matrix3d::Identity (), Vector3d::Zero ());
    new_frame_->detectFeatures ();
    new_frame_->setKeyframe ();
    map_.addKeyframe (new_frame_);
    stage_ = STAGE_DEFAULT_FRAME;
    SVO_INFO_STREAM ("Init: Selected first frame.");
    return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerRGBD::processFrame ()
{
    SVO_INFO_STREAM ("process frame");
    // Set initial pose TODO use prior
    new_frame_->T_f_w_ = last_frame_->T_f_w_;

    // sparse image align
    SVO_INFO_STREAM ("sparse image align starts");
    SVO_START_TIMER ("sparse_img_align");
    SparseImgAlign img_align (Config::kltMaxLevel (), Config::kltMinLevel (),
                              30, SparseImgAlign::GaussNewton, false, false);
    size_t img_align_n_tracked = img_align.run (last_frame_, new_frame_);
    SVO_STOP_TIMER ("sparse_img_align");
    SVO_LOG (img_align_n_tracked);
    SVO_INFO_STREAM ("Img Align:\t Tracked = " << img_align_n_tracked);
    SVO_INFO_STREAM ("sparse image align ends");

    // map reprojection & feature alignment
    SVO_INFO_STREAM ("map reprojection & feature alignment starts");
    SVO_START_TIMER ("reproject");
    reprojector_.reprojectMap (new_frame_, overlap_kfs_);
    SVO_STOP_TIMER ("reproject");
    const size_t repr_n_new_references = reprojector_.n_matches_;
    const size_t repr_n_mps = reprojector_.n_trials_;
    SVO_LOG2 (repr_n_mps, repr_n_new_references);
    SVO_INFO_STREAM ("Reprojection:\t nPoints = "
                     << repr_n_mps << "\t \t nMatches = " << repr_n_new_references);
    if (repr_n_new_references < Config::qualityMinFts ())
    {
        SVO_WARN_STREAM_THROTTLE (1.0, "Not enough matched features.");
        new_frame_->T_f_w_ =
        last_frame_->T_f_w_; // reset to avoid crazy pose jumps
        tracking_quality_ = TRACKING_INSUFFICIENT;
        return RESULT_FAILURE;
    }
    SVO_INFO_STREAM ("map reprojection & feature alignment ends");

    // pose optimization
    SVO_INFO_STREAM ("pose optimization starts");
    SVO_START_TIMER ("pose_optimizer");
    size_t sfba_n_edges_final;
    double sfba_thresh, sfba_error_init, sfba_error_final;
    pose_optimizer::optimizeGaussNewton (Config::poseOptimThresh (),
                                         Config::poseOptimNumIter (), false,
                                         new_frame_, sfba_thresh, sfba_error_init,
                                         sfba_error_final, sfba_n_edges_final);
    SVO_STOP_TIMER ("pose_optimizer");
    SVO_LOG4 (sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
    SVO_INFO_STREAM ("PoseOptimizer:\t ErrInit = " << sfba_error_init
                                                   << "px\t thresh = " << sfba_thresh);
    SVO_INFO_STREAM ("PoseOptimizer:\t ErrFin. = "
                     << sfba_error_final << "px\t nObsFin. = " << sfba_n_edges_final);
    if (sfba_n_edges_final < 20)
    {
        SVO_WARN_STREAM ("sfba_n_edges_final < 20: " << sfba_n_edges_final);
        return RESULT_FAILURE;
    }
    SVO_INFO_STREAM ("pose optimization ends");

    // structure optimization
    SVO_INFO_STREAM ("structure optimization starts");
    SVO_START_TIMER ("point_optimizer");
    optimizeStructure (new_frame_, Config::structureOptimMaxPts (),
                       Config::structureOptimNumIter ());
    SVO_STOP_TIMER ("point_optimizer");
    SVO_INFO_STREAM ("structure optimization ends");

    // select keyframe
    core_kfs_.insert (new_frame_);
    setTrackingQuality (sfba_n_edges_final);
    if (tracking_quality_ == TRACKING_INSUFFICIENT)
    {
        new_frame_->T_f_w_ =
        last_frame_->T_f_w_; // reset to avoid crazy pose jumps
        return RESULT_FAILURE;
    }
    //    double depth_mean, depth_min;
    //    frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
    //    if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
    //    {
    //      depth_filter_->addFrame(new_frame_);
    //      return RESULT_NO_KEYFRAME;
    //    }
    new_frame_->setKeyframe ();
    SVO_INFO_STREAM ("New keyframe selected.");

    std::cout << "new_frame_->fts_.size = " << new_frame_->fts_.size () << std::endl;
    // new keyframe selected
    for (Features::iterator it = new_frame_->fts_.begin ();
         it != new_frame_->fts_.end (); ++it)
        if ((*it)->point != NULL) (*it)->point->addFrameRef (*it);
    map_.point_candidates_.addCandidatePointToFrame (new_frame_);

    // if limited number of keyframes, remove the one furthest apart
    if (Config::maxNKfs () > 2 && map_.size () >= Config::maxNKfs ())
    {
        SVO_WARN_STREAM ("remove furthest frame");
        FramePtr furthest_frame = map_.getFurthestKeyframe (new_frame_->pos ());
        depth_filter_->removeKeyframe (furthest_frame); // TODO this interrupts
                                                        // the mapper thread,
                                                        // maybe we can solve
                                                        // this better
        map_.safeDeleteFrame (furthest_frame);
    }

    // add keyframe to map
    map_.addKeyframe (new_frame_);
    new_frame_->detectFeatures ();
    std::cout << "new_frame_->fts.size = " << new_frame_->fts_.size () << std::endl;

    return RESULT_IS_KEYFRAME;
}

void FrameHandlerRGBD::resetAll ()
{
    resetCommon ();
    last_frame_.reset ();
    new_frame_.reset ();
    core_kfs_.clear ();
    overlap_kfs_.clear ();
    depth_filter_->reset ();
}

FrameHandlerBase::UpdateResult
FrameHandlerRGBD::relocalizeFrame (const SE3d& T_cur_ref, FramePtr ref_keyframe)
{
    SVO_WARN_STREAM_THROTTLE (1.0, "Relocalizing frame");
    if (ref_keyframe == nullptr)
    {
        SVO_INFO_STREAM ("No reference keyframe.");
        return RESULT_FAILURE;
    }
    SparseImgAlign img_align (Config::kltMaxLevel (), Config::kltMinLevel (),
                              30, SparseImgAlign::GaussNewton, false, false);
    size_t img_align_n_tracked = img_align.run (ref_keyframe, new_frame_);
    if (img_align_n_tracked > 30)
    {
        SE3d T_f_w_last = last_frame_->T_f_w_;
        last_frame_ = ref_keyframe;
        FrameHandlerBase::UpdateResult res = processFrame ();
        if (res != RESULT_FAILURE)
        {
            stage_ = STAGE_DEFAULT_FRAME;
            SVO_INFO_STREAM ("Relocalization successful.");
        }
        else
            new_frame_->T_f_w_ =
            T_f_w_last; // reset to last well localized pose
        return res;
    }
    return RESULT_FAILURE;
}
}
