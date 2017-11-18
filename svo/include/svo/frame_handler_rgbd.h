#ifndef FRAME_HANDLER_RGBD_H
#define FRAME_HANDLER_RGBD_H

#include <set>
#include <svo/frame_handler_base.h>
#include <svo/initialization.h>
#include <svo/reprojector.h>
#include <vikit/abstract_camera.h>
#include <svo/feature_detection.h>

namespace svo
{

class FrameHandlerRGBD : public FrameHandlerBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FrameHandlerRGBD (vk::AbstractCamera* cam);

    virtual ~FrameHandlerRGBD ();

    void addImage (const cv::Mat& mono_img, const cv::Mat& depth_img, double timestamp);

    /// Get the last frame that has been processed.
    FramePtr lastFrame() { return last_frame_; }

    /// Get the set of spatially closest keyframes of the last frame.
    const set<FramePtr>& coreKeyframes() { return core_kfs_; }

protected:
    vk::AbstractCamera* cam_; //!< Camera model, can be ATAN, Pinhole or Ocam
                              //!(see vikit).
    Reprojector reprojector_; //!< Projects points from other keyframes into the
                              //!current frame
    FramePtr new_frame_;      //!< Current frame.
    FramePtr last_frame_;     //!< Last frame, not necessarily a keyframe.
    set<FramePtr> core_kfs_;  //!< Keyframes in the closer neighbourhood.
    vector<pair<FramePtr, size_t>> overlap_kfs_; //!< All keyframes with
                                                 //!overlapping field of view.
                                                 //!the paired number specifies
                                                 //!how many common mappoints
                                                 //!are observed TODO: why
                                                 //!vector!?

    initialization::KltHomographyInit klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.
    DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.

    /// Initialize the visual odometry algorithm.
    virtual void initialize ();

    /// Process the first frame and sets it as a keyframe.
    virtual UpdateResult processFirstFrame();

    /// Process all frames after the first frame.
    virtual UpdateResult processFrame();

    /// Try to relocalizing the frame at relative position to provided keyframe.
    virtual UpdateResult relocalizeFrame(
            const SE3d& T_cur_ref,
            FramePtr ref_keyframe);
    /// Reset the frame handler. Implement in derived class.
    virtual void resetAll();
};
}


#endif // FRAME_HANDLER_RGBD_H
