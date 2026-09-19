#include "cvproject/pose_tracker_state.hpp"

namespace cvproject {

std::string trackerStateToString(const PoseTrackerState state) {
    switch (state) {
        case PoseTrackerState::Initializing:
            return "INITIALIZING";
        case PoseTrackerState::Tracking:
            return "TRACKING";
        case PoseTrackerState::WeakTracking:
            return "WEAK_TRACKING";
        case PoseTrackerState::Relocalizing:
            return "RELOCALIZING";
        case PoseTrackerState::Lost:
            return "LOST";
    }

    return "UNKNOWN";
}

}  // namespace cvproject
