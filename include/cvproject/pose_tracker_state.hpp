#pragma once

#include <string>

namespace cvproject {

enum class PoseTrackerState {
    Initializing,
    Tracking,
    WeakTracking,
    Relocalizing,
    Lost,
};

std::string trackerStateToString(PoseTrackerState state);

}  // namespace cvproject
