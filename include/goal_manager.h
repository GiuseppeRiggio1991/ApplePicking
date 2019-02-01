#include <ros/ros.h>
#include <hydra_utils/CloudService.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Float32MultiArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <sawyer_planner/AppleCheck.h>
#include <kalman_filter.h>
#include <vector>

class GoalManager
{

    public:
        GoalManager();
        ~GoalManager();

        void updateGoal(const ros::TimerEvent& event);
        bool appleCheck(sawyer_planner::AppleCheck::Request &req, sawyer_planner::AppleCheck::Response &res);
        bool removeApple(int apple_index);
    private:

        // NodeHandle
        ros::NodeHandle nh_;

        // Publishers
        ros::Publisher goal_pub_, goal_array_pub_;

        // Servers
        ros::ServiceServer apple_check_server_;

        // Clients
        ros::ServiceClient pcl_client_;

        ros::Timer timer_sub_;

        KalmanFilter kf_;

        Eigen::VectorXf apples_; // state for the kalman filter
        int state_size_;
        Eigen::VectorXf apples_observations_;

        geometry_msgs::Point goal_;
        std::vector<float> apples_array_;
};
