#include <ros/ros.h>
#include <iostream>
#include <std_srvs/Trigger.h>
#include <hydra_utils/CloudService.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/Bool.h>

class HydraBridge
{
    public:
        HydraBridge();
        ~HydraBridge();

        void update();
        void enableBridge(const std_msgs::Bool::ConstPtr& enable);
        bool startPipeline(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
        
    private:

        bool enable_;

        ros::NodeHandle nh_;
        ros::ServiceServer pipeline_server_;
        ros::ServiceClient state_machine_client_;
        ros::ServiceClient camera_client_;
        ros::ServiceClient cloud_client_;
        ros::ServiceClient map_client_;
        ros::ServiceClient segmentation_client_;
        ros::ServiceClient clear_map_client_;
        ros::ServiceClient clear_segmentation_client_;
        
        ros::Subscriber enable_bridge_sub_;
};
