#include <hydra_bridge.h>

HydraBridge::HydraBridge()
{
    pipeline_server_ = nh_.advertiseService("/sawyer_planner/start_pipeline", &HydraBridge::startPipeline, this);

    camera_client_ = nh_.serviceClient<std_srvs::Trigger>("/left_arm/mvps/camera_module/query_data");
    cloud_client_ = nh_.serviceClient<hydra_utils::CloudService>("/left_arm/mvps/camera_module/cloud");
    map_client_ = nh_.serviceClient<std_srvs::Trigger>("/left_arm/mvps/mapping_module/add_frame");
    segmentation_client_ = nh_.serviceClient<std_srvs::Trigger>("/left_arm/mvps/segmentation_module/start");
    clear_map_client_ = nh_.serviceClient<std_srvs::Trigger>("/left_arm/mvps/mapping_module/clear_map");
    clear_segmentation_client_ = nh_.serviceClient<std_srvs::Trigger>("/left_arm/mvps/segmentation_module/clear");

    state_machine_client_ = nh_.serviceClient<std_srvs::Trigger>("mvps/state_machine/start");

    enable_bridge_sub_ = nh_.subscribe("/sawyer_planner/enable_bridge", 1, &HydraBridge::enableBridge, this);

    enable_ = false;
}

HydraBridge::~HydraBridge()
{
}

void HydraBridge::update()
{
    std_srvs::Trigger srv;
    hydra_utils::CloudService cloud_srv;
    pcl::PointCloud<pcl::PointXYZRGB> cloud;

    if (enable_)
    {
        if(camera_client_.call(srv))
        {
            ros::Duration(0.15).sleep();
            if(map_client_.call(srv))
            {
                ros::Duration(0.35).sleep();
                if(segmentation_client_.call(srv))
                {
                    ros::Duration(0.35).sleep();
                    clear_map_client_.call(srv);
                    clear_segmentation_client_.call(srv);
                }
            }
        }
    }

    ros::spinOnce();
}

bool HydraBridge::startPipeline(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res)
{
    std_srvs::Trigger srv;

    res.success = true;

    if(!state_machine_client_.call(srv))
    {
        res.success = false;
        return false;
    }

    return true;
}

void HydraBridge::enableBridge(const std_msgs::Bool::ConstPtr& enable)
{
    enable_ = enable->data;
}
