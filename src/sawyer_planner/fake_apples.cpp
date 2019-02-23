#include <ros/ros.h>
#include <hydra_utils/CloudService.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <cstdlib>
#include <ctime>

using namespace std;

pcl::PointCloud<pcl::PointXYZRGB> pc_array;

bool pub_fake_apples(hydra_utils::CloudService::Request &req, hydra_utils::CloudService::Response &res)
{

    if(pc_array.size() != 0)
    {
        sensor_msgs::PointCloud2 cloudMsg;
        pcl::toROSMsg(pc_array, cloudMsg);
        cloudMsg.header.stamp = ros::Time::now();
        cloudMsg.header.frame_id = "map";
        res.cloud = cloudMsg;
        return true;
    }
    else
    {
        return false;
    }

}

int main(int argc, char** argv)
{

    ros::init(argc, argv, "Fake_apples_node");

    ros::NodeHandle n;

    ros::ServiceServer service = n.advertiseService("/left_arm/mvps/segmentation_module/spheres", pub_fake_apples);

    ros::Rate loop_rate(1);

    pcl::PointXYZRGB pc;

    float LO = -0.02f;
    float HI = 0.02f;

    srand(static_cast<unsigned int>(clock()));

    int count = 0;
    while (ros::ok())
    {
        if (count > 10)
        {
            pc_array.clear();
            pc.x = 0.8 + (LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));
            pc.y = 0.25 + (LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));
            pc.z = 0.15 + (LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));

            pc_array.push_back(pc);
        }
        else
        {
            pc_array.clear();

            pc.x = 0.8 + (LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));
            pc.y = 0.2 + (LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));
            pc.z = 0.2 + (LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));

            pc_array.push_back(pc);
        }


        ros::spinOnce();
        loop_rate.sleep();

        count++;
        //if (count > 20)
        //   count = 0;
    }


}
