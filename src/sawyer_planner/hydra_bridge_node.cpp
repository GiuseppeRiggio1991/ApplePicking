#include <hydra_bridge.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "Hydra_Bridge");

    HydraBridge* hydra = new HydraBridge();

    ros::Rate loop_rate(1000);

    while (ros::ok())
    {
        hydra->update();
        loop_rate.sleep();
    }

    delete hydra;

}

