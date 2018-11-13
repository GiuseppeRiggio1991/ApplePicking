#include <goal_manager.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "Goal_Manager");

    GoalManager* goal = new GoalManager();

    ros::Rate loop_rate(100);

    while (ros::ok())
    {
        //goal->updateGoal();
        ros::spinOnce();
        loop_rate.sleep();
    }

    delete goal;

}

