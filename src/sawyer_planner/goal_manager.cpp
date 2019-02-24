#include <goal_manager.h>

GoalManager::GoalManager()
{
    timer_sub_ = nh_.createTimer(ros::Duration(0.01), &GoalManager::updateGoal, this);

    // Publishers
    goal_pub_ = nh_.advertise<geometry_msgs::Point>("/sawyer_planner/goal", 1);
    goal_array_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/sawyer_planner/goal_array", 1);
    raw_points_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/sawyer_planner/raw_points", 1);

    // Server
    apple_check_server_ = nh_.advertiseService("/sawyer_planner/apple_check", &GoalManager::appleCheck, this);

    // Clients
    pcl_client_ = nh_.serviceClient<hydra_utils::CloudService>("/left_arm/mvps/segmentation_module/spheres"); 

    state_size_ = 0;

    // get params
    ros::NodeHandle nh("~");
    if (nh.hasParam("initial_covariance"))
        nh.getParam("initial_covariance", covariance_);
    else
    {
        ROS_WARN("initial covariance is not set! Using default!");
        covariance_ = 0.5f;
    }

    if (nh.hasParam("obs_covariance"))
        nh.getParam("obs_covariance", obs_covariance_);
    else
    {
        ROS_WARN("observation covariance is not set! Using default!");
        obs_covariance_ = 0.1f;
    }

    if (nh.hasParam("model_covariance"))
        nh.getParam("model_covariance", me_covariance_);
    else
    {
        ROS_WARN("model covariance is not set! Using default!");
        me_covariance_ = 0.0f;
    }
}

GoalManager::~GoalManager()
{

    ros::NodeHandle nh("~");
    if (nh.hasParam("initial_covariance"))
        nh.deleteParam("initial_covariance");
    if (nh.hasParam("obs_covariance"))
        nh.deleteParam("obs_covariance");
    if (nh.hasParam("model_covariance"))
        nh.deleteParam("model_covariance");
}

bool GoalManager::removeApple(int apple_index)
{
    Eigen::MatrixXf covariance = kf_.getCovariance();
    Eigen::MatrixXf me_covariance = kf_.getModelErrorCovariance();

    // remove from the state
    if (apples_.size() > 3)
    {
        if (apple_index <= apples_.size() - 3)
        {
            apples_.segment(apple_index, apples_.size() - 3 - apple_index) = apples_.tail(apples_.size() - apple_index - 3);

            // remove row
            covariance.block(apple_index, 0, covariance.rows() - 3 - apple_index, covariance.cols()) = covariance.block(apple_index + 3, 0, covariance.rows() - apple_index - 3, covariance.cols());
            me_covariance.block(apple_index, 0, me_covariance.rows() - 3 - apple_index, me_covariance.cols()) = me_covariance.block(apple_index + 3, 0, me_covariance.rows() - apple_index - 3, me_covariance.cols());

            // remove column
            covariance.block(0, apple_index, covariance.rows(), covariance.cols() - 3 - apple_index) = covariance.block(0, apple_index + 3, covariance.rows(), covariance.cols() - apple_index - 3);
            me_covariance.block(0, apple_index, me_covariance.rows(), me_covariance.cols() - 3 - apple_index) = me_covariance.block(0, apple_index + 3, me_covariance.rows(), me_covariance.cols() - apple_index - 3);

            // reshape
            apples_.conservativeResize(apples_.size() - 3);
            covariance.conservativeResize(covariance.rows() - 3, covariance.cols() - 3);
            me_covariance.conservativeResize(me_covariance.rows() - 3, me_covariance.cols() - 3);
        }else{
            ROS_WARN("tried to remove apple from the state that doesn't exist");
            return false;
        }
    }
    else
    {
        apples_.resize(0);
        covariance.resize(0, 0);
        me_covariance.resize(0, 0);
    }

    state_size_ = apples_.size();
    Eigen::MatrixXf A = Eigen::MatrixXf::Identity(state_size_, state_size_);
    Eigen::MatrixXf C = A;
    LinearModel model(A, C); // it doesn't really matter to update C
    kf_.setModel(model);
    kf_.setInitialStateAndCovariance(apples_, covariance);
    kf_.setModelErrorCovariance(me_covariance);

    ROS_INFO("Removed from the state");
    return true;

}


bool GoalManager::appleCheck(sawyer_planner::AppleCheck::Request &req, sawyer_planner::AppleCheck::Response &res)
{
    float threshold = 0.0001; // change with some parameter
    float distance = std::numeric_limits<double>::infinity();

    hydra_utils::CloudService cloud_srv;
    bool is_there = false;
    int tries = 0;
    int apple_index = apples_.size();
    int observed_index = 0;
    pcl::PointCloud<pcl::PointXYZRGBA> cloud;

    // identify the apple in the state
    for (int i = 0; i < apples_.size(); i += 3)
    {

        if (sqrt( pow(req.apple_pose.x - apples_[i], 2) + pow(req.apple_pose.y - apples_[i+1], 2) + pow(req.apple_pose.z - apples_[i+2], 2)) < distance)
        {
            distance = sqrt( pow(req.apple_pose.x - apples_[i], 2) + pow(req.apple_pose.y - apples_[i+1], 2) + pow(req.apple_pose.z - apples_[i+2], 2));
            apple_index = i;
            ROS_WARN("apple index: %i", i);

        }
    }

    // if (distance > threshold)
    // {
    //     ROS_WARN("The requested apple was not in the state!");
    //     res.apple_is_there = false;
    //     return true;
    // }
    // else
    // {
    //     distance = std::numeric_limits<double>::infinity();

    //     while (tries < 10 && is_there == false)
    //     {
    //         if (pcl_client_.call(cloud_srv))
    //         {

    //             pcl::fromROSMsg(cloud_srv.response.cloud, cloud);
    
    //             for (int i = 0; i < cloud.size(); i++)
    //             {
    
    //                 pcl::PointXYZRGBA& p = cloud.points[i];
    
    //                 if (sqrt( pow(p.x - apples_[apple_index], 2) + pow(p.y - apples_[apple_index + 1], 2) + pow(p.z - apples_[apple_index + 2], 2)) < distance)
    //                 {   
    //                     distance = sqrt( pow(p.x - apples_[apple_index], 2) + pow(p.y - apples_[apple_index + 1], 2) + pow(p.z - apples_[apple_index + 2], 2));
    //                     observed_index = i;
    //                 }
    //             }
    
    //             if (distance < threshold)
    //             {   
    //                 is_there = true;
    //                 break;
    //             }
    
    //         }
    //         ros::Duration(0.1).sleep();
    //         tries++;
    //     }

    //     if (is_there)
    //     {
    //         // check if there is another apple closer to the observed one (for close apples)
    //         distance = std::numeric_limits<double>::infinity();

    //         pcl::PointXYZRGBA& p = cloud.points[observed_index];

    //         for (int i = 0; i < apples_.size(); i += 3)
    //         {
    //             if (i != apple_index)
    //             {
    //                 if (sqrt( pow(req.apple_pose.x - apples_[i], 2) + pow(req.apple_pose.y - apples_[i+1], 2) + pow(req.apple_pose.z - apples_[i+2], 2)) < distance)
    //                 {
    //                     distance = sqrt( pow(req.apple_pose.x - apples_[i], 2) + pow(req.apple_pose.y - apples_[i+1], 2) + pow(req.apple_pose.z - apples_[i+2], 2));
    //                 }
    //             }
    //         }
        
    //         if (sqrt( pow(p.x - apples_[apple_index], 2) + pow(p.y - apples_[apple_index + 1], 2) + pow(p.z - apples_[apple_index + 2], 2)) < distance)
    //         {
    //             res.apple_is_there = true;
    //             return true;
    //         }
    //         else
    //         {
    //             // the observed apple is closer to another apple, so it should be that one instead of the requested one
    //             res.apple_is_there = false;
    //             return true;
    //         }
    //     }
    //     else
        // {
            removeApple(apple_index);
            res.apple_is_there = false;
            return true;
        // }
    // }
}

void GoalManager::updateGoal(const ros::TimerEvent& event)
{
    hydra_utils::CloudService cloud_srv;
    if (pcl_client_.call(cloud_srv))
    {
        pcl::PointCloud<pcl::PointXYZRGBA> cloud;
        pcl::fromROSMsg(cloud_srv.response.cloud, cloud);

        std::vector<float> raw_points;
        for (int i = 0; i < cloud.size(); i++)
        {
            pcl::PointXYZRGBA& p = cloud.points[i];
            raw_points.insert(raw_points.end(), {p.x, p.y, p.z});
        }
        std_msgs::Float32MultiArray raw_points_msg;
        raw_points_msg.data = raw_points;
        raw_points_pub_.publish(raw_points_msg);

        std::cout << "apples_.size(): " << apples_.size() << std::endl;
        if (init_update_) // had to change this because it kept adding the last apples in view again
        // if (apples_.size() == 0)
        {
            init_update_ = false;
            apples_.resize(cloud.size() * 3);
            state_size_ = apples_.size();

            for (int i = 0; i < cloud.size(); i++)
            {
                pcl::PointXYZRGBA& p = cloud.points[i];
                std::cout << "raw points: " << p << std::endl;
                apples_[i * 3] = p.x;
                apples_[i * 3 + 1] = p.y;
                apples_[i * 3 + 2] = p.z;
            }

            // create initial covariance matrix assuming diagonal matrix
            Eigen::VectorXf covariance_vector = covariance_ * Eigen::VectorXf::Ones(state_size_);
            Eigen::MatrixXf covariance_matrix = covariance_vector.asDiagonal();

            Eigen::VectorXf me_covariance_vector = me_covariance_ * Eigen::VectorXf::Ones(state_size_);
            Eigen::MatrixXf me_covariance_matrix = me_covariance_vector.asDiagonal();


            // create the model
            //   _
            // _|  x(k+1) = A x(k)    A = C = I
            //  |_ y(k+1) = C x(k)
            //

            Eigen::MatrixXf A = Eigen::MatrixXf::Identity(state_size_, state_size_);
            Eigen::MatrixXf C = A;

            LinearModel model(A, C);
            kf_.setModel(model);
            kf_.setInitialStateAndCovariance(apples_, covariance_matrix);
            kf_.setModelErrorCovariance(me_covariance_matrix);
        }
        else
        {

            state_size_ = apples_.size();

            Eigen::VectorXf apples = apples_;
            Eigen::VectorXf observations;
            Eigen::VectorXf new_apples;
            Eigen::MatrixXf C;

            for (int i = 0; i < cloud.size(); i++)
            {

                pcl::PointXYZRGBA& p = cloud.points[i];
                std::cout << "raw points: " << p << std::endl;

                float distance = std::numeric_limits<double>::infinity();

                float threshold = 0.2; // change with some parameter
                int index = state_size_; 

                for (int j = 0; j < apples.size(); j = j + 3)
                {
                    
                    if (sqrt( pow(p.x - apples[j], 2) + pow(p.y - apples[j+1], 2) + pow(p.z - apples[j+2], 2)) < distance)
                    {
                        distance = sqrt( pow(p.x - apples[j], 2) + pow(p.y - apples[j+1], 2) + pow(p.z - apples[j+2], 2));
                        if (distance < threshold)
                        {
                            index = j;
                            
                        }
                    }
                    
                }

                if (index != state_size_)
                {

                    // I'm virtually erasing the mathing apple keeping the same indeces
                    apples[index] = apples[index+1] = apples[index+2] = std::numeric_limits<double>::infinity();


                    // there is a match
                    std::cout << "apple matched" << std::endl;
                    observations.conservativeResize(observations.size()+3); //x, y, z of the new point

                    observations[observations.size()-3] = p.x;
                    observations[observations.size()-2] = p.y;
                    observations[observations.size()-1] = p.z;
                    
                    C.conservativeResize(observations.size(), state_size_);
                    C.block(observations.size()-3, 0, 3, state_size_) = Eigen::MatrixXf::Zero(3, state_size_);

                    C.block(observations.size()-3, index, 3, 3) = Eigen::Matrix3f::Identity();
                }
                // else
                // {
                //     // no match, it is a new apple
                //     std::cout << "new apple" << std::endl;
                //     new_apples.conservativeResize(new_apples.size()+3);

                //     new_apples[new_apples.size()-3] = p.x;
                //     new_apples[new_apples.size()-2] = p.y;
                //     new_apples[new_apples.size()-1] = p.z;

                // }
            }

            Eigen::VectorXf obs_covariance_vector = obs_covariance_ * Eigen::VectorXf::Ones(observations.size());
            Eigen::MatrixXf observation_covariance = obs_covariance_vector.asDiagonal();

            LinearModel model = kf_.getCurrentModel();
            Eigen::MatrixXf A = model.getA();

            model.setMatrices(A, C);
            kf_.setModel(model);
            kf_.estimate(observations, observation_covariance);

            apples_ = kf_.getState();

            // add the new apples to the state
            std::cout << "new_apples.size(): " << new_apples.size() << std::endl;
            apples_.conservativeResize(apples_.size() + new_apples.size());
            apples_.block(state_size_, 0, new_apples.size(), 1) = new_apples;

            Eigen::MatrixXf state_covariance = kf_.getCovariance();
            Eigen::VectorXf covariance_vector = state_covariance.diagonal();
            covariance_vector.conservativeResize(apples_.size());
            covariance_vector.block(state_size_, 0, new_apples.size(), 1) = covariance_ * Eigen::VectorXf::Ones(new_apples.size());
            Eigen::MatrixXf covariance_matrix = covariance_vector.asDiagonal();


            Eigen::MatrixXf current_me_covariance = kf_.getModelErrorCovariance();
            Eigen::VectorXf me_covariance_vector = current_me_covariance.diagonal();
            me_covariance_vector.conservativeResize(apples_.size());
            me_covariance_vector.block(state_size_, 0, new_apples.size(), 1) = me_covariance_ * Eigen::VectorXf::Ones(new_apples.size());

            Eigen::MatrixXf me_covariance_matrix = me_covariance_vector.asDiagonal();

            

            C.conservativeResize(observations.size(), apples_.size());
            C.block(0, state_size_, observations.size(), new_apples.size()) = Eigen::MatrixXf::Zero(observations.size(), new_apples.size());

            state_size_ = apples_.size();
            A = Eigen::MatrixXf::Identity(state_size_, state_size_);

            model.setMatrices(A, C); // it doesn't really matter to update C
            kf_.setModel(model);
            kf_.setInitialStateAndCovariance(apples_, covariance_matrix);
            kf_.setModelErrorCovariance(me_covariance_matrix);
             
       }

        
        /*Eigen::MatrixXf covariance_matrix = kf_.getCovariance();
    
        std::cout << "N: " << apples_.size() << std::endl;

        for (int i = 0; i < apples_.size(); i = i + 3)
        {
            std::cout << i << " " << apples_[i] << " " << apples_[i+1] << " " << apples_[i+2]
                           << " " << covariance_matrix(i, i) << " " << covariance_matrix(i+1, i+1) << " " << covariance_matrix(i+2, i+2)  << std::endl; 
        }
        std::cout << std::endl;*/

    }else{
        ROS_WARN("Failed to call cloud srv for apple positions");
    }


    Eigen::MatrixXf covariance_matrix = kf_.getCovariance();

    std::cout << "N: " << apples_.size() << std::endl;

    apples_array_.clear();
    for (int i = 0; i < apples_.size(); i = i + 3)
    {
        std::cout << i << " " << apples_[i] << " " << apples_[i+1] << " " << apples_[i+2]
                           << " " << covariance_matrix(i, i) << " " << covariance_matrix(i+1, i+1) << " " << covariance_matrix(i+2, i+2)  << std::endl;
        apples_array_.insert(apples_array_.end(), {apples_[i], apples_[i+1], apples_[i+2]});
    }
    std::cout << std::endl;

    // publish apples array
    std_msgs::Float32MultiArray apples_array_msg_;
    apples_array_msg_.data = apples_array_;
    goal_array_pub_.publish(apples_array_msg_);



    /*************************************************/
    /*          Set the goal from the state          */
    /*************************************************/

    if (apples_.size()!=0)
    {
        goal_.x = apples_[0];
        goal_.y = apples_[1];
        goal_.z = apples_[2];

        goal_pub_.publish(goal_);
    }

/*
    if (goal_.x != 0.0 || goal_.y != 0.0 || goal_.z != 0.0)
    {
        goal_pub_.publish(goal_);
    }
*/

}
