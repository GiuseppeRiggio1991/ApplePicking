#include <goal_manager.h>

GoalManager::GoalManager()
{
    timer_sub_ = nh_.createTimer(ros::Duration(0.01), &GoalManager::updateGoal, this);

    // Publishers
    goal_pub_ = nh_.advertise<geometry_msgs::Point>("/sawyer_planner/goal", 1);

    // Server
    apple_check_server_ = nh_.advertiseService("/sawyer_planner/apple_check", &GoalManager::appleCheck, this);

    // Clients
    pcl_client_ = nh_.serviceClient<hydra_utils::CloudService>("/left_arm/mvps/segmentation_module/spheres"); 

    state_size_ = 0;
}

GoalManager::~GoalManager()
{
}

bool GoalManager::appleCheck(sawyer_planner::AppleCheck::Request &req, sawyer_planner::AppleCheck::Response &res)
{
    float threshold = 0.1; // change with some parameter
    float distance = std::numeric_limits<double>::infinity();

    hydra_utils::CloudService cloud_srv;
    bool is_there = false;
    int tries = 0;
    while (tries < 100 && is_there == false)
    {
        if (pcl_client_.call(cloud_srv))
        {

            pcl::PointCloud<pcl::PointXYZRGBA> cloud;
            pcl::fromROSMsg(cloud_srv.response.cloud, cloud);


            for (int i = 0; i < cloud.size(); i++)
            {

                pcl::PointXYZRGBA& p = cloud.points[i];

                if (sqrt( pow(p.x - req.apple_pose.x, 2) + pow(p.y - req.apple_pose.y, 2) + pow(p.z - req.apple_pose.z, 2)) < distance)
                {
                    distance = sqrt( pow(p.x - req.apple_pose.x, 2) + pow(p.y - req.apple_pose.y, 2) + pow(p.z - req.apple_pose.z, 2));

                    if (distance < threshold){
                        is_there = true;
                        break;
                    }
                }
            }

        }

        tries++;
    }

    if (distance < threshold)
    {
        res.apple_is_there = true;
        return true;
    }
    else
    {

        distance = std::numeric_limits<double>::infinity();
        int index = apples_.size();

        for (int i = 0; i < apples_.size(); i += 3)
        {   
            
            if (sqrt( pow(req.apple_pose.x - apples_[i], 2) + pow(req.apple_pose.y - apples_[i+1], 2) + pow(req.apple_pose.z - apples_[i+2], 2)) < distance)
            {
                distance = sqrt( pow(req.apple_pose.x - apples_[i], 2) + pow(req.apple_pose.y - apples_[i+1], 2) + pow(req.apple_pose.z - apples_[i+2], 2));
                index = i;
            }
        }

        if (distance < threshold)
        {   
            Eigen::MatrixXf covariance = kf_.getCovariance();

            // remove from the state
            if (apples_.size() > 3)
            {
                if (index < apples_.size() - 3)
                {
                    apples_.segment(index, apples_.size() - 3 - index) = apples_.tail(apples_.size() - index - 3);

                    // remove row
                    covariance.block(index, 0, covariance.rows() - 3 - index, covariance.cols()) = covariance.block(index + 3, 0, covariance.rows() - index - 3, covariance.cols());

                    // remove column
                    covariance.block(0, index, covariance.rows(), covariance.cols() - 3 - index) = covariance.block(0, index + 3, covariance.rows(), covariance.cols() - index - 3);

                    // reshape
                    apples_.conservativeResize(apples_.size() - 3);
                    covariance.conservativeResize(covariance.rows() - 3, covariance.cols() - 3);
                }
            }
            else
            {
                apples_.resize(0);
                covariance.resize(0, 0);
            }

            

            state_size_ = apples_.size();
            Eigen::MatrixXf A = Eigen::MatrixXf::Identity(state_size_, state_size_);
            Eigen::MatrixXf C = A;
            LinearModel model(A, C); // it doesn't really matter to update C
            kf_.setModel(model);
            kf_.setInitialStateAndCovariance(apples_, covariance);

            ROS_INFO("Removed from the state");

        }
        else
        {
            ROS_INFO("The requested apple was not in the state!");
        }

        res.apple_is_there = false;
        return true;       
    }
}

void GoalManager::updateGoal(const ros::TimerEvent& event)
{
    hydra_utils::CloudService cloud_srv;
    if (pcl_client_.call(cloud_srv))
    {
        pcl::PointCloud<pcl::PointXYZRGBA> cloud;
        pcl::fromROSMsg(cloud_srv.response.cloud, cloud);

        std::cout << "apples_.size(): " << apples_.size() << std::endl;
        if (apples_.size() == 0)
        {
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
            float covariance = 0.5; // change with some parameter
            Eigen::VectorXf covariance_vector = covariance * Eigen::VectorXf::Ones(state_size_);
            Eigen::MatrixXf covariance_matrix = covariance_vector.asDiagonal();

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
                else
                {
                    // no match, it is a new apple
                    std::cout << "new apple" << std::endl;
                    new_apples.conservativeResize(new_apples.size()+3);

                    new_apples[new_apples.size()-3] = p.x;
                    new_apples[new_apples.size()-2] = p.y;
                    new_apples[new_apples.size()-1] = p.z;

                }
            }

            float obs_covariance = 0.5; // change with some parameter
            Eigen::VectorXf obs_covariance_vector = obs_covariance * Eigen::VectorXf::Ones(observations.size());
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
            float covariance = 0.5; // change with some parameter
            Eigen::VectorXf covariance_vector = state_covariance.diagonal();
            covariance_vector.conservativeResize(apples_.size());
            covariance_vector.block(state_size_, 0, new_apples.size(), 1) = covariance * Eigen::VectorXf::Ones(new_apples.size());
            Eigen::MatrixXf covariance_matrix = covariance_vector.asDiagonal();

            C.conservativeResize(observations.size(), apples_.size());
            C.block(0, state_size_, observations.size(), new_apples.size()) = Eigen::MatrixXf::Zero(observations.size(), new_apples.size());

            state_size_ = apples_.size();
            A = Eigen::MatrixXf::Identity(state_size_, state_size_);

            model.setMatrices(A, C); // it doesn't really matter to update C
            kf_.setModel(model);
            kf_.setInitialStateAndCovariance(apples_, covariance_matrix);
             
       }

        
        /*Eigen::MatrixXf covariance_matrix = kf_.getCovariance();
    
        std::cout << "N: " << apples_.size() << std::endl;

        for (int i = 0; i < apples_.size(); i = i + 3)
        {
            std::cout << i << " " << apples_[i] << " " << apples_[i+1] << " " << apples_[i+2]
                           << " " << covariance_matrix(i, i) << " " << covariance_matrix(i+1, i+1) << " " << covariance_matrix(i+2, i+2)  << std::endl; 
        }
        std::cout << std::endl;*/

    }


    Eigen::MatrixXf covariance_matrix = kf_.getCovariance();

    std::cout << "N: " << apples_.size() << std::endl;

    for (int i = 0; i < apples_.size(); i = i + 3)
    {
        std::cout << i << " " << apples_[i] << " " << apples_[i+1] << " " << apples_[i+2]
                           << " " << covariance_matrix(i, i) << " " << covariance_matrix(i+1, i+1) << " " << covariance_matrix(i+2, i+2)  << std::endl;
    }
    std::cout << std::endl;



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
