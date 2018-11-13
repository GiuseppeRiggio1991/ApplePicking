#include <ros/ros.h>
#include <kalman_filter.h>
#include <std_msgs/Float64.h>
#include <iostream>

using namespace std;

float measure = 0.0;
bool new_msg = false;

void callback(const std_msgs::Float64::ConstPtr& msg)
{
    measure = msg->data;
    new_msg = true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "KF_Test");

    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("fake_data", 1, callback);

    Eigen::VectorXf initial_state(1);
    initial_state << 0.0;

    Eigen::MatrixXf initial_covariance(1, 1);
    initial_covariance << 10.0;

    Eigen::MatrixXf A(1, 1);
    A << 1.0;

    Eigen::MatrixXf C(1, 1);
    C << 1.0;

    Eigen::MatrixXf R(1, 1);
    R << 0.5;

    Eigen::MatrixXf Q(1, 1);
    Q << 0.5;

    LinearModel model(A, C);

    KalmanFilter* kf = new KalmanFilter(initial_state, initial_covariance, model);
    
    kf->setModelErrorCovariance(Q);

    ros::Rate loop_rate(1000);

    while (ros::ok())
    {
        new_msg = false;
        ros::spinOnce();
        if (new_msg)
        {
            Eigen::VectorXf meas(1);
            meas << measure;
            kf->estimate(meas, R);
            cout << "state: " << kf->getState() << " covariance: " << kf->getCovariance() << endl;
        }
        loop_rate.sleep();
    }

    delete kf;
}
