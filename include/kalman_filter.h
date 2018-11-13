#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include <eigen3/Eigen/Dense>
#include <linear_model.h>
#include <ros/ros.h>
#include <iostream>


class KalmanFilter
{
    public:
        KalmanFilter();
        KalmanFilter(Eigen::VectorXf &initial_state, Eigen::MatrixXf &initial_covariance);
        KalmanFilter(Eigen::VectorXf &initial_state, Eigen::MatrixXf &initial_covariance, LinearModel &model);
        KalmanFilter(LinearModel &model);
        ~KalmanFilter();

        void setInitialStateAndCovariance(Eigen::VectorXf &initial_state, Eigen::MatrixXf &initial_covariance);
        Eigen::VectorXf getState();
        Eigen::MatrixXf getCovariance();
        void setModel(LinearModel &model);
        void setModelErrorCovariance(Eigen::MatrixXf &model_covariance);
        LinearModel getCurrentModel();
        bool estimate(Eigen::VectorXf &observations, Eigen::MatrixXf &observation_covariance); // autonomous system
        bool estimate(Eigen::VectorXf &observations, Eigen::MatrixXf &observation_covariance, Eigen::VectorXf &input);
        
    private:

        Eigen::VectorXf state_; //state
        Eigen::MatrixXf covariance_; //covariance matrix
       
        LinearModel model_;
        Eigen::MatrixXf model_error_covariance_;

        bool prediction();
        bool prediction(Eigen::VectorXf &input);
        bool update(Eigen::VectorXf &observations, Eigen::MatrixXf &observation_covariance);
};

#endif /*KALMAN_FILTER_H_ */
