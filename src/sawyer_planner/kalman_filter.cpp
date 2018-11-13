#include <kalman_filter.h>

KalmanFilter::KalmanFilter()
{

}

KalmanFilter::KalmanFilter(Eigen::VectorXf &initial_state, Eigen::MatrixXf &initial_covariance): state_(initial_state), covariance_(initial_covariance)
{

}

KalmanFilter::KalmanFilter(Eigen::VectorXf &initial_state, Eigen::MatrixXf &initial_covariance, LinearModel &model): state_(initial_state), covariance_(initial_covariance), model_(model)
{
    model_.setInitialState(initial_state);
}

KalmanFilter::KalmanFilter(LinearModel &model): model_(model)
{
    
}

KalmanFilter::~KalmanFilter()
{

}

void KalmanFilter::setInitialStateAndCovariance(Eigen::VectorXf &initial_state, Eigen::MatrixXf &initial_covariance)
{
    state_ = initial_state;
    covariance_ = initial_covariance;

    if (model_.getType() != NOT_SET)
    {
        model_.setInitialState(initial_state);
    }
    else
    {
        ROS_ERROR("Model not set!");
    }
}

Eigen::VectorXf KalmanFilter::getState()
{
    return state_;
}

Eigen::MatrixXf KalmanFilter::getCovariance()
{
    return covariance_;
}

void KalmanFilter::setModel(LinearModel &model)
{
    model_ = model;
    if (state_.size() != 0)
    {
        model_.setInitialState(state_);
    }
}

void KalmanFilter::setModelErrorCovariance(Eigen::MatrixXf &model_covariance)
{
    model_error_covariance_ = model_covariance;
}

LinearModel KalmanFilter::getCurrentModel()
{
    return model_;
}

bool KalmanFilter::estimate(Eigen::VectorXf &observations, Eigen::MatrixXf &observation_covariance)
{
    if (prediction() )
        if ( update(observations, observation_covariance) ) 
            return true;

    return false;
}

bool KalmanFilter::estimate(Eigen::VectorXf &observations, Eigen::MatrixXf &observation_covariance, Eigen::VectorXf &input)
{
    if (prediction(input))
        if (update(observations, observation_covariance) )
            return true;

    return false;


}

bool KalmanFilter::prediction()
{
    if (model_.getType() != NOT_SET)
    {

        if (model_.updateState())
        {
            state_ = model_.getState();
            covariance_ = model_.getA() * covariance_ * model_.getA().transpose();
            if (model_error_covariance_.size() != 0)
                covariance_ += model_error_covariance_;
            return true;
        }
                 
        return false;
    }

    return false;
}

bool KalmanFilter::prediction(Eigen::VectorXf &input)
{

    if (model_.getType() != NOT_SET)
    {
        if (model_.updateState(input))
        {
            state_ = model_.getState();
            covariance_ = model_.getA() * covariance_ * model_.getA().transpose();
            if (model_error_covariance_.size() != 0)
                covariance_ += model_error_covariance_;
            return true;
        }
        
        return false;    
    }

    return false;
}

bool KalmanFilter::update(Eigen::VectorXf &observations, Eigen::MatrixXf &observation_covariance)
{

    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(covariance_.rows(), covariance_.cols());
    Eigen::MatrixXf C = model_.getC();

    std::cout << observations.size() << " " << C.size() << " " << state_.size() << std::endl;

    if (C.size() != 0)
    {
        Eigen::VectorXf innovation = observations - C * state_;
        Eigen::MatrixXf innovation_cov = C * covariance_ * C.transpose() + observation_covariance;
        Eigen::MatrixXf kalman_gain = covariance_ * C.transpose() * innovation_cov.inverse();
        state_ = state_ + kalman_gain * innovation;
        covariance_ = (I - kalman_gain * C) * covariance_;
        model_.setInitialState(state_);
    }

    return true;
}

