#include<linear_model.h>

LinearModel::LinearModel()
{
    type_ = NOT_SET;
}

LinearModel::LinearModel(Eigen::MatrixXf& A, Eigen::MatrixXf& C): A_(A), C_(C)
{
    type_ = AUTONOMOUS;
}

LinearModel::LinearModel(Eigen::MatrixXf& A, Eigen::MatrixXf& B, Eigen::MatrixXf& C): A_(A), B_(B), C_(C)
{
    type_ = PURELY_DYNAMIC;
}

LinearModel::LinearModel(Eigen::MatrixXf& A, Eigen::MatrixXf& B, Eigen::MatrixXf& C, Eigen::MatrixXf& D): A_(A), B_(B), C_(C), D_(D)
{
    type_ = DYNAMIC;
}


LinearModel LinearModel::operator=(LinearModel& model_to_copy)
{
    this->type_ = model_to_copy.getType();
    this->A_ = model_to_copy.getA();
    this->B_ = model_to_copy.getB();
    this->C_ = model_to_copy.getC();
    this->D_ = model_to_copy.getD();
    this->state_ = model_to_copy.getState();
    this->output_ = model_to_copy.getOutput();
    
    return *this;
}


void LinearModel::setInitialState(Eigen::VectorXf& state)
{
    state_ = state;
}

bool LinearModel::updateState()
{

    if (A_.size() == 0)
    {
        ROS_ERROR("The State matrix (A) is required!");
        return false;
    } 
    else if (state_.size() == 0)
    {
        ROS_ERROR("The state is not initialized!");
        return false;
    }
    else
    {
        switch (type_)
        {
            case AUTONOMOUS:

                state_ = A_ * state_;
                return true;

            break;
        
            case PURELY_DYNAMIC:
            case DYNAMIC:

                ROS_ERROR("The system is not autonomous, an input is required!");
                return false;

            break;
        }
    }
    
}

bool LinearModel::updateState(Eigen::VectorXf input)
{

    if (A_.size() == 0)
    {
        ROS_ERROR("The State matrix (A) is required!");
        return false;
    }
    else if (state_.size() == 0)
    {
        ROS_ERROR("The state is not initialized!");
        return false;
    }
    else
    {
        switch (type_)
        {
            case AUTONOMOUS:

                ROS_WARN("The system is autonomous, the input is ignored");
                state_ = A_ * state_;

            break;
    
            case PURELY_DYNAMIC:
            case DYNAMIC:

                if(B_.size() == 0)
                {
                    ROS_ERROR("The input matrix (B) is required");
                    return false;
                }
                else
                {
                    state_ = A_ * state_ + B_ * input;
                    return true;
                }

            break;
        }
    }
    
}

bool LinearModel::computeOutput()
{

    if (C_.size() == 0)
    {
        ROS_ERROR("The output matrix (C) is required!");
        return false;
    } 
    else if (state_.size() == 0)
    {
        ROS_ERROR("The state is not initialized!");
        return false;
    }
    else
    {
        switch (type_)
        {
            case AUTONOMOUS:
            case PURELY_DYNAMIC:

                output_ = C_ * state_;
                return true;

            break;

            case DYNAMIC:
            
                ROS_ERROR("An input is required!");
                return false;
            
            break;
        }
    }
}

bool LinearModel::computeOutput(Eigen::VectorXf input)
{
    if (C_.size() == 0)
    {
        ROS_ERROR("The output matrix (C) is required!");
        return false;
    }
    else if (state_.size() == 0)
    {
        ROS_ERROR("The state is not initialized!");
        return false;
    }
    else
    {
        switch (type_)
        {
            case AUTONOMOUS:
            case PURELY_DYNAMIC:

                ROS_WARN("The system is autonomous or purely dinamic, the input is ignored");
                output_ = C_ * state_;
                return true;

            break;

            case DYNAMIC:

                if (D_.size() == 0)
                {
                    ROS_ERROR("The input-output matrix (D) is required");
                    return false;
                }
                else
                {
                    output_ = C_ * state_ + D_ * input;
                    return true;
                }

            break;
        }
    }
}

void LinearModel::setMatrices(Eigen::MatrixXf& A, Eigen::MatrixXf& C)
{
    A_ = A;
    C_ = C;
    type_ = AUTONOMOUS;
}

void LinearModel::setMatrices(Eigen::MatrixXf& A, Eigen::MatrixXf& B, Eigen::MatrixXf& C)
{
    A_ = A;
    B_ = B;
    C_ = C;
    type_ = PURELY_DYNAMIC;
}
void LinearModel::setMatrices(Eigen::MatrixXf& A, Eigen::MatrixXf& B, Eigen::MatrixXf& C, Eigen::MatrixXf& D)
{
    A_ = A;
    B_ = B;
    C_ = C;
    D_ = D;
    type_ = DYNAMIC;
}

Eigen::MatrixXf LinearModel::getA()
{
    return A_;
}

Eigen::MatrixXf LinearModel::getB()
{
    return B_;
}

Eigen::MatrixXf LinearModel::getC()
{
    return C_;
}

Eigen::MatrixXf LinearModel::getD()
{
    return D_;
}

Eigen::VectorXf LinearModel::getState()
{
    return state_;
}

Eigen::VectorXf LinearModel::getOutput()
{
    return output_;
}      

SystemType LinearModel::getType()
{
    return type_;
}
