#ifndef LINEAR_MODEL_H_
#define LINEAR_MODEL_H_

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <ros/ros.h>

enum SystemType{ NOT_SET, AUTONOMOUS, PURELY_DYNAMIC, DYNAMIC };

class LinearModel
{
    public:
        LinearModel();
        LinearModel(Eigen::MatrixXf& A, Eigen::MatrixXf& C); // autonomous system
        LinearModel(Eigen::MatrixXf& A, Eigen::MatrixXf& B, Eigen::MatrixXf& C); // purely dynamic system
        LinearModel(Eigen::MatrixXf& A, Eigen::MatrixXf& B, Eigen::MatrixXf& C, Eigen::MatrixXf& D);
        LinearModel operator=(LinearModel& model_to_copy);

        void setInitialState(Eigen::VectorXf& initial_state);
        bool updateState();
        bool updateState(Eigen::VectorXf input);
        bool computeOutput();
        bool computeOutput(Eigen::VectorXf input);

        void setMatrices(Eigen::MatrixXf& A, Eigen::MatrixXf& C);
        void setMatrices(Eigen::MatrixXf& A, Eigen::MatrixXf& B, Eigen::MatrixXf& C);
        void setMatrices(Eigen::MatrixXf& A, Eigen::MatrixXf& B, Eigen::MatrixXf& C, Eigen::MatrixXf& D);
        Eigen::MatrixXf getA();
        Eigen::MatrixXf getB();
        Eigen::MatrixXf getC();
        Eigen::MatrixXf getD();
        Eigen::VectorXf getState();
        Eigen::VectorXf getOutput();
        SystemType getType();


    private:

        Eigen::VectorXf state_;
        Eigen::VectorXf output_;

        SystemType type_;

        Eigen::MatrixXf A_; //state matrix
        Eigen::MatrixXf B_; //input matrix
        Eigen::MatrixXf C_; //output matrix
        Eigen::MatrixXf D_; //input-output matrix

};

#endif /* LINEAR_MODEL_H_ */
