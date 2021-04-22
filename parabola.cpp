#include <cassert>
#include <iostream>
#include <math.h>

#include <Eigen/Core>
using namespace Eigen;

#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>

using avar = autodiff::var;

double H = 0;
double K = 0;

int EPOCHS = 100;
int BATCH_SIZE = 5;
int RANGE = 50;

avar f(avar x)
{
   //return pow((x - H), 2.0) + K;
   return (5 * x) + 10;
}

avar rmse(const MatrixXvar& pred, const MatrixXvar& y)
{
    return sqrt((y - pred).array().square().mean());
}


template<typename T, int R, int C>
void calcGrads(avar& loss, Eigen::Matrix<T, R, C>& xs, Eigen::Matrix<T, R, C>& grads)
{
    assert(xs.rows() == grads.rows());
    assert(xs.cols() == grads.cols());

    for (size_t i = 0, size = xs.size(); i < size; i++)
    {
        T& x = xs.data()[i];
        T& g = grads.data()[i];

        auto [u] = autodiff::reverse::derivativesx(loss, wrt(x));
        g = u;

    }
}

template<typename T, int R, int C>
void updateWeights(Eigen::Matrix<T, R, C>& weights, Eigen::Matrix<T, R, C>& grads, double lr=0.001)
{
    auto g_T = grads.transpose(); 

    assert(weights.rows() == g_T.rows());
    assert(weights.cols() == g_T.cols());

    for (size_t i = 0, size = weights.size(); i < size; i++)
    {
        // weights.data()[i]
        T& w = weights.data()[i];
        T& g = g_T.data()[i];

        w *= -(g * lr);
    }
}

int main()
{
    std::cout << "H: " << H << ", K: " << K << std::endl;

    // 1 x 5 weights and 5 x 5 bias
    MatrixXd W = MatrixXd::Random(1, BATCH_SIZE) * 10;


    // 5 x 1 inpuq

    MatrixXvar tx(5, 1);
    tx << -2, -1, 0, 1, 2;

    MatrixXvar ty = tx.unaryExpr(&f);


    for (int e = 1; e <= EPOCHS; e++)
    {
        MatrixXvar x = MatrixXvar::Random(BATCH_SIZE, 1) * RANGE;
        MatrixXvar grads(BATCH_SIZE, 1);

        MatrixXvar w = W;

        MatrixXvar y = x.unaryExpr(&f);
        MatrixXvar y_hat = ((x * w)).rowwise().sum();
    
        avar loss = rmse(y_hat, y);

        calcGrads(loss, y_hat, grads);
        updateWeights(w, grads);

        // Doing test loss
        MatrixXvar ty_hat = ((tx * w)).rowwise().sum();
        avar test_loss = rmse(ty_hat, ty);

        std::cout << "epoch " << e << " --- loss: " << loss << " test loss: " << test_loss << std::endl;
        std::cout << "\tgrad mean: " << grads.mean() << std::endl;

        for (auto i = 0;  i < BATCH_SIZE; i++) {
            W(0, i) = (double)w(0, i);
            std::cout << (double) w(0, i) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

