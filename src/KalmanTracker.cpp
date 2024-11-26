#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

class KalmanFilter {
public:
    KalmanFilter(const Eigen::MatrixXd& F,
                 const Eigen::MatrixXd& H,
                 const Eigen::MatrixXd& Q,
                 const Eigen::MatrixXd& R,
                 const Eigen::MatrixXd& P,
                 const Eigen::VectorXd& x)
        : F_(F), H_(H), Q_(Q), R_(R), P_(P), x_(x) {}

    void predict() {
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void update(const Eigen::VectorXd& z) {
        Eigen::VectorXd y = z - H_ * x_;
        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
        Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
        x_ = x_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P_;
    }

    Eigen::VectorXd getState() const {
        return x_;
    }

private:
    Eigen::MatrixXd F_;
    Eigen::MatrixXd H_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd P_;
    Eigen::VectorXd x_;
};

// Binding code
PYBIND11_MODULE(kalman_cpp, m) {
    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<const Eigen::MatrixXd&,
                      const Eigen::MatrixXd&,
                      const Eigen::MatrixXd&,
                      const Eigen::MatrixXd&,
                      const Eigen::MatrixXd&,
                      const Eigen::VectorXd&>())
        .def("predict", &KalmanFilter::predict)
        .def("update", &KalmanFilter::update)
        .def("getState", &KalmanFilter::getState);
}
