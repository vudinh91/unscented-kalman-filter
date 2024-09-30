#pragma once

#include <math.h>

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "utilities.h"

/**
 * @brief Class to estimate the state of objects from sensor measurements.
 *
 */
template<typename Real, size_t StateDim, size_t MeasDim>
class UnscentedKalmanFilter final
{
 public:

  /// @brief Type aliases for various vectors and matrices
  using state_vec = Eigen::Matrix<Real, StateDim, 1>;
  using cov_mat = Eigen::Matrix<Real, StateDim, StateDim>;
  using f_mat = Eigen::Matrix<Real, StateDim, StateDim>;
  using ft_mat = Eigen::Matrix<Real, StateDim, StateDim>;
  using q_mat = Eigen::Matrix<Real, StateDim, StateDim>;

  using meas_vec = Eigen::Matrix<Real, MeasDim, 1>;
  using r_mat = Eigen::Matrix<Real, MeasDim, MeasDim>;

  using k_mat = Eigen::Matrix<Real, StateDim, MeasDim>;
  using kt_mat = Eigen::Matrix<Real, MeasDim, StateDim>;

  using sig_mat = Eigen::Matrix<Real, StateDim, StateDim * 2 + 1>;
  using xx_mat = Eigen::Matrix<Real, StateDim, StateDim * 2 + 1>;
  using yy_mat = Eigen::Matrix<Real, MeasDim, StateDim * 2 + 1>;
  
  UnscentedKalmanFilter();

  ~UnscentedKalmanFilter() = default;

  /**
   * @brief Initialize the filter
   */
  void initialize(const state_vec& sensor_state, 
                  const meas_vec& measurement_vec);

  /**
   * @brief Propagate the state and covariance
   * 
   */
  void propagateStep(const Real dt);

  /**
   * @brief Update the state and covariance
   * 
   */
  void updateStep(const meas_vec& measurement_vec, 
                  const meas_vec& measurement_noise_vec, 
                  const state_vec& sensor_state);

  /**
   * @brief Update the estimated state of all tracks based on the sensor state and sensor measurement.
   *
   */
  void update(const meas_vec& measurement_vec, 
              const meas_vec& measurement_noise_vec, 
              const state_vec& sensor_state,
              const Real dt);


  /**
   * @brief Set the State transition matrix
   */
  void setF(const Real dt)
  {
    // Constant velocity motion
    F_(0,2) = dt;
    F_(1,3) = dt;
  }

  /**
   * @brief Get the State transition matrix
   */
  f_mat getF() const { return F_; }

  /**
   * @brief Set the measurement noise matrix
   */
  void setR(const meas_vec& measurement_noise_vec)
  {
    // Use the noise from measurement to initialize R
    R_(0,0) = measurement_noise_vec(0,0) * measurement_noise_vec(0,0);
    R_(1,1) = measurement_noise_vec(1,0) * measurement_noise_vec(1,0);
  }

  /**
   * @brief Get the measurement noise matrix
   */
  r_mat getR() const { return R_; }

  /**
   * @brief Set the measurement noise matrix
   */
  void setQ(const Real dt, const Real sigma)
  {
    // Process noise for constant velocity model
    Q_(0,0) = (dt * dt * dt * dt) / 4 * sigma * sigma;
    Q_(0,2) = (dt * dt * dt) / 2 * sigma * sigma;

    Q_(1,1) = (dt * dt * dt * dt) / 4 * sigma * sigma;
    Q_(1,3) = (dt * dt * dt) / 2 * sigma * sigma;

    Q_(2,0) = (dt * dt * dt) / 2 * sigma * sigma;
    Q_(2,2) = (dt * dt) * sigma * sigma;

    Q_(3,1) = (dt * dt * dt) / 2 * sigma * sigma;
    Q_(3,3) = (dt * dt) * sigma * sigma;
  }

  /**
   * @brief Get the measurement noise matrix
   */
  q_mat getQ() const { return Q_; }

  /**
   * @brief Get the predicted state
   */
  state_vec getPredictedState() const { return x_minus_; }

  /**
   * @brief Get the predicted covariance
   */
  cov_mat getPredictedCov() const { return P_minus_; }

  /**
   * @brief Get the updated state
   */
  state_vec getUpdatedState() const { return x_plus_; }

  /**
   * @brief Get the updated covariance
   */
  cov_mat getUpdatedCov() { return P_plus_; }

  /**
   * @brief Get the state dimension
   */
  size_t getStateDim() const { return state_dim_; }

  /**
   * @brief Get the measurement dimension
   */
  size_t getMeasDim() const { return meas_dim_; }

  /**
   * @brief Get the number of sigma points
   */
  size_t getNumSigmaPoints() const { return num_sigma_points_; }

  /**
   * @brief Get mean weight 0
   */
  Real getW0m() const { return w0m_; }

  /**
   * @brief Get mean weight ith
   */
  Real getWim() const { return wim_; }

  /**
   * @brief Get covariance weight 0
   */
  Real getW0c() const { return w0c_; }

  /**
   * @brief Get covariance weight ith
   */
  Real getWic() const { return wic_; }

 private:

  /**
   * @brief Compute UT weights
   */
  void computeUTWeights();

  /**
   * @brief Compute sigma points
   */
  void computeSigmaPoints();

  /**
   * @brief Propagate the sigma points
   */
  void propagateSigmaPoints();

  /**
   * @brief Calculate predicted mean
   */
  void calculatePredictedMean();

  /**
   * @brief Calculate predicted cov
   */
  void calculatePredictedCov();

  /**
   * @brief Measurement mapping function
   */
  meas_vec measMappingFunc(const state_vec& predicted_state, const state_vec& sensor_state);

  /**
   * @brief Calculate mean observation
   */
  meas_vec calculateMeanObservation(const state_vec& sensor_state);

  /**
   * @brief Calculate Pyy (innovation covariance)
   */
  r_mat calculatePyy(const meas_vec& y_minus);

  /**
   * @brief Calculate Pxy (cross covariance)
   */
  k_mat calculatePxy(const meas_vec& y_minus);

  /**
   * @brief Calculate Kalman gain
   */
  k_mat calculateKalmanGain(const k_mat& Pxy, const r_mat& Pyy);

  /// @brief Flags to indicate whether init, predict, update has been performed
  bool init_flag_ = false;
  bool predict_flag_ = false;
  bool update_flag_ = false;

  /// @brief State and measurement dimensions
  size_t state_dim_;
  size_t meas_dim_;

  /// @brief Filter predicted state and covariance
  state_vec x_minus_;
  cov_mat P_minus_;

  /// @brief Filter updated state and covariance
  state_vec x_plus_;
  cov_mat P_plus_;

  /// @brief State transition matrix
  f_mat F_;

  /// @brief Process noise
  q_mat Q_;
  Real process_noise_sig_{ 0.05 };

  /// @brief Measurement noise
  r_mat R_;

  /// @brief Sigma points, used for unscented transform
  sig_mat sigma_points_;
  xx_mat xx_predicted_;
  yy_mat yy_predicted_;

  /// @brief Unscented transform params
  Real alph_;
  Real beta_;
  Real kap_;
  Real lamb_;
  Real gam_;
  Real sqrt_gam_;
  Real w0m_;
  Real wim_;
  Real w0c_;
  Real wic_;
  size_t num_sigma_points_;

};


template<typename Real, size_t StateDim, size_t MeasDim>
UnscentedKalmanFilter<Real, StateDim, MeasDim>::UnscentedKalmanFilter()
{
  // Define the state and measurement dimensions
  state_dim_ = StateDim;
  meas_dim_ = MeasDim;

  // Init State vector and Covariance matrices
  x_minus_.setZero();
  P_minus_.setIdentity();
  x_plus_.setZero();
  P_plus_.setIdentity();

  // Init State transition matrix F
  F_.setIdentity();

  // Init Process noise matrix Q
  Q_.setZero();

  // Init Measurement noise matrix R
  R_.setIdentity();

  // Initialize parameters for Unscented Transform
  alph_ = 1.0E-2;
  beta_ = 2.0;
  kap_  = 3.0 - static_cast<Real>(state_dim_);

  // Init sigma points
  num_sigma_points_ = 2 * state_dim_ + 1;
  sigma_points_.setZero();
  xx_predicted_.setZero();
  yy_predicted_.setZero();
};

/**
 * @brief Initialize the filter
 */
template<typename Real, size_t StateDim, size_t MeasDim>
void UnscentedKalmanFilter<Real, StateDim, MeasDim>::initialize(const state_vec& sensor_state, 
                                                                const meas_vec& measurement_vec)
{
  
  // Initialize the state using first measurement
  x_minus_(0,0) = measurement_vec(0,0) * cos(measurement_vec(1,0)) + sensor_state(0,0);
  x_minus_(1,0) = measurement_vec(0,0) * sin(measurement_vec(1,0)) + sensor_state(1,0);

  x_plus_ = x_minus_;

  // Position uncertainty
  const Real pos_var = 25.0; // 5m stdev

  // Velocity uncertainty
  const Real vel_var = 9.0; // 3m/s stdev

  // Initialize the covariance to some large value
  for (size_t i = 0; i < state_dim_; i++) {
    for (size_t j = 0; j < state_dim_; j++) {
      if (i==j) {

        if (i < state_dim_ / 2) {
          P_minus_(i,j) = pos_var;
          P_plus_(i,j) = pos_var;
        } else {
          P_minus_(i,j) = vel_var;
          P_plus_(i,j) = vel_var;
        }
        
      }
    }
  }

  // Compute weights for Unscented Transform
  computeUTWeights();

  // Set init flag to true
  init_flag_ = true;
}

/**
 * @brief Propagate the state and covariance
 * 
 */
template<typename Real, size_t StateDim, size_t MeasDim>
void UnscentedKalmanFilter<Real, StateDim, MeasDim>::propagateStep(const Real dt)
{
  // Ensure the filter has been initialized 
  assert(init_flag_);

  // If update hasn't been performed, set the a posteriori state and covariance
  // to the a priori state and covariance
  if (!update_flag_) {
    x_plus_ = x_minus_;
    P_plus_ = P_minus_;
  }

  // Construct state transition matrix F
  setF(dt);
  
  // Set Q based on dt and sigma
  setQ(dt, process_noise_sig_);

  // Generate the sigma points using P_plus_
  computeSigmaPoints();

  // Propagate the sigma points through F
  propagateSigmaPoints();

  // Calculate predicted mean
  calculatePredictedMean();

  // Calculate predicted covariance
  calculatePredictedCov();

  // Set the flags
  predict_flag_ = true;
  update_flag_ = false;

}

/**
 * @brief Update the state and covariance
 * 
 */
template<typename Real, size_t StateDim, size_t MeasDim>
void UnscentedKalmanFilter<Real, StateDim, MeasDim>::updateStep(const meas_vec& measurement_vec, 
                                                                const meas_vec& measurement_noise_vec, 
                                                                const state_vec& sensor_state)
{
  // Only perform update if state and covariance have been propagated
  assert(predict_flag_);

  // Make sure we update the measurement noise (in case it changes)
  setR(measurement_noise_vec);

  // Map the predicted state to measurement space
  // and calculate the mean observation
  const meas_vec y_minus = calculateMeanObservation(sensor_state);

  // Calculate Pyy
  const r_mat Pyy = calculatePyy(y_minus);

  // Calculate Pxy
  const k_mat Pxy = calculatePxy(y_minus);

  // Calculate Kalman gain
  //Kalman gain K;
  const k_mat K = calculateKalmanGain(Pxy,Pyy);

  // Calculate measurement residual
  meas_vec meas_resid = measurement_vec - y_minus;

  // Handle pi wrapping
  handle_pi_wrap(meas_resid(1,0));

  // Update the state
  x_plus_ = x_minus_ + K * (meas_resid);

  // Update the covariance
  P_plus_ = P_minus_ - K * Pyy * K.transpose();

  // Set the flags
  update_flag_ = true;
  predict_flag_ = false;

}

/**
 * @brief Update the estimated state of all tracks based on the sensor state and sensor measurement.
 * 
 */
template<typename Real, size_t StateDim, size_t MeasDim>
void UnscentedKalmanFilter<Real, StateDim, MeasDim>::update(const meas_vec& measurement_vec, 
                                                            const meas_vec& measurement_noise_vec, 
                                                            const state_vec& sensor_state,
                                                            const Real dt)
{
  
  // Handle pi wrapping
  handle_pi_wrap(measurement_vec(1,0));
  
  // Initialize
  if (!init_flag_) {

    // Init state and covariance using measurement
    initialize(sensor_state, measurement_vec);

    return;
  }

  ////// Predict/Propagate step //////

  // Propagate
  propagateStep(dt);

  //////////////////////////

  ////// Update step //////

  // Update the predicted state with the measurement
  updateStep(measurement_vec, measurement_noise_vec, sensor_state);

  //////////////////////////

}

/**
 * @brief Compute UT weights
 */
template<typename Real, size_t StateDim, size_t MeasDim>
void UnscentedKalmanFilter<Real, StateDim, MeasDim>::computeUTWeights()
{
  // Compute params
  lamb_ = alph_ * alph_ * (static_cast<Real>(state_dim_) + kap_) - static_cast<Real>(state_dim_);

  gam_ = static_cast<Real>(state_dim_) + lamb_;
  sqrt_gam_ = sqrt(gam_);

  // Compute weights for the mean
  w0m_ = lamb_ / gam_;
  wim_ = 1.0 / (2.0 * gam_);

  // Compute weights for the covariance
  w0c_ = w0m_ + (1.0 - alph_*alph_ + beta_);
  wic_ = wim_;
}

/**
 * @brief Compute sigma points
 */
template<typename Real, size_t StateDim, size_t MeasDim>
void UnscentedKalmanFilter<Real, StateDim, MeasDim>::computeSigmaPoints()
{
  cov_mat PQ = P_plus_ + Q_;
  cov_mat P_sqrt = PQ.llt().matrixL();
      
  for (size_t c = 0; c <= state_dim_; c++) {

    if (c==0) {

      sigma_points_.col(c) = x_plus_;

    } else {

      sigma_points_.col(c) = x_plus_ + sqrt_gam_ * P_sqrt.col(c-1);
      sigma_points_.col(c + state_dim_) = x_plus_ - sqrt_gam_ * P_sqrt.col(c-1);

    }
  }
}

/**
 * @brief Propagate the sigma points
 */
template<typename Real, size_t StateDim, size_t MeasDim>
void UnscentedKalmanFilter<Real, StateDim, MeasDim>::propagateSigmaPoints()
{
  // Propagate each sigma points using state transition matrix
  for (size_t c = 0; c < num_sigma_points_; c++) {
      xx_predicted_.col(c) = F_ * sigma_points_.col(c);
  }
}

/**
 * @brief Calculate predicted mean
 */
template<typename Real, size_t StateDim, size_t MeasDim>
void UnscentedKalmanFilter<Real, StateDim, MeasDim>::calculatePredictedMean()
{
  // Allocate
  state_vec x_minus_tmp = state_vec::Zero();

  Real mean_weight;

  // Multiply each sigma points with the weight and sum them up
  for (size_t c = 0; c < num_sigma_points_; c++) {

    mean_weight = c == 0 ? w0m_ : wim_;

    x_minus_tmp += mean_weight * xx_predicted_.col(c);
  }

  // Set the predicted state
  x_minus_ = x_minus_tmp;
}

/**
 * @brief Calculate predicted mean
 */
template<typename Real, size_t StateDim, size_t MeasDim>
void UnscentedKalmanFilter<Real, StateDim, MeasDim>::calculatePredictedCov()
{
  // Allocate
  cov_mat P_minus_tmp = cov_mat::Zero();

  Real cov_weight;

  state_vec xx_delta = state_vec::Zero();

  // Loop through each sigma points and compute the covariance
  for (size_t c = 0; c < num_sigma_points_; c++) {

    cov_weight = c == 0 ? w0c_ : wic_;

    xx_delta = xx_predicted_.col(c) - x_minus_;

    P_minus_tmp += cov_weight * xx_delta * xx_delta.transpose();
  }

  // Add the Q and set the predicted covariance
  P_minus_ = P_minus_tmp + Q_;
}

/**
 * @brief Measurement mapping function
 */
template<typename Real, size_t StateDim, size_t MeasDim>
typename UnscentedKalmanFilter<Real, StateDim, MeasDim>::meas_vec UnscentedKalmanFilter<Real, StateDim, MeasDim>::measMappingFunc(const state_vec& predicted_state, 
                                                             const state_vec& sensor_state)
{
  // Map the state back to measurement
  // For this particular problem, measurement vector is [range,angle] where:
  // range = sqrt(x^2 + y^2)
  // angle = atan2(y,x)
  meas_vec predicted_meas = meas_vec::Zero();
  
  // Compute LOS vector between sensor and object
  meas_vec los_sensor2obj = meas_vec::Zero();
  los_sensor2obj(0,0) = predicted_state(0,0) - sensor_state(0,0);
  los_sensor2obj(1,0) = predicted_state(1,0) - sensor_state(1,0);
  
  // range
  predicted_meas(0,0) = std::sqrt(los_sensor2obj(0,0)*los_sensor2obj(0,0) + los_sensor2obj(1,0)*los_sensor2obj(1,0));

  // angle
  predicted_meas(1,0) = std::atan2(los_sensor2obj(1,0), los_sensor2obj(0,0));

  // Handle pi wrapping
  handle_pi_wrap(predicted_meas(1,0));

  return predicted_meas;
}

/**
 * @brief Calculate mean observation
 */
template<typename Real, size_t StateDim, size_t MeasDim>
typename UnscentedKalmanFilter<Real, StateDim, MeasDim>::meas_vec UnscentedKalmanFilter<Real, StateDim, MeasDim>::calculateMeanObservation(const state_vec& sensor_state)
{
  // Map the predicted state back to measurement space
  // and calculate the mean observation
  meas_vec y_minus = meas_vec::Zero();

  Real mean_weight;
  
  for (size_t c = 0; c < num_sigma_points_; c++) {

    mean_weight = c == 0 ? w0m_ : wim_;

    yy_predicted_.col(c) = measMappingFunc(xx_predicted_.col(c),sensor_state);

    y_minus += mean_weight * yy_predicted_.col(c);
    
  }

  return y_minus;
}

/**
 * @brief Calculate Pyy (innovation covariance)
 */
template<typename Real, size_t StateDim, size_t MeasDim>
typename UnscentedKalmanFilter<Real, StateDim, MeasDim>::r_mat UnscentedKalmanFilter<Real, StateDim, MeasDim>::calculatePyy(const meas_vec& y_minus)
{
  // Allocate
  r_mat Pyy = r_mat::Zero();

  Real cov_weight;

  meas_vec delta_yy = meas_vec::Zero();

  for (size_t c = 0; c < num_sigma_points_; c++) {
    
    cov_weight = c == 0 ? w0c_ : wic_;
    
    delta_yy = yy_predicted_.col(c) - y_minus;

    Pyy += cov_weight * delta_yy * delta_yy.transpose();
  }

  // Add measurement noise and return
  return Pyy + R_;
}

/**
 * @brief Calculate Pxy (cross covariance)
 */
template<typename Real, size_t StateDim, size_t MeasDim>
typename UnscentedKalmanFilter<Real, StateDim, MeasDim>::k_mat UnscentedKalmanFilter<Real, StateDim, MeasDim>::calculatePxy(const meas_vec& y_minus)
{
  // Allocate
  k_mat Pxy = k_mat::Zero();

  Real cov_weight;

  state_vec delta_xx = state_vec::Zero();
  meas_vec delta_yy = meas_vec::Zero();

  for (size_t c = 0; c < num_sigma_points_; c++) {
    
    cov_weight = c == 0 ? w0c_ : wic_;

    delta_xx = xx_predicted_.col(c) - x_minus_;
    delta_yy = yy_predicted_.col(c) - y_minus;

    Pxy += cov_weight * delta_xx * delta_yy.transpose();

  }

  return Pxy;

}

/**
 * @brief Calculate Kalman gain
 */
template<typename Real, size_t StateDim, size_t MeasDim>
typename UnscentedKalmanFilter<Real, StateDim, MeasDim>::k_mat UnscentedKalmanFilter<Real, StateDim, MeasDim>::calculateKalmanGain(const k_mat& Pxy, const r_mat& Pyy)
{
  return Pxy * Pyy.inverse();
}
