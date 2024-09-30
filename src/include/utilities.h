#include <math.h>
#include <Eigen/Dense>

void handle_pi_wrap(double& angle_rad)
{
  if (angle_rad > M_PI) {
    angle_rad -= 2*M_PI;
  }

  if (angle_rad < -M_PI) {
    angle_rad += 2*M_PI;
  }
}

// /**
//  * @brief Create DCM from euler angle vector (ypr) in yaw/pitch/roll sequence (3-2-1)
//  */
// Eigen::Matrix3d angle2dcm(const Eigen::Vector3d& ypr_eul_rad)
// {
//   // Yaw
//   const double cpsi = std::cos(ypr_eul_rad(0,0));
//   const double spsi = std::sin(ypr_eul_rad(0,0));

//   // Pitch
//   const double ctheta = std::cos(ypr_eul_rad(1,0));
//   const double stheta = std::sin(ypr_eul_rad(1,0));

//   // Roll
//   const double cphi = std::cos(ypr_eul_rad(2,0));
//   const double sphi = std::sin(ypr_eul_rad(2,0));

//   Eigen::Matrix3d dcm{{                 ctheta*cpsi,                  ctheta*spsi,     -stheta},
//                        {sphi*stheta*cpsi - cphi*spsi, sphi*stheta*spsi + cphi*cpsi, sphi*ctheta},
//                        {cphi*stheta*cpsi + sphi*spsi, cphi*stheta*spsi - sphi*cpsi, cphi*ctheta}};

//   return dcm;

// }

// const double sgn(double val) {
//     return static_cast<double>((0.0 < val) - (val < 0.0));
// }

// /**
//  * @brief Compute euler angle (ypr) given the DCM
//  */
// Eigen::Vector3d dcm2angle(const Eigen::Matrix3d& dcm)
// {
//   const double r11 = dcm(0,0);
//   const double r12 = dcm(0,1);
//   double r13 = dcm(0,2);
//   const double r23 = dcm(1,2);
//   const double r33 = dcm(2,2);

//   if (r13 >= 1.0) {
//     r13 = 1.0;
//     printf("[WARNING]: Gimbal lock!");
//   } else if (r13 <= -1.0) {
//     r13 = -1.0;
//     printf("[WARNING]: Gimbal lock!");
//   }

//   // Pitch
//   const double theta = std::asin(-r13);

//   const double sign = sgn(std::cos(theta));

//   const double psi = std::atan2(sign * r12, sign * r11);

//   const double phi = std::atan2(sign * r23, sign * r33);

//   const Eigen::Vector3d ypr_eul_rad {psi, theta, phi};

//   return ypr_eul_rad;
// }