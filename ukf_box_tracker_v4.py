import math
from re import L
import numpy as np
from scipy.linalg import block_diag, cholesky, sqrtm
from parameter_ukf import *
import copy
from numpy import pi
from math import *

'''
v004 : normalize angle 추가
'''
# --------------------------------------------------------------------------
# Lidar_Tracking : (5, 1) Input
# --------------------------------------------------------------------------
# KalmanBoxTracker
# LIDAR_TRACKING_STATE_NUMBER = 18
# LIDAR_TRACKING_TRACK_NUMBER = 32
# ----------------------------------
# LIDAR_TRACKING_REL_POS_X = 0
# LIDAR_TRACKING_REL_POS_Y = 1
# LIDAR_TRACKING_REL_VEL = 2
# LIDAR_TRACKING_REL_VEL_X = 3
# LIDAR_TRACKING_REL_VEL_Y = 4
# LIDAR_TRACKING_REL_ACC_X = 5
# LIDAR_TRACKING_REL_ACC_Y = 6
# LIDAR_TRACKING_REL_ACC = 7
# LIDAR_TRACKING_YAW = 8
# LIDAR_TRACKING_YAW_RATE = 9
# LIDAR_TRACKING_WIDTH = 10
# LIDAR_TRACKING_HEIGHT = 11
# LIDAR_TRACKING_LENGTH = 12
# LIDAR_TRACKING_LIFE_TIME = 13
# LIDAR_TRACKING_CLASS = 14
# LIDAR_TRACKING_SCORE = 15
# LIDAR_TRACKING_UPDATE_TIME = 16
# LIDAR_TRACKING_ID = 17


# Unscented Kalman box Tracker

class UnscentedKalman_box_tracker():
    def __init__(self, tracking, P, state_dim, measurement_dim, process_noise, measurement_noise, kappa = -2, alpha = 0.0025, beta = 2):

        self.state      = tracking.squeeze()
        self.pred_state = tracking.squeeze()
        self.lambda_    = alpha**2*(state_dim + kappa) - state_dim

        self.ukf = UnscentedKalmanFilter(state_dim = state_dim                , measurement_dim = measurement_dim   , process_noise = process_noise, 
                                         measurement_noise = measurement_noise, mean = tracking[LIDAR_UKF_PARAMETER], cov = P                      ,
                                         lambda_ = self.lambda_               , alpha = alpha                       , beta = beta                   )

    def getPredState(self):

        return self.pred_state
    
    def getFinalState(self):
        
        return self.state
    
    def predict(self):
        self.state[LIDAR_TRACKING_UPDATE_TIME] += 1
        self.state[  LIDAR_TRACKING_LIFE_TIME] += 1

        dt = self.state[LIDAR_TRACKING_UPDATE_TIME] * SAMPLE_TIME

        self.ukf.predict(dt)
        self.pred_state[LIDAR_UKF_PARAMETER], self.cov = self.ukf.getPredState()

        
    
    def correct(self, measurement):
        self.state[LIDAR_TRACKING_UPDATE_TIME] = 0
        self.ukf.correct(measurement)
        tmp_state = self.state.copy()
        
        
        self.state[LIDAR_UKF_PARAMETER], self.cov       = self.ukf.getFinalState()
        self.pred_state[LIDAR_UKF_TRACKING_FIX_PARAM]   = measurement[LIDAR_UKF_MEASUREMENT_FIX_PARAM]
        self.state[LIDAR_UKF_TRACKING_FIX_PARAM]        = measurement[LIDAR_UKF_MEASUREMENT_FIX_PARAM]

        # self.state[LIDAR_UKF_VELOCITY_PARAM]            = self.update_velocity(tmp_state)
        # self.state[LIDAR_UKF_ACCELERATION_PARAM]        = self.update_acceleration(tmp_state)

        self.state[LIDAR_TRACKING_REL_VEL_X]            = self.state[LIDAR_TRACKING_REL_VEL] * cos(self.state[LIDAR_TRACKING_YAW])
        self.state[LIDAR_TRACKING_REL_VEL_Y]            = self.state[LIDAR_TRACKING_REL_VEL] * sin(self.state[LIDAR_TRACKING_YAW])

    def update_velocity(self, tmp_state):
        dx = self.state[LIDAR_TRACKING_REL_POS_X] - tmp_state[LIDAR_TRACKING_REL_POS_X]
        dy = self.state[LIDAR_TRACKING_REL_POS_Y] - tmp_state[LIDAR_TRACKING_REL_POS_Y]
        dt = SAMPLE_TIME * 2 

        vx = dx/dt
        vy = dy/dt
        v  = np.sqrt(vx**2 + vy**2)

        return [v, vx, vy]
    
    def update_acceleration(self, tmp_state):
        dvx = self.state[LIDAR_TRACKING_REL_VEL_X] - tmp_state[LIDAR_TRACKING_REL_VEL_X]
        dvy = self.state[LIDAR_TRACKING_REL_VEL_Y] - tmp_state[LIDAR_TRACKING_REL_VEL_Y]
        dt  = SAMPLE_TIME * 2 

        ax  = dvx/dt
        ay  = dvy/dt
        a   = np.sqrt(ax**2 + ay**2)

        return [a, ax, ay]


def ukf_track_f(xhat, dt):
    xp   = np.zeros((5, 1))
    p_x  = xhat[0]
    p_y  = xhat[1]
    v    = xhat[2]
    yaw  = xhat[3]
    yawd = xhat[4]
    # avoid division by zero

    if abs(yawd) > 0.001:
        px_p = p_x+v/yawd*(math.sin(yaw+yawd*dt)-math.sin(yaw))
        py_p = p_y + v/yawd*(math.cos(yaw) - math.cos(yaw+yawd*dt))
    else:
        px_p = p_x+v*dt*math.cos(yaw)
        py_p = p_y+v*dt*math.sin(yaw)

    v_p    = v
    yaw_p  = yaw+yawd*dt
    yawd_p = yawd

    xp[0]  = px_p
    xp[1]  = py_p
    xp[2]  = v_p
    xp[3]  = yaw_p
    xp[4]  = yawd_p

    return xp.squeeze()

def ukf_track_h(xhat):
    H = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0]])

    y = H@xhat
    return y



class UnscentedKalmanFilter():
    def __init__(self, state_dim, measurement_dim, process_noise, measurement_noise, mean, cov, lambda_ = -2.0, alpha = 0.0025, beta = 2):
        self.state_dim       = state_dim
        self.measurement_dim = measurement_dim

        self.mean            = mean.squeeze()
        self.cov             = cov

        self.Q               = process_noise      
        self.R               = measurement_noise

        self.lambda_         = lambda_

        self.mean_pred       = mean.squeeze()
        self.cov_pred        = cov

        self.z_mean          = mean.squeeze()
        self.S               = cov

        self.kalmanGain      = None

        self.Wm, self.Wc     = self.computeWeights(lambda_, alpha, beta)

        self.normalize_heading()    # 상태가 수정될 때마다 Heading 방향 수정

    def getPredState(self):
        return self.mean_pred, self.cov_pred
    
    def getFinalState(self):
        return self.mean, self.cov
    
    def getKalmaGain(self):
        return self.kalmanGain
    
    def getMeasureState(self):
        return self.z_mean, self.S


    def computeSigmapoints(self, mean, cov):
        
        N            = self.state_dim
        sigma_points = np.zeros((2*N+1, N))
        l            = np.linalg.cholesky((N + self.lambda_) * self.cov)
        # l          = np.real(sqrtm((N + self.lambda_) * self.cov))
        sigma_points[0, :] = mean

        for i in range(N):
            sigma_points[1 + 2*i, :] = mean + l[i, :]
            sigma_points[2 + 2*i, :] = mean - l[i, :]
        

        return sigma_points
    
    def computeWeights(self, lambda_, alpha, beta, ext_bool = False):
        N  = self.state_dim

        Wm = np.zeros((2*N+1, 1))  # weight for the state
        Wc = np.zeros((2*N+1, 1))  # weight for the covariance

        if ext_bool:
            K      = float(self.measurement_dim)
            Wm[0]  = lambda_ / (N + lambda_ + K)
            Wc[0]  = Wm[0] + (1 - alpha**2 + beta + K)
            Wm[1:] = Wc[1:] = 0.5 / (N + lambda_ + K)

        else:
            N      = int(N)
            Wm[0]  = lambda_ / (N + lambda_)
            Wc[0]  = Wm[0] + (1 - alpha**2 + beta)
            Wm[1:] = Wc[1:] = 0.5 / (N + lambda_)

            
        return Wm, Wc

    def predict(self, dt):
        self.sigma_points = self.computeSigmapoints(self.mean, self.cov) # 2n+1, n
        fXi               = np.zeros((2*self.state_dim + 1, self.state_dim))

        for i in range(2*self.state_dim + 1):
            fXi[i,:] = ukf_track_f(self.sigma_points[i,:], dt)

        self.mean_pred, self.cov_pred = self.unscentedTransform(fXi, self.Q)

        yaw_rate = self.mean_pred[LIDAR_UKF_PREDICTION_REL_YAW_RATE]
        yaw_rate = self.limit_yaw_rateRange(yaw_rate)
        self.mean_pred[LIDAR_UKF_PREDICTION_REL_YAW_RATE] = yaw_rate

        self.fXi = fXi  # 2n+1, n

        self.normalize_heading()
    
    def correct(self, measurement):

        # sigma_points = self.computeSigmapoints(self.mean_pred, self.cov_pred) # 2n+1, n
        hXi          = np.zeros((2*self.state_dim+1, self.measurement_dim))

        for i in range(2*self.state_dim + 1):
            hXi[i, :] = ukf_track_h(self.sigma_points[i,:])
        

        self.hXi = hXi

        self.z_mean, self.S  = self.unscentedTransform(hXi, self.R)
        self.crossCovariance = self.computeCrossCovariance()
        self.kalmanGain      = self.computeKalmanGain()
        
        self.calculation_heading_direction(measurement)

        z_diff    = measurement[LIDAR_UKF_MEASUREMENT_PARAM] - self.z_mean

        self.mean = self.mean_pred + self.kalmanGain @ z_diff
        self.cov  = self.cov_pred  - self.kalmanGain @ self.S @ self.kalmanGain.T

        
        yaw_diff  = self.mean[LIDAR_UKF_PREDICTION_REL_YAW] - measurement[LIDAR_DETECTION_YAW]
        yaw_diff  = self.limit_yawRange(yaw_diff)

        self.mean[LIDAR_UKF_PREDICTION_REL_YAW] = measurement[LIDAR_DETECTION_YAW] + yaw_diff


        self.normalize_heading()

    def computeCrossCovariance(self):
        cov_xz = 0

        
        for i in range(2*self.state_dim + 1):
            x_diff    = (self.fXi[i, :] - self.mean_pred)  # n,1
            z_diff    = (self.hXi[i, :] - self.z_mean)     # m,1

            x_diff = x_diff.reshape(-1,1)
            z_diff = z_diff.reshape(-1,1)
            cov_xz += x_diff@(z_diff.T)*self.Wc[i]     # n,m



        return cov_xz

    def computeKalmanGain(self):
        kalmanGain = self.crossCovariance@np.linalg.inv(self.S)

        return kalmanGain


    def unscentedTransform(self, sigma_points, noise):
        row = sigma_points.shape[0]
        col = sigma_points.shape[1]

        mean_pred = np.zeros((col, 1)).squeeze()
        cov_pred  = np.zeros((col, col))

    
        for i in range(row):
            mean_pred[:] += sigma_points[i,:]*self.Wm[i]

        for i in range(row):

            diff = sigma_points[i,:] - mean_pred[:]
            diff = diff.reshape(-1,1)
            cov_pred[:] += diff@diff.T*self.Wc[i]

        cov_pred += noise

        return mean_pred, cov_pred
    
    def limit_yawRange(self, yaw_diff):
        if yaw_diff > YAW_MAX_VARIATION:
            yaw_diff = YAW_MAX_VARIATION
        elif yaw_diff < -YAW_MAX_VARIATION:
            yaw_diff = -YAW_MAX_VARIATION

        return yaw_diff
    
    def limit_yaw_rateRange(self, yaw_rate):
        if yaw_rate > YAW_RATE_MAXIMUM:
            yaw_rate = YAW_RATE_MAXIMUM
        elif yaw_rate < -YAW_RATE_MAXIMUM:
            yaw_rate = -YAW_RATE_MAXIMUM

        return yaw_rate

    
    def normalize_heading(self):
        # -pi/2 ~ pi/2
        yaw      = self.mean[LIDAR_UKF_PREDICTION_REL_YAW]
        yaw_pred = self.mean_pred[LIDAR_UKF_PREDICTION_REL_YAW]   # prediction

        yaw_rate      = self.mean[LIDAR_UKF_PREDICTION_REL_YAW_RATE]
        yaw_pred_rate = self.mean_pred[LIDAR_UKF_PREDICTION_REL_YAW_RATE]   # prediction

        while not (-pi/2 < yaw <= pi/2):
            if yaw > pi/2:
                yaw -= np.pi
            elif yaw <= -pi/2:
                yaw += np.pi

        while not (-pi/2 < yaw_pred <= pi/2):
            if yaw_pred > pi/2:
                yaw_pred -= np.pi
            elif yaw_pred <= -pi/2:
                yaw_pred += np.pi

        while not (-pi/2 < yaw_rate <= pi/2):
            if yaw_rate > pi/2:
                yaw_rate -= np.pi
            elif yaw_rate <= -pi/2:
                yaw_rate += np.pi

        while not (-pi/2 < yaw_pred_rate <= pi/2):
            if yaw_pred_rate > pi/2:
                yaw_pred_rate -= np.pi
            elif yaw_pred_rate <= -pi/2:
                yaw_pred_rate += np.pi

        self.mean[     LIDAR_UKF_PREDICTION_REL_YAW] = yaw
        self.mean_pred[LIDAR_UKF_PREDICTION_REL_YAW] = yaw_pred

        self.mean[     LIDAR_UKF_PREDICTION_REL_YAW_RATE] = yaw_rate
        self.mean_pred[LIDAR_UKF_PREDICTION_REL_YAW_RATE] = yaw_pred_rate
    
    def calculation_heading_direction(self, measurement):
        yaw_meas = measurement[          LIDAR_DETECTION_YAW]   # correction 
        yaw_pred = self.z_mean[LIDAR_UKF_MEASUREMENT_REL_YAW]   # prediction 

        while not (-pi/2 < yaw_meas <= pi/2):
            if yaw_meas > pi/2:
                yaw_meas -= np.pi
            elif yaw_meas <= -pi/2:
                yaw_meas += np.pi

        while not (-pi/2 < yaw_pred <= pi/2):
            if yaw_pred > pi/2:
                yaw_pred -= np.pi
            elif yaw_pred <= -pi/2:
                yaw_pred += np.pi

        yaw_diff = yaw_meas - yaw_pred

        if abs(yaw_diff) >= pi/2:
            if yaw_meas > yaw_pred:
                yaw_pred += pi
            
            elif yaw_pred > yaw_meas:
                yaw_pred -= pi

        yaw_diff = yaw_pred - yaw_meas
        self.z_mean[LIDAR_UKF_MEASUREMENT_REL_YAW] = yaw_pred


    


