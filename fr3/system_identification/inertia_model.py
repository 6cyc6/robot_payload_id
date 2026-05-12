import time

from os import system

import numpy as np
import pinocchio as pin

from scipy.linalg import block_diag
from system_identification.utils import *

# This System identification algorithm is based on the original paper by:
#
#   Christopher G Atkeson, Chae H An, and John M Hollerbach.
#   Estimation of inertial parameters of manipulator loads and links.
#   The International Journal of Robotics Research, 5 (3), 101-119, 1986
#
# Most results come from chapter 6.3 Inertial Parameter Estimation in Springer
# Handbook of Robotics:
#
#   Siciliano, Bruno, Oussama Khatib, and Torsten Kröger, eds. Springer handbook
#   of robotics. Vol. 200. Berlin: springer, 2008.


class InertiaModel(object):
    def __init__(self, robot_name, **kwargs):
        self.name = "inertiaCoM"
        urdf_file = f"{robot_name}.urdf"
        urdf_file = find_path(urdf_file, "../robot_description")
        self._friction = kwargs.get("friction", False)

        self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()
        self.njoints = self.model.njoints - 1  # the first joint is the universe joint
        system_inertia = self.model.inertias.tolist()[
            1 : 1 + self.njoints
        ]  # skip the universe joint
        self.damping = self.model.damping
        self.coulomb = self.model.friction
        self._gravity = np.array(self.model.gravity.linear, copy=True)
        self.total_mass = 0
        self.iden_masses = []
        self.masses = []
        self.CoMs_lever = []
        self.inertia_CoM = []

        # record the index of continuous joints, in pinocchio,  the joint position
        # for continuous joints is (cos(alpha), sin(alpha)) instead of (alpha)
        # the inertia from model.inertias is at CoM
        self.idx_continuous_joint = []
        joint_configs = self.model.nqs[1:]  # the first joint is the universe joint
        for i in range(self.njoints):
            if joint_configs[i] > 1:
                self.idx_continuous_joint.append(i)
        self._joint_limit()

        # attribute buffers
        self._proj = []
        self.dyn_param_0 = []

        # API to retrieve the dynamic parameters first
        for i in range(self.njoints):
            _dyn_param_i = system_inertia[i].toDynamicParameters()
            self.dyn_param_0.append(_dyn_param_i)
            self.total_mass += system_inertia[i].mass
            self.masses.append(system_inertia[i].mass)
            self.CoMs_lever.append(system_inertia[i].lever)
            self.inertia_CoM.append(system_inertia[i].inertia)
        self.masses_CAD, self.GoMs_lever, self.inertias_CAD = retrieve_geo_fromCAD(
            robot_name
        )

        self._dyn_param_0 = np.vstack(self.dyn_param_0)
        self.dyn_param = self._dyn_param_0.reshape(-1)  # (njointsx10, )
        self.iden_param = None
        self._init_ref_param()
        self._init_data()

    def _disable_gravity(self):
        logger.info(f"Disable gravity")
        self.model.gravity.linear = np.zeros_like(self._gravity)

    def _init_ref_param(self):
        self.inertia_param_per_joint = 10
        self.num_param_total = 10 * self.njoints
        self.ref_param = self.dyn_param

    def _init_data(self):
        self.J_prior = []
        self.Q = []
        self.Qs = []
        for i in range(self.njoints):
            pi_dyn_i = self.dyn_param[
                i
                * self.inertia_param_per_joint : (i + 1)
                * self.inertia_param_per_joint
            ]
            J = inertiaVecToPinertia(pi_dyn_i)
            self.J_prior.append(J)
            Ic, Qs = inertiaVecToIcQs(pi_dyn_i)
            self.Qs.append(Qs)
            Q = inertiaVecToQ(pi_dyn_i)
            self.Q.append(Q)

    def _joint_limit(self):
        joint_pos_limits = []
        for joint_idx in range(self.njoints):
            if joint_idx in self.idx_continuous_joint:
                pos_limit = np.pi * 10  # special case for continuous joints
            else:
                pos_limit = self.model.upperPositionLimit[joint_idx]
            joint_pos_limits.append(pos_limit)
        self.joint_pos_limits = joint_pos_limits
        self.joint_vel_limits = self.model.velocityLimit

    def _api_regressor(self, q, qd, qdd):
        # the part without friction
        reg_without_friction = pin.computeJointTorqueRegressor(
            self.model, self.data, q, qd, qdd
        )
        if self._friction:
            qds = np.zeros(shape=(self.njoints, self.njoints))
            for i in range(self.njoints):
                qds[i][i] = qd[i]
            reg = np.hstack((reg_without_friction, qds))
        else:
            reg = reg_without_friction
        return reg

    def _construct_regressor(self, q, qd, qdd):
        # preprocess data for the continuous joint case
        q = self._preprocess_q(q)
        return self._api_regressor(q, qd, qdd)

    def _preprocess_q(self, q):
        if len(self.idx_continuous_joint) != 0:
            for idx in self.idx_continuous_joint:
                q = q.tolist()
                theta = q[idx]
                # pop out the joint position theta of the continuous joint
                # and replace with (costheta, sintheta)
                q.pop(idx)
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                q.insert(idx, sin_theta)
                q.insert(idx, cos_theta)
        q = np.array(q)
        return q

    def _regressor(self, q_qd_qdd):
        q, qd, qdd = q_qd_qdd
        reg = self._construct_regressor(q, qd, qdd)
        return reg

    def regressor(self, qs, qds, qdds):
        qs_qds_qdds = zip(qs, qds, qdds)
        reg_list = list(map(self._regressor, qs_qds_qdds))
        reg_batch = np.vstack(reg_list)
        return reg_batch

    def inv_dyn(self, q, qd, qdd):
        reg = self._construct_regressor(q, qd, qdd)
        tau = np.dot(reg, self.dyn_param)
        return tau

    def for_dyn(self, q, qd, tau):
        M_q, c_qqd, g_q, f = self.dyn_model(q, qd)
        inv_M_q = np.linalg.inv(M_q)
        qdd = np.dot(inv_M_q, tau + f - c_qqd - g_q)
        return qdd

    def dyn_model(self, q, qd):
        if self._friction:
            self.dyn_param_idx = self.dyn_param.reshape(
                self.njoints, -1
            )  # for indexing the friction parameters
            f = self.dyn_param_idx[:, 10] * qd
        else:
            f = np.zeros(shape=qd.shape)

        # f = f.reshape(-1, 1)
        q_zero = np.zeros(qd.shape)
        g_q = self.inv_dyn(q, q_zero, q_zero)
        c_qqd = self.inv_dyn(q, qd, q_zero) - g_q + f
        M_q = np.zeros(shape=(self.njoints, self.njoints))
        for i in range(self.njoints):
            _qdd = np.zeros(shape=qd.shape)
            _qdd[i] = 1.0
            M_q[:, i] = self.inv_dyn(q, q_zero, _qdd) - g_q
        return M_q, c_qqd, g_q, f

    def simulate(self, q, qd, tau, dt=1e-4):
        M_q, c_qqd, g_q, f = self.dyn_model(q, qd)
        inv_M_q = np.linalg.inv(M_q)
        qdd = np.dot(inv_M_q, tau + f - c_qqd - g_q)
        qd = qd + qdd * dt
        q = q + qd * dt
        return q, qd, qdd

    def sync_param(self, dyn_param_iden):
        self.iden_param = dyn_param_iden
        self.iden_masses = [
            dyn_param_iden[i * self.inertia_param_per_joint]
            for i in range(self.njoints)
        ]
        self.inertia_param = self.iden_param

    def predict(self, q, dq, ddq):
        if self.iden_param is None:
            param = self.dyn_param.reshape(-1, 1)
        else:
            param = self.iden_param.reshape(-1, 1)
        regressor = self.regressor(q, dq, ddq)
        return regressor @ param

    def param2dict(self):
        inertia_param_dict = {}
        for i in range(self.njoints):
            start_idx, end_idx = i * 10, (i + 1) * 10
            param = self.inertia_param[start_idx:end_idx]
            mass = float(param[0])
            xyz = [float(p) / mass for p in param[1:4]]
            l_inertia_body_frame = [float(p) for p in param[4:10]]
            I_inertia_body_frame = I(l_inertia_body_frame)
            skew_xyz = skew_symmetric(xyz)
            I_inertia_CoM_frame = I_inertia_body_frame - mass * (skew_xyz @ skew_xyz.T)
            ixx = float(I_inertia_CoM_frame[0][0])
            ixy = float(I_inertia_CoM_frame[0][1])
            iyy = float(I_inertia_CoM_frame[1][1])
            ixz = float(I_inertia_CoM_frame[0][2])
            iyz = float(I_inertia_CoM_frame[1][2])
            izz = float(I_inertia_CoM_frame[2][2])

            link = f"link_{i+1}"
            link_inertia = {
                "mass": mass,
                "xyz": xyz,
                "ixx": ixx,
                "ixy": ixy,
                "iyy": iyy,
                "ixz": ixz,
                "iyz": iyz,
                "izz": izz,
            }
            inertia_param_dict[link] = link_inertia
        return inertia_param_dict


class InertiaModelGeometryInvariant(InertiaModel):
    def __init__(self, robot_name, **kwargs):
        super(InertiaModelGeometryInvariant, self).__init__(robot_name, **kwargs)
        self.name = "inertiaGoM"
        self._init_GoM_related_terms()
        self._get_volumn_GoM()

    def _init_ref_param(self):
        self.inertia_param_per_joint = 4
        self.num_param_total = 4 * self.njoints
        ref_param = []
        for i in range(self.njoints):
            start = 10 * i
            ref_param.extend(self.dyn_param[start : start + 4])
        self.ref_param = np.array(ref_param)

    def _get_volumn_GoM(self):
        _file = find_path("iiwas_volumn.yml", "../robot_description")
        _dict = load_yaml(_file)
        self.Vs = [_dict[f"link{i}"] for i in range(8)]

    def _init_GoM_related_terms(self):
        Gs = []
        normalized_I_gs = []
        r_gs = []

        for i in range(self.njoints):
            m_i = self.masses[i]
            m_gi = self.masses_CAD[i]
            r_gi = self.GoMs_lever[i] / m_gi
            I_CoM_CAD = self.inertias_CAD[i]
            g_x, g_y, g_z = r_gi[0], r_gi[1], r_gi[2]

            I_gi = I_CoM_CAD + m_gi * (skew_symmetric(r_gi) @ skew_symmetric(r_gi).T)
            I_GoM = I_gi - m_gi * (skew_symmetric(r_gi) @ skew_symmetric(r_gi).T)
            normalized_I_GoM = I_GoM / m_gi
            I_xxGoM = normalized_I_GoM[0][0]
            I_xyGoM = normalized_I_GoM[0][1]
            I_yyGoM = normalized_I_GoM[1][1]
            I_xzGoM = normalized_I_GoM[0][2]
            I_yzGoM = normalized_I_GoM[1][2]
            I_zzGoM = normalized_I_GoM[2][2]
            G_i = np.array(
                [
                    [I_xxGoM, 0.0, 2 * g_y, 2 * g_z],
                    [I_xyGoM, -g_y, -g_x, 0.0],
                    [I_yyGoM, 2 * g_x, 0.0, 2 * g_z],
                    [I_xzGoM, -g_z, 0.0, -g_x],
                    [I_yzGoM, 0.0, -g_z, -g_y],
                    [I_zzGoM, 2 * g_x, 2 * g_y, 0.0],
                ]
            )
            Gs.append(G_i)
            normalized_I_gs.append(I_gi / m_i)
            r_gs.append(r_gi)

        self.G_block_diag = block_diag(
            Gs[0], Gs[1], Gs[2], Gs[3], Gs[4], Gs[5], Gs[6]
        )  # @TODO: find a elegant way to build block diagonal matrix
        self.normalized_I_gs = normalized_I_gs
        self.r_gs = r_gs

        idx_CoM = []
        idx_I = []
        for i in range(self.njoints * 10):
            if i % 10 < 4:
                idx_CoM.append(i)
            else:
                idx_I.append(i)
        self.idx_CoM = idx_CoM
        self.idx_I = idx_I

    def recover_gi(self):
        normalized_I_gs = []
        for i in range(self.njoints):
            start_idx, end_idx = i * 10, (i + 1) * 10
            param = self.inertia_param[start_idx:end_idx]
            mass = param[0]
            r_ci = [p / mass for p in param[1:4]]
            l_inertia_body = [p for p in param[4:10]]
            I_inertia_body = I(l_inertia_body)
            skew_ci = skew_symmetric(r_ci)
            I_inertia_CoM = I_inertia_body - mass * (skew_ci @ skew_ci.T)
            r_gi = self.r_gs[i]
            r_cg = r_ci - r_gi
            skew_cg = skew_symmetric(r_cg)
            I_gi = I_inertia_CoM + mass * (skew_cg @ skew_cg.T)
            normalized_I_gs.append(I_gi / mass)
        self.normalized_I_gs_iden = normalized_I_gs

    def sync_param(self, dyn_param_iden):
        self.iden_param = dyn_param_iden
        self.iden_masses = [
            dyn_param_iden[i * self.inertia_param_per_joint]
            for i in range(self.njoints)
        ]

        # note that inertia here is in the body frame
        pi_I = self.G_block_diag @ self.iden_param
        pi_CoM = self.iden_param
        pi_CoM = np.split(pi_CoM, self.njoints)
        pi_I = np.split(pi_I, self.njoints)
        pi = []
        for i in range(self.njoints):
            pi.extend(pi_CoM[i])
            pi.extend(pi_I[i])
        self.inertia_param = pi
        self.recover_gi()

    def regressor(self, qs, qds, qdds):
        reg = super().regressor(qs, qds, qdds)

        # split regressor into CoM and inertia related part
        # shape of Y_CoM and Y_I are (N, 4*7), (N, 6*7) respectively
        Y_CoM = reg[:, self.idx_CoM]
        Y_I = reg[:, self.idx_I]
        Y_GoM = Y_CoM + Y_I @ self.G_block_diag
        return Y_GoM


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rigid body dynamics")
    parser.add_argument("--robot", type=str, default="iiwas")
    args = parser.parse_args()

    robot_name = args.robot
    SysID = InertiaModel(robot_name)
    q = np.array([0.0, 3 / 4 * np.pi])
    qd = np.array([0.0, 0.0])
    qdd = np.array([0.0, 0.0])
    _tau = SysID.inv_dyn(q, qd, qdd)
    _M_q, _c_qqd, _g_q, _f = SysID.dyn_model(q, qd)
    _qdd = SysID.for_dyn(q, qd, _tau)

    qs, qds, qdds = [], [], []
    tau = np.array([0.0, 0.0])
    horizon = 20000
    for i in range(horizon):
        qs.append(q)
        qds.append(qd)
        qdds.append(qdd)
        q, qd, qdd = SysID.simulate(q, qd, tau)
    qs = np.array(qs)
    qds = np.array(qds)
    qdds = np.array(qdds)
    t = np.arange(horizon)
    vis_traj(t, qs, qds, qdds)
    time1 = time.time()
    result = SysID.regressor(qs, qds, qdds)
    print(f"Robot name: {robot_name}")
    print(f"Time cost: {time.time()-time1}")
    print(f"Shape of regressor: {result.shape}")
