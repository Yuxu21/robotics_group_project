'''g1_ball_v.py

   Move the ball periodically between the 

'''

import rclpy
import numpy as np
import tf2_ros

from math               import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from asyncio            import Future
from rclpy.node         import Node
from geometry_msgs.msg  import PoseStamped, TwistStamped
from geometry_msgs.msg  import TransformStamped
from sensor_msgs.msg    import JointState
from std_msgs.msg       import Header
from std_msgs.msg       import Float64

# Marker-related imports (like balldemo.py)
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg      import Point, Vector3, Quaternion
from std_msgs.msg           import ColorRGBA
from rclpy.qos              import QoSProfile, DurabilityPolicy

# Grab the Utilities
from utils.TransformHelpers     import *
from utils.TrajectoryUtils      import *

# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain
# Grab the repulsion torque function (unused for now, but OK)
from hw7code.repulsion          import repulsion


#
#   Trajectory Generator Node Class
#
class TrajectoryNode(Node):
    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object (to end).
        super().__init__(name)
        self.future = future

        # Active joints along the left arm, from torso_link to the hand
        self.jointnames = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]

        # Set up the kinematic chain object. 
        self.chain = KinematicChain(
            self,
            "waist_yaw_link",        # baseframe (link name)
            "left_paddle_link",  # tipframe  (link name)
            self.jointnames,     # expected active joint names
        )
        #self.elbowchain=KinematicChain(self,'world','elbow',self.jointnames[0:4])
        #self.wristchain=KinematicChain(self,'world','wrist',self.jointnames[0:5])


        
        # Initial joint pose and tip pose
        self.q0 = np.radians(np.zeros(len(self.jointnames)))
        (p0, R0, Jv0, Jw0) = self.chain.fkin(self.q0)
        self.p0 = p0
        self.R0 = R0
        self.q0dot=np.zeros(len(self.jointnames))

        # Goal joint pose and tip pose
        # Ball + hit pose definition
        self.ball_radius = 0.03
        self.p_ball = np.array([0.1, 0.4+0.25, 0.2])
        x_ball = np.array([1.0,0.0,0.0])
        # the prehit position
        self.ppre=self.p_ball - 9*self.ball_radius * x_ball
        self.Rpre=R0@Rotx(-pi/2)
        self.qpre=self.solve_ik(self.q0, self.ppre, 0, -pi/2)
        # Command to hit the ball at the back
        self.pg = self.p_ball - self.ball_radius * x_ball
        self.Rg = R0@Rotx(-pi/2)
        self.qg = self.solve_ik(self.qpre, self.pg, -pi/2, -pi/2)


        # Period T = 8 s (4s out, 4s back)
        self.T = 2.5
        self.T_pre=1.5
        self.T_back=self.T-self.T_pre
        self.T_hit=self.T_pre+(self.T-self.T_pre)/2

        # Publishers for joint states and task-space pose/velocity
        self.pubjoint = self.create_publisher(JointState, '/joint_states', 10)
        self.pubpose  = self.create_publisher(PoseStamped, '/pose', 10)
        self.pubtwist = self.create_publisher(TwistStamped, '/twist', 10)
        self.pubcond  = self.create_publisher(Float64, '/condition', 10)
        self.tfbroad  = tf2_ros.TransformBroadcaster(self)

        # ----------------------------------------------------------
        # Ball marker setup (just like balldemo.py, but fixed frame torso_link)
        # ----------------------------------------------------------
        self.ball_radius = 0.03
        self.diam = 2 * self.ball_radius

        # ball dynamics
        self.a       = np.array([0.0, 0.0, -0.981/2])
        self.p0_ball = np.array([4.0, 0.5, self.ball_radius])
        self.v0_ball = (self.p_ball - self.p0_ball - 0.5 * self.a * self.T_hit**2) / self.T_hit
        self.p = self.p0_ball.copy()
        self.v = self.v0_ball.copy()
        self.e=3.0


        #create markers
        self.ball_marker = Marker()
        self.ball_marker.header.frame_id  = "waist_yaw_link"
        self.ball_marker.header.stamp     = self.get_clock().now().to_msg()
        self.ball_marker.action           = Marker.ADD
        self.ball_marker.ns               = "ball"
        self.ball_marker.id               = 1
        self.ball_marker.type             = Marker.SPHERE

        self.ball_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.ball_marker.pose.position    = Point_from_p(self.p)
        self.ball_marker.scale            = Vector3(x=self.diam, y=self.diam, z=self.diam)
        self.ball_marker.color            = ColorRGBA(r=1.0, g=0.0,
                                                      b=0.0, a=1.0)

        # MarkerArray wrapping the single ball
        self.ball_array = MarkerArray(markers=[self.ball_marker])

        # Latched publisher on /visualization_marker_array
        quality = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.pub_ball = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)

        # ----------------------------------------------------------

        # Wait for a connection to happen.  Not strictly necessary.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while (not self.count_subscribers('/joint_states')):
            pass

        # Timer logistics
        self.dt    = 0.01                       # 100Hz.
        self.t     = -self.dt                   # Seconds since start
        self.now   = self.get_clock().now()     # ROS time since 1970
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, 1/self.dt))

    def solve_ik(self, q0, pg, a0, ag):
        """
        Closed-loop inverse kinematics via numerical integration
        along a C^1 cubic spline path in SE(3).

        - Position: spline from p0 to p_hit with v(0) = v(T) = 0.
        - Orientation: spline for scalar angle alpha from 0 to theta
        about axis k, with alpha_dot(0) = alpha_dot(T) = 0.

        Returns:
            q_goal : np.ndarray
                Joint configuration that places the tip at (p_hit, R_hit).
        """
        # Gains and timing
        lp = 20.0
        lR = 20.0
        dt = 0.01
        T  = 4.0                 # "virtual" IK motion duration (s)
        N  = int(T / dt)         # number of integration steps

        # Initial joint configuration and pose
        q_c = q0.copy()
        p0, R0, _, _ = self.chain.fkin(q_c)
        self.get_logger().info(
            f"solve_ik started, initial position = {p0}"
        )
        self.get_logger().info(
            f"initial orientation = {R0}"
        )

        # ---------------------------------------------------------
        # 2) Closed-loop IK integration along the spline path
        # ---------------------------------------------------------
        for n in range(0, N + 1):
            t = n * dt
            if t > T:
                t = T

            # --- Desired position + linear velocity (C^1 spline) ---
            # v(0) = v(T) = 0
            v0 = np.zeros(3)
            vf = np.zeros(3)
            p_d, v_d = spline(t, T, p0, pg, v0, vf)
            ad, addot= spline(t,T,a0,ag,0,0)
            R_d=self.R0@Rotx(ad)
            w_d=self.R0@nx()*addot
            # --- Actual pose and Jacobian at current q_c ---
            p_c, R_c, Jv, Jw = self.chain.fkin(q_c)
            Jq = np.vstack((Jv, Jw))

            # Pseudo-inverse of the full Jacobian
            lam = 0.1
            Jdinv = Jq.T @ np.linalg.inv(Jq @ Jq.T + lam**2 * np.eye(6))

            #Jinv = np.linalg.pinv(Jq)

            # Closed-loop task-space reference
            vref = v_d + lp * ep(p_d, p_c)
            wref = w_d + lR * eR(R_d, R_c)
            xref = np.concatenate((vref, wref))

            # Integrate in joint space
            qdot = Jdinv @ xref
            q_c  = q_c + dt * qdot

        # Final pose for logging
        p_final, R_final, _, _ = self.chain.fkin(q_c)
        pos_err = np.linalg.norm(pg - p_final)
        # self.get_logger().info(
        #     f"solve_ik finished, final position error = {pos_err:.3e}"
        # )
        # self.get_logger().info(
        #     f"final position = {p_final}"
        # )
        # self.get_logger().info(
        #     f"final orientation = {R_final}"
        # )

        return q_c



        

    # Shutdown
    def shutdown(self):
        self.timer.destroy()
        self.destroy_node()


    def update(self):
        # Increment time explicitly to avoid jitter.
        self.t   = self.t   + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt)

        # Stop after 40s (arbitrary).
        if self.t > 8*self.T:
            self.future.set_result("Trajectory has ended")
            return

        ## update the trajectory of robot arm
        t = fmod(self.t, self.T)
        if   (t < self.T_pre):
            #qd,qddot=spline(t,4.0,self.q0,self.qg,self.q0dot, self.qgdot, 0, 0)
            qd,qddot=goto(t,self.T_pre,self.q0,self.qpre)
        else:
            tt=t-self.T_pre
            s=cos(2*pi*tt/(2*self.T_back))
            sdot=-2*pi/(2*self.T_back)*sin(2*pi*tt/(2*self.T_back))
            qd=1/2*(self.q0+self.qpre-2*self.qg)*s**2+1/2*(self.qpre-self.q0)*s+self.qg
            qddot=(self.q0+self.qpre-2*self.qg)*s*sdot+1/2*(self.qpre-self.q0)*sdot
            
        qc    = qd
        qcdot = qddot

        # Forward kinematics at qc
        (pd, Rd, Jv, Jw) = self.chain.fkin(qc)
        vd = Jv @ qcdot
        wd = Jw @ qcdot
        
        ## update the trajectory of the ball:
        # If we are at the start of a new 0..8s cycle, reinitialize
        if t < self.dt:
            self.p = self.p0_ball.copy()
            self.v = self.v0_ball.copy()
        # update the velocity and position
        self.v=self.v+self.a*self.dt
        self.p=self.p+self.v*self.dt
        R = self.ball_radius

        p_rel = Rd.T @ (self.p - pd)   # position relative to hand, in hand coords
        v_rel = Rd.T @ (self.v - vd)   # velocity of ball relative to hand, in hand coords
        dist=np.linalg.norm(p_rel)
        z_plane = R


        if dist<2*self.ball_radius and p_rel[2] < z_plane and v_rel[2]<0.0:
            #dist=np.linalg.norm(p_rel)
            self.get_logger().info(
            f"At impact, time is = {self.t}"
        )
            self.get_logger().info(
            f"dist is = {dist}")

            self.get_logger().info(
            f"speed is = {np.linalg.norm(self.v)}")

            self.get_logger().info(
            f"paddle is = {np.linalg.norm(vd)}")

                    # update the velocity and position
            # --- Position mirror along local y (just like z in balldemo) ---
            # y_new = y_plane + (y_plane - y_old) = 2*y_plane - y_old
            p_rel[2] = 2.0 * z_plane - p_rel[2]
            v_rel[2] *= -self.e

            # (tangential x/z components in hand frame are unchanged)

            # Back to world frame:
            # world: x_world = pd + Rd x_hand
            self.p = pd + Rd @ p_rel
            self.v = vd + Rd @ v_rel


            
        #     # Redirect ball velocity to point along hand velocity vd.
        #     ball_speed = np.linalg.norm(self.v)
        #     hand_speed = np.linalg.norm(vd)

        #         # ---- Velocity update cases ----
        #     if hand_speed > 1e-6 and ball_speed < 1e-6:
        #         # Case 1: fast hand, slow ball -> ball picks up hand velocity
        #         self.v = 3*vd.copy()
        #         self.get_logger().info(f'get into case 1, v is={self.v}')

        #     elif hand_speed > 1e-6 and ball_speed >= 1e-6:
        #         # Case 2: both moving -> keep ball speed, align direction with hand
        #         self.v = 3*ball_speed * vd / hand_speed
        #         self.get_logger().info(f'get into case 2, v is={self.v}')

        #     else:
        #         # Case 3: hand almost stationary -> bounce along normal,
        #         # keeping the ball's speed
        #         if ball_speed > 1e-6:
        #             if dist > 1e-8:
        #                 normal = (self.p - pd) / dist
        #             else:
        #                 # degenerate: hand and ball at same point, pick any direction
        #                 normal = np.array([1.0, 0.0, 0.0])
        #             self.v = 3*ball_speed * normal
        #             self.get_logger().info(f'get into case 3, v is={self.v}')
        #     if dist > 1e-8:
        #         normal = (self.p - pd) / dist
        #     else:
        #         normal = np.array([1.0, 0.0, 0.0])
        #     self.p = pd + normal * self.ball_radius
        # # In case it inpact the ground:
        #  # Check for a bounce - not the change in x velocity is non-physical.
        if self.p[2] < self.ball_radius:
            self.p[2] = self.ball_radius + (self.ball_radius - self.p[2])
            self.v[2] *= -1.0




        # ------------------------------------------
        # Publish joint, pose, twist, TF as before
        # ------------------------------------------
        header = Header(stamp=self.now.to_msg(), frame_id='waist_yaw_link')

        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=qc.tolist(),
            velocity=qcdot.tolist()))

        self.pubpose.publish(PoseStamped(
            header=header,
            pose=Pose_from_Rp(Rd, pd)))

        self.pubtwist.publish(TwistStamped(
            header=header,
            twist=Twist_from_vw(vd, wd)))

        self.tfbroad.sendTransform(TransformStamped(
            header=header,
            child_frame_id='desired',
            transform=Transform_from_Rp(Rd, pd)))

        # ------------------------------------------
        # Publish the ball MarkerArray
        # ------------------------------------------
        self.ball_marker.header.stamp = self.now.to_msg()
        self.ball_marker.pose.position = Point_from_p(self.p)
        self.pub_ball.publish(self.ball_array)


#
#  Main Code
#
def main(args=None):
    rclpy.init(args=args)

    # Future to signal when done
    future = Future()

    # Initialize the node
    trajectory = TrajectoryNode('trajectory', future)

    # Spin until finished
    rclpy.spin_until_future_complete(trajectory, future)

    # Report and shutdown
    if future.done():
        trajectory.get_logger().info("Stopping: " + future.result())
    else:
        trajectory.get_logger().info("Stopping: Interrupted")

    trajectory.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
