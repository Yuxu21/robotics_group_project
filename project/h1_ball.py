'''h1_ball.py

   Move the H1 left arm and publish a fixed ball marker in RVIZ.

   - Publishes joint states for the H1 left arm.
   - Publishes a red ball as a MarkerArray on /visualization_marker_array.

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
            "left_hand_joint",
        ]

        # Set up the kinematic chain object. 
        self.chain = KinematicChain(
            self,
            "torso_link",        # baseframe (link name)
            "L_hand_base_link",  # tipframe  (link name)
            self.jointnames,     # expected active joint names
        )

        # Initial joint pose and tip pose
        self.q0 = np.radians(np.zeros(len(self.jointnames)))
        (p0, R0, Jv0, Jw0) = self.chain.fkin(self.q0)
        self.p0 = p0
        self.R0 = R0

        

        # Goal joint pose and tip pose
        # self.qg = np.array([-pi/2, pi/4, pi, pi/4, 0.0])
        # (pg, Rg, Jvg, Jwg) = self.chain.fkin(self.qg)
        # self.pg = pg
        # self.Rg = Rg

        # Period T = 8 s (4s out, 4s back)
        self.T = 8.0

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
        diam = 2 * self.ball_radius

        # Create the marker once
        self.ball_marker = Marker()
        self.ball_marker.header.frame_id  = "torso_link"
        self.ball_marker.header.stamp     = self.get_clock().now().to_msg()
        self.ball_marker.action           = Marker.ADD
        self.ball_marker.ns               = "ball"
        self.ball_marker.id               = 1
        self.ball_marker.type             = Marker.SPHERE

        self.ball_marker.pose.orientation = Quaternion(x=0.0, y=0.0,
                                                       z=0.0, w=1.0)
        self.ball_marker.pose.position    = Point(x=0.5, y=0.4, z=0.3)

        self.ball_marker.scale            = Vector3(x=diam, y=diam, z=diam)
        self.ball_marker.color            = ColorRGBA(r=1.0, g=0.0,
                                                      b=0.0, a=1.0)
        
        # Define the hit position and orientation
        self.p_ball = np.array([0.5, 0.4, 0.3])
        # (Optional) keep marker position in sync:
        #self.ball_marker.pose.position = Point_from_p(self.p_ball)
        # Ball orientation: currently identity (aligned with torso_link)
        R_ball = Reye()  # from utils.TransformHelpers
        # Ball's +z axis in torso_link frame
        z_ball = R_ball @ np.array([0.0, 0.0, 1.0])
        # Desired hit position: on the ball surface along +z of the ball
        self.p_hit = self.p_ball + self.ball_radius * z_ball
        # Desired orientation: flip the z-axis (rotate 180° around x)
        # So tool-z points *into* the ball.
        self.R_hit = R_ball @ Rotx(pi)

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


    # Shutdown
    def shutdown(self):
        self.timer.destroy()
        self.destroy_node()


    def update(self):
        # Increment time explicitly to avoid jitter.
        self.t   = self.t   + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt)

        # Stop after 40s (arbitrary).
        if self.t > 40.0:
            self.future.set_result("Trajectory has ended")
            return

        # Simple joint-space trapezoid between q0 and qg.
        t = fmod(self.t, 8.0)
        if t < 4.0:
            (s0, s0dot) = goto(t, 4.0, 0.0, 1.0)
            qd    = self.q0 + (self.qg - self.q0) * s0
            qddot = (self.qg - self.q0) * s0dot
        else:
            (s0, s0dot) = goto(t - 4.0, 4.0, 0.0, 1.0)
            qd    = self.qg + (self.q0 - self.qg) * s0
            qddot = (self.q0 - self.qg) * s0dot

        qc    = qd
        qcdot = qddot

        # Forward kinematics at qc
        (pd, Rd, Jv, Jw) = self.chain.fkin(qc)
        vd = Jv @ qcdot
        wd = Jw @ qcdot

        # ------------------------------------------
        # Publish joint, pose, twist, TF as before
        # ------------------------------------------
        header = Header(stamp=self.now.to_msg(), frame_id='torso_link')

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
