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

# Grab the Utilities
from utils.TransformHelpers     import *
from utils.TrajectoryUtils      import *
# Import the format for the condition number message
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Point, Vector3, Quaternion
from std_msgs.msg import Float64
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray
from std_msgs.msg               import ColorRGBA
# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain
# Grab the repulsion torque function
from hw7code.repulsion import repulsion


#
#   Trajectory Generator Node Class
#
#   This inherits all the standard ROS node stuff, but adds an
#   update() method to be called regularly by an internal timer and a
#   shutdown method to stop the timer.
#
#   Arguments are the node name and a future object (to force a shutdown).
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
        #self.elbowchain=KinematicChain(self,'world','elbow',self.jointnames[0:4])
        #self.wristchain=KinematicChain(self,'world','wrist',self.jointnames[0:5])
                # ----------------------------------------------------------
        # Ball marker: fixed in torso_link frame
        # ----------------------------------------------------------
        self.ball_radius = 0.01
        diam = 2 * self.ball_radius

        # Create the marker once, like in balldemo
        self.ball_marker = Marker()
        self.ball_marker.header.frame_id  = "torso_link"
        self.ball_marker.header.stamp     = self.get_clock().now().to_msg()
        self.ball_marker.action           = Marker.ADD
        self.ball_marker.ns               = "ball"
        self.ball_marker.id               = 1
        self.ball_marker.type             = Marker.SPHERE

        self.ball_marker.pose.orientation = Quaternion(x=0.0, y=0.0,
                                                       z=0.0, w=1.0)
        self.ball_marker.pose.position    = Point(x=0.5, y=0.5, z=0.5)

        self.ball_marker.scale            = Vector3(x=diam, y=diam, z=diam)
        self.ball_marker.color            = ColorRGBA(r=1.0, g=0.0,
                                                      b=0.0, a=1.0)

        # Put it in a MarkerArray
        self.ball_array = MarkerArray(markers=[self.ball_marker])

        # Publisher for the MarkerArray, latched (TRANSIENT_LOCAL) like balldemo
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.pub_ball = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)



        #FIXME: WHAT DO YOU NEED TO DO TO INITIALIZE THE TRAJECTORY?
        # Define the matching initial joint/task positions.
        self.q0 = np.radians(np.zeros(len(self.jointnames)))
        (p0,R0,Jv0, Jw0)=self.chain.fkin(self.q0)
        self.p0 = p0
        self.R0 = R0

        self.qg=np.array([-pi/2,pi/4,pi,pi/4,0.0])
        (pg,Rg,Jvg,Jwg)=self.chain.fkin(self.qg)
        self.pg=pg
        self.Rg=Rg
        
        ## set a period of 8 s. first 4 s: q0 to qg last 4s: qg to q0
        self.T=8.0

        # self.qc=self.q0.copy()
        # self.lp= 20
        # self.lR=20
        # self.ls=5
        # self.pdp=self.p0.copy()
        # self.Rdp=self.R0.copy()

        self.pubjoint = self.create_publisher(JointState, '/joint_states', 10)
        self.pubpose  = self.create_publisher(PoseStamped, '/pose', 10)
        self.pubtwist = self.create_publisher(TwistStamped, '/twist', 10)
        # Create the publisher for the condition number.
        self.pubcond = self.create_publisher(Float64, '/condition', 10)
        self.tfbroad  = tf2_ros.TransformBroadcaster(self)
        # Publisher for the visualization marker (the ball)
        self.pubmarker = self.create_publisher(Marker, '/visualization_marker', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        self.dt    = 0.01                       # 100Hz.
        self.t     = -self.dt                   # Seconds since start
        self.now   = self.get_clock().now()     # ROS time since 1970
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, 1/self.dt))
        
        

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()


    def update(self):
        # Increment time.  We do so explicitly to avoid system jitter.
        self.t   = self.t   + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt)
        # Stop everything after 8s - makes the graphing nicer.
        if self.t > 40.0:
            self.future.set_result("Trajectory has ended")
            return
        

        t = fmod(self.t, 8.0)
        if   (t < 4.0):
            (s0,s0dot)=goto(t,4.0,0.0,1.0)
            qd=self.q0+(self.qg-self.q0)*s0
            qddot=(self.qg-self.q0)*s0dot
            
        else:
            (s0,s0dot)=goto(t-4.0,4.0,0.0,1.0)
            qd=self.qg+(self.q0-self.qg)*s0
            qddot=(self.q0-self.qg)*s0dot
        qc=qd
        qcdot=qddot
        # t= self.t % 8.0
        # if t<4.0:
        #     ## From (p0,R0)->(pleft,Rleft)
        #     (s0,s0dot)=goto(self.t,3.0,0.0,1.0)
        #     pd=self.p0+(self.pright-self.p0)*s0
        #     vd=(self.pright-self.p0)*s0dot
        #     Rd=Rotx(0)
        #     wd=nx()*0
        # else:
        #     t=self.t-3.0
        #     s=cos(2*pi*t/5)
        #     sdot=-2*pi/5*sin(2*pi*t/5)
        #     pd=1/2*(self.pleft+self.pright-2*self.pmid)*s**2+1/2*(self.pright-self.pleft)*s+self.pmid
        #     vd=(self.pleft+self.pright-2*self.pmid)*s*sdot+1/2*(self.pright-self.pleft)*sdot
        #     Rd=Rotz(-pi/4*(s-1))@Rotx(pi/4*(s-1))
        #     wd=nz()*(-pi/4)*sdot+(pi/4*sdot)*Rotz(-pi/4*(s-1))@nx()
        (pd,Rd, Jv, Jw)=self.chain.fkin(qc)
        vd=Jv@qcdot
        wd=Jw@qcdot
            
    

        # qclast=self.qc
        # pdp=self.pdp
        # Rdp=self.Rdp
        # #Compute forward kinematic, get the previous tip position and orientation
        # (pcp,Rcp, Jv,Jw)=self.chain.fkin(qclast)
        # #Compute the previous Jacobian
        # Jq=np.vstack((Jv,Jw))
        # if 5.4 <= self.t <= 5.6:
        #     self.get_logger().info(f"cond(J) = {np.linalg.cond(Jq):.2e}")
        # L=0.4
        # Jb=np.diag(np.array([1/L,1/L,1/L,1.0,1.0,1.0]))@Jq
        
        # Jinv=np.linalg.pinv(Jq)
        # #Compute xdotref
        # vref=vd+self.lp*ep(pdp, pcp)
        # wref=wd+self.lR*eR(Rdp, Rcp)
        # xref=np.concatenate((vref,wref))
        # # Get the repulsion torque
        # tau = repulsion(qclast, self.elbowchain, self.wristchain)
        # qsdot=self.ls*tau
        # #Compute qcdot
        # Proj=np.eye(7)-Jinv@Jq
        # qcdot=Jinv@xref+Proj@qsdot
        # #numerical integration
        # qc=qclast+self.dt*qcdot
        # #update the qc, pdp, Rdp
        # self.qc=qc
        # self.pdp=pd
        # self.Rdp=Rd



        ##############################################################
        # Finish by publishing the data (joint and task commands).
        #  qc and qcdot = Joint Commands  as  /joint_states  to view/plot
        #  pd and Rd    = Task pos/orient as  /pose & TF     to view/plot
        #  vd and wd    = Task velocities as  /twist         to      plot
        header=Header(stamp=self.now.to_msg(), frame_id='torso_link')
        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=qc.tolist(),
            velocity=qcdot.tolist()))
        self.pubpose.publish(PoseStamped(
            header=header,
            pose=Pose_from_Rp(Rd,pd)))
        self.pubtwist.publish(TwistStamped(
            header=header,
            twist=Twist_from_vw(vd,wd)))
        self.tfbroad.sendTransform(TransformStamped(
            header=header,
            child_frame_id='desired',
            transform=Transform_from_Rp(Rd,pd)))
        # Publish the condition number.
        #self.pubcond.publish(Float64(data=condition))
        self.ball_marker.header.stamp = self.now.to_msg()
        self.pub_ball.publish(self.ball_array)
        
        


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal when the trajectory ends.
    future = Future()

    # Initialize the trajectory generator node.
    trajectory = TrajectoryNode('trajectory', future)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory is
    # complete (as signaled by the future object).
    rclpy.spin_until_future_complete(trajectory, future)

    # Report the reason for shutting down.
    if future.done():
        trajectory.get_logger().info("Stopping: " + future.result())
    else:
        trajectory.get_logger().info("Stopping: Interrupted")

    # Shutdown the node and ROS.
    trajectory.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


