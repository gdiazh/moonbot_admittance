# ROS 2 Joint-Space Admittance Controller

A ROS 2 implementation of joint-space admittance control for the 7-DOF
SRL-Moonbot robot equipped with an end-effector force/torque (F/T) sensor (Leptrino). 
The controller modifies reference joint trajectories from a leader arm or
another reference source, allowing the follower robot to remain compliant during
the end-effector physical interactions.

> **Safety notice:** Test with conservative limits and low speeds first.
> Keep an accessible emergency stop, verify the wrench frame and sign
> convention, and validate all compensation terms before enabling
> physical interaction. Incorrect frame transforms, sensor calibration,
> or controller gains can cause unexpected motion.

## Features

-   Joint-space admittance for a 7-DOF manipulator.
-   Reference tracking from a leader arm or another joint-reference source.
-   Process end-effector F/T sensor input.
-   Tool mass and center-of-mass (COM) estimation from static measurements at multiple robot orientations.
-   Gravity and bias compensation for the tool attached to the sensor.
-   First-order low-pass filtering of the compensated wrench.
-   Deadband and safety checks for the measured wrench.
-   Directional force shaping for insertion tasks.

## Control overview

The controller treats the commanded joint offset as a virtual
mass-spring-damper system:

$$
M_q \ddot{q}_{\mathrm{off}} + D_q
\dot{q}_{\mathrm{off}} + K_q q_{\mathrm{off}} =
\tau_{\mathrm{ext}}
$$

Here, $q_{\mathrm{off}}$ is the compliant joint offset, $M_q$,
$D_q$, and $K_q$ are virtual inertia, damping, and stiffness matrices,
and $\tau _{\mathrm{ext}}$ is the external wrench mapped
into joint space.

The wrench is mapped using the manipulator Jacobian:

$$\tau *{\mathrm{ext}} = J(q)\^\mathsf{T} F*{\mathrm{ext}}$$

The exact sign depends on the sensor convention. 
This has to be carefully verified before operation.

The continuous dynamics are evaluated in acceleration form:

$$
\ddot{q}_{\mathrm{off}} =
M_{q}^{-1}\left(\tau *{\mathrm{ext}}
-D_q\dot{q}*{\mathrm{off}}
-K_q q_{\mathrm{off}}\right) 
$$

The update is done via a simple forward-Euler rule:

$$
\dot{q}_{\mathrm{off}, k+1} =
\dot{q}_{\mathrm{off} ,k}
+\ddot{q}_{\mathrm{off} ,k} \Delta t 
$$

$$
q_{\mathrm{off}, k+1} = q_{\mathrm{off}, k}
+\dot{q}_{\mathrm{off}, k+1}\Delta t 
$$

The final target is constructed by adding the offset to the reference:

$$
q_{\mathrm{target}} = q_{\mathrm{ref}}+q_{\mathrm{off}} 
$$

The reference $q_{\mathrm{ref}}$ may come from leader-arm joint
mapping or another source, such as a learned policy. The follower's
existing joint controller is responsible for tracking the resulting
target.

## Wrench-processing pipeline

The processing sequence is:

1.  Read the F/T sensor wrench.
2.  Subtract the sensor bias.
3.  Estimate and subtract the gravity wrench of the tool attached to the sensor.
4.  Apply a low-pass filter to reduce measurement noise.
5.  Apply a deadband and safety checks.
6.  Optionally shape the force along a task-specific direction.
7.  Transform the wrench into the frame expected by the Jacobian.
8.  Compute $\tau_{\mathrm{ext}} = J^{\mathsf{T}} F_{\mathrm{ext}}$.
9.  Integrate the admittance dynamics and update the joint target.

### Low-pass filter

A first-order discrete low-pass filter can be written as

$$F_k = \alpha F^{\mathrm{raw}}*k + (1-\alpha )F*{k-1}$$

where $F\^{\mathrm{raw}}_k$ is the current compensated wrench,
$F_k$ is the filtered wrench, and $0 < \alpha < 1$. Larger
$\alpha$ responds faster but smooths less. Smaller
$\alpha$ provides more smoothing but adds delay.

### Tool gravity compensation

Let $m$ be the estimated tool mass, $r_{\mathrm{com}}$ the COM
position relative to the sensor origin, and
$g_b=[0,0,-9.81]^{\mathsf{T}}, \mathrm{m\/s^2}$ the
gravity vector in the robot base frame. If $R_{bs}$ rotates vectors
from the sensor frame to the base frame, gravity expressed in the sensor frame is

$$g_s = R_{bs}\^\mathsf{T} g_b$$

The modeled gravity wrench is

$$
F_{\mathrm{grav}} =
\left[
\begin{matrix}
m g_s\\
r_{\mathrm{com}}\times(m g_s)
\end{matrix}
\right] 
$$

The compensated wrench is then
$F_{\mathrm{ext}}=F_{\mathrm{meas}}-F_{\mathrm{bias}}-F_{\mathrm{grav}}$,
subject to the sensor's sign convention.

## Tool mass and COM calibration

The calibration estimates the mass and COM of the rigid assembly whose
weight is measured by the sensor. The process consist of collecting static 
wrench averages at different end-effector orientations. At each pose,
we use forward kinematics to obtain the sensor orientation and calculate
gravity in the sensor frame.

For pose $i$, let $f_i$ be the measured force after appropriate bias
handling and $g_i$ the gravity vector expressed in the sensor frame.
Under the static-gravity model,

$$f_i \approx m g_i, \qquad  \hat m_i = \frac{g_i^\mathsf{T} f_i}{g_i^\mathsf{T}g_i}$$

Then, we combine measurements across poses to estimate mass robustly, and
estimate the COM from the measured torque:

$$\tau *i \approx r*{\mathrm{com}}\times (m g_i) = -m[g_i]*\times r*{\mathrm{com}}$$

where $[g_i]_{\times}$ is the skew-symmetric matrix for the
cross product. The resulting system is solved by least squares.

Calibration tips:

-   Keep the robot stationary while collecting each sample.
-   Use diverse rotations so gravity points in different directions in the sensor frame.
-   Avoid moving the robot during the averaging window.
-   Check residuals and estimated values for plausibility.

## Parameter tuning

The virtual parameters determine how the offset responds to external interaction:

-   **Inertia $M_q$:** higher values generally reduce acceleration for a given input.
-   **Damping $D_q$:** higher values suppress velocity and usually reduce overshoot.
-   **Stiffness $K_q$:** higher values pull the offset back toward zero more strongly.

For seven joints, a simple initial parameterization is

$$M_q=m_q I_7,\qquad D_q=d_q I_7,\qquad K_q=k_q I_7$$

A general rule is to start with low gains and conservative limits. Increase damping if the
response oscillates. Adjust inertia to change the speed of the initial response.
Reduce stiffness if the robot returns too strongly to the reference. Tune one parameter at
a time and validate at low speed and external forces.

## Integration guidelines

- Set the correct joint-state topic and joint names in the control node.
- Set the correct wrench topic and message type in the control node.
- Confirm the sensor frame, robot base frame, and Jacobian frame.
- Configure the end-effector link used for forward kinematics and the Jacobian.
- Load validated tool mass and COM estimates.
- Set filter, deadband, wrench limits, offset limits, and joint limits.
- Test first without contact, then with soft interactions.

## License

This project is licensed under the Apache License 2.0. You may use, modify, and distribute this
software in accordance with the terms of the license. See the LICENSE file for the full license text.

Unless otherwise stated, the license applies to the original code and documentation in this repository.
Third-party components remain subject to their respective licenses.
