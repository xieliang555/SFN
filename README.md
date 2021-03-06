# Learning to Fill the Seam by Vision: Sub-millimeter Peg-in-hole on Unseen Shapes in Real World

on arXiv: [https://arxiv.org/abs/2204.07776](https://arxiv.org/abs/2204.07776)

by Liang Xie, Hongxiang Yu, Yinghao Zhao, Haodong Zhang, Zhongxiang Zhou, Minhang Wang, Yue Wang$^*$, Rong Xiong (Zhejiang University)

> Peg-in-hole is essential in both industrial and service robots. Compared with the force-based methods, the vision-based methods make less object contact and perform better convergence under large initial error, but show worse generalization to new objects with unseen shapes. This paper proposes a visual peg-in-hole framework that enables training with several shapes in simulation, and adapting to arbitrary unseen shapes in real world with minimal sim-to-real cost. The core idea is to decouple the generalization of the sensory-motor policy to the design of a fast-adaptable perception module and a simulation-based generic policy module. The framework consists of a segmentation network (SN), a virtual sensor network (VSN), and a controller network (CN). Concretely, the VSN is trained in simulation to measure the peg-hole pose from a peg-hole segmented image, of which the generalization is achieved by inductive bias in architecture. After that, given the shape-agnostic pose measurement, the CN is trained to achieve generic peg-in-hole in simulation. Finally, when applying to real-world unseen holes, we only have to fine-tune the SN for image segmentation required by VSN+CN in real-world peg-in-hole. To further minimize the transfer cost, we propose to automatically collect and annotate the segmentation training data after one-minute human teaching. We deploy the above policy to two typical configurations i.e. eye-in-hand and eye-to-hand, and compare it with a traditional method and the learning-based end-to-end methods on 8 simulated and real unseen holes, exhibiting superior success rate, insertion efficiency, and generalization. At last, an electric vehicle (EV) charging system with the proposed policy inside, achieves a 10/10 success rate in about 3s, using only hundreds of auto-labeled samples for segmentation transfer. 

*Note: This is the reference implementation of the accepted paper in ICAR 2022.   We extend the work to a broader application scenario with extensive experiments.  The evolved paper based on this repository can be found in [https://arxiv.org/abs/2205.04297](https://arxiv.org/abs/2205.04297)*

<div align=center>
    <img src="assets/cover.png" width="40%" ></img>
</div> 

## Automatic data collection and annotation
The sim2real adaptation is achieved by fine-tunning for the SN, with the training data (image+segmentation mask) collected and annotated automaticly. The data collection and model training can be achieved efficiently within 30mins in real world.
<div align=center>
    <img src="assets/v1.gif" width="40%" ></img>
</div> 

## Real world experiments
**Generalization:** the proposed framework generalizes well on different seen and unseen 3D-printed models with sub-millimeter(0.6mm) tolerance.
<div align=center>
    <p float="left">
        <img src="assets/v2.gif" width="60%" />
    </p>
</div>

**Robustness:** the hole base can be either fixed (static insertion) or under disturbance (dynamic insertion).
<!-- The proposed method generalizes well on the eye-in-hand experiment setting, where the camera is fixed on the robot end-effector and follows up with the robot. The hole base can be either fixed or under disturbance. -->


<div align="center">

| <img src="assets/v3.gif" width="300" /> | <img src="assets/v4.gif" width="300"  /> |
|:--:|:--:|
|static insertion|dynamic insertion|

</div>


## EV charging application
(a) The complete insertion process starts by commonding the robot-peg to approach the hole, and touch the surface by force control. Then the pose alignment is achieved by the proposed algotithm with pure vision feedback. Finally, when the alignment finishes, the peg will be pushed into the hole compliantly by the force control. (b) We test the aligning algorithm in the automatic EV charging system with a real car (Tesla Model 3) and achieve 10/10 success rate. (c) We evaluate the robustness of the algorithm by changing the charging scenario. (d) We further evaluate the algorithm in more challenging conditions. The agent needs to perform dynamic insertion while a person is manually moving the EV socket.

<!-- <div align=center>
    <p float="left">
        <img src="assets/v1_f.gif" title="a" width="300" height="200"/ >
        <img src="assets/v2_f.gif" width="300" height="200" />
    </p>
</div>


<div align="center">
    <p float="left">
        <figure>
                <img src="assets/v3_f.gif" width="300" height="200" title="c" alt="alt text"/>
                <img src="assets/v4_f.gif" width="300" height="200"/>
            <figcaption>a</figcaption>
        </figure>
    </p>
</div> -->



<div align="center">

| <img src="assets/v3_f.gif" width="300" height="200"/> | <img src="assets/v2_f.gif" width="300" height="200" /> |
|:--:|:--:|
|(a)|(b)|
| <img src="assets/v1_f.gif" width="300" height="200" title="c" alt="alt text"/> ) | <img src="assets/v4_f.gif" width="300" height="200"/> |
|(c)|(d)|

</div>


## TODO
- [ ] add instructions for this repository.
- [x] add demos of the EV charging applications.
