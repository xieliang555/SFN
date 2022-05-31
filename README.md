# Precision Peg-in-hole

Peg-in-hole is essential in both industrial and service robots. Compared with the force-based methods, the vision-based methods make less object contact and perform better convergence under large initial error, but show worse generalization to new objects with unseen shapes. This paper proposes a visual peg-in-hole framework that enables training with several shapes in simulation, and adapting to arbitrary unseen shapes in real world with minimal sim-to-real cost. The core idea is to decouple the generalization of the sensory-motor policy to the design of a fast-adaptable perception module and a simulation-based generic policy module. The framework consists of a segmentation network (SN), a virtual sensor network (VSN), and a controller network (CN). Concretely, the VSN is trained in simulation to measure the peg-hole pose from a peg-hole segmented image, of which the generalization is achieved by inductive bias in architecture. After that, given the shape-agnostic pose measurement, the CN is trained to achieve generic peg-in-hole in simulation. Finally, when applying to real-world unseen holes, we only have to fine-tune the SN for image segmentation required by VSN+CN in real-world peg-in-hole. To further minimize the transfer cost, we propose to automatically collect and annotate the segmentation training data after one-minute human teaching. We deploy the above policy to two typical configurations i.e. eye-in-hand and eye-to-hand, and compare it with a traditional method and the learning-based end-to-end methods on 8 simulated and real unseen holes, exhibiting superior success rate, insertion efficiency, and generalization. At last, an electric vehicle (EV) charging system with the proposed policy inside, achieves a 10/10 success rate in about 3s, using only hundreds of auto-labeled samples for segmentation transfer. 

<div align=center>
    <img src="assets/cover.png" width="40%" ></img>
</div> 

## Automatic data collection and annotation
The sim2real adaptation is achieved by fine-tunning for the SN, with the training data (image+segmentation mask) collected and annotated automaticly. The data collection and model training can be achieved efficiently within 30mins in real world.
<div align=center>
    <img src="assets/v1.gif" width="40%" ></img>
</div> 

## Real world experiments
**Generalization:** the proposed framework generalizes well on different seen and unseen 3D-printed models with sub-millimeter tolerance.
<div align=center>
    <p float="left">
        <img src="assets/v2.gif" width="60%" />
    </p>
</div>

**Robustness:** the hole base can be either fixed (static insertion) or under disturbance (dynamic insertion).
<!-- The proposed method generalizes well on the eye-in-hand experiment setting, where the camera is fixed on the robot end-effector and follows up with the robot. The hole base can be either fixed or under disturbance. -->
<div align=center>
    <p float="left">
<!--         <img src="assets/sim.png" width="150" height="198"/ > -->
<!--         <img src="assets/real.png" width="150" height="198" /> -->
        <img src="assets/v3.gif" width="30%"/>
        <img src="assets/v4.gif" width="30%"/>
    </p>
</div>



## EV charging application
<div align=center>
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
</div>



<div align="center">

| ![](assets/v1_f.gif) | ![](assets/v2_f.gif) |
|:--:|:--:|
|(a)|(b)|
| ![](assets/v3_f.gif ) | ![](assets/v4_f.gif){width=40%} |
|(c)|(d)|

</div>


## Downloads
arXiv: [[SFN](https://arxiv.org/abs/2204.07776)] [[VSN](https://arxiv.org/abs/2205.04297)]
