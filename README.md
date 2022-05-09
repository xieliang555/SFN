# Precision Peg-in-hole

Contact-rich peg-in-hole continues to remain challenging for autonomous robots as unsafe insertions limit the robot's ability to learn to understand the contact configuration via extensive trials in real world, especially when considering tight tolerance and general unseen shapes. In this paper, we design the two-stage policy for the high-precision insertion tasks, consisting of a first-stage sensor neural network (SNN) which functions as a sensor to detect general features of position and rotation, and a second-stage architecture which learns to plan and control by reinforcement learning. To achieve sim-to-real, a learnable segmentation module with safe data collection and automatic labeling can be easily trained to decouple the perception and the policy, which helps the policy trained in simulation directly adapt to real world with zero-shot learning. The proposed system applies to both the eye-in-hand and eye-to-hand configurations, under which substantial experiments are conducted in simulation and real world. The results demonstrates that we achieve higher or comparable success rate and efficiency on the unseen shape assembly compared with the baseline methods.

<!-- <center>![(a) experinment setting (b) seen peg shapes (c) unseen peg shapes](assets/cover.png)</center> -->

## Automatic data collection and annotation
The data collection and model training for the segmentation module can be achieved efficiently within 30mins in real world under the automatic data collection and annotation pipeline.
<div align=center>
    <img src="assets/v1.gif" width="40%" ></img>
</div> 

## Eye-to-hand
The proposed method generalizes well on different unseen shapes under sub-millimeter tolerance in real world.
<div align=center>
    <p float="left">
        <img src="assets/cover.png" width="250" height="270" / >
        <img src="assets/v2.gif" width="60%" />
    </p>
</div>

## Eye-in-hand
The proposed method generalizes well on the eye-in-hand experiment setting, where the camera is fixed on the robot end-effector and follows up with the robot. The hole base can be either fixed or under disturbance.
<div align=center>
    <p float="left">
        <img src="assets/sim.png" width="150" height="198"/ >
        <img src="assets/real.png" width="150" height="198" />
        <img src="assets/v3.gif" width="30%"/>
        <img src="assets/v4.gif" width="30%"/>
    </p>
</div>


## Demos
<iframe 
    src="ass/v1_f.mp4" 
    scrolling="no" 
    border="0" 
    frameborder="no" 
    framespacing="0" 
    allowfullscreen="true" 
    height=600 
    width=800> 
</iframe>s


## Downloads

- Video: [[YouTube](https://www.youtube.com/watch?v=L5AhgDvevKA)][[bilibili](https://www.bilibili.com/video/BV1Zf4y1w7ea?spm_id_from=333.999.0.0)]
- PDF: [[Supplementary](https://xieliang555.github.io/post/text/icra_supplementary.pdf)]
- arXiv: 

