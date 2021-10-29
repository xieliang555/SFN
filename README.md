# Seam Filling Net

In the peg insertion task,  human pays attention to the seam between the peg and the hole and tries to fill it continuously with visual feedback. By imitating the human's manner, we design architecture with position and orientation estimators based on the seam representation for pose alignment, which proves to generalize well over unseen peg geometries. By putting the estimators into the closed-loop control with reinforcement learning, we further achieve a higher or comparable success rate, efficiency, precision, and robustness compared with the baseline methods. The policy is trained totally in simulation without any manual intervention. To achieve sim-to-real, a learnable segmentation module with automatic data collecting and labeling can be easily trained to decouple the perception and the policy, which helps the model trained in simulation quickly adapting to the real world with negligible effort. The proposed method works well on different experiment settings including eye-to-hand and eye-in-hand. Results are presented in simulation and on a physical robot.

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
        <img src="assets/cover.png" width="250" height="300" / >
        <img src="assets/v2.gif" width="60%" />
    </p>
</div>

## Eye-in-hand
The proposed method generalizes well on the eye-in-hand experiment setting, where the camera is fixed on the robot end-effector and follows up with the robot. The hole base can be either fixed or under disturbance.
<div align=center>
    <p float="left">
        <img src="assets/sim.png" width="150" height="248"/ >
        <img src="assets/real.png" width="150" height="248" />
        <img src="assets/v3.gif" width="30%"/>
        <img src="assets/v4.gif" width="30%"/>
    </p>
</div>


## Downloads

- Video: [[YouTube](https://www.youtube.com/watch?v=L5AhgDvevKA)][[bilibili](https://www.bilibili.com/video/BV1Zf4y1w7ea?spm_id_from=333.999.0.0)]
- PDF: [[Supplementary](https://xieliang555.github.io/post/text/icra_supplementary.pdf)]
- arXiv: coming soon

