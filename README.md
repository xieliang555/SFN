# SFN

<!-- <center>![(a) experinment setting (b) seen peg shapes (c) unseen peg shapes](assets/cover.png)</center> -->

<div align=center>
<img src="assets/cover.png" width="40%" ></img>
</div>


## Abstract

<p align="center">In the peg insertion task,  human pays attention to the seam between the peg and the hole and tries to fill it continuously with visual feedback. By imitating the human's manner, we design architecture with position and orientation estimators based on the seam representation for pose alignment, which proves to generalize well over unseen peg geometries. By putting the estimators into the closed-loop control with reinforcement learning, we further achieve a higher or comparable success rate, efficiency, and robustness compared with the baseline methods. The policy is trained totally in simulation without any manual intervention. To achieve sim-to-real, a learnable segmentation module with automatic data collecting and labeling can be easily trained to decouple the perception and the policy, which helps the model trained in simulation quickly adapting to the real world with negligible effort. Results are presented in simulation and on a physical robot.</p>


## Downloads

- Video: [[YouTube](https://www.youtube.com/watch?v=L5AhgDvevKA)][[bilibili](https://www.bilibili.com/video/BV1Zf4y1w7ea?spm_id_from=333.999.0.0)]
- PDF: [[Supplementary](https://xieliang555.github.io/post/text/icra_supplementary.pdf)]
