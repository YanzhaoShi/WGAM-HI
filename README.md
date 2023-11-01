# WGAM-HI

Official code for "Weakly Guided Attention Model with Hierarchical Interaction for Brain CT Report Generation" (CIBM, 2023)

In this paper, we propose a novel Weakly Guided Attention Model with Hierarchical Interaction, named WGAM-HI, to improve Brain CT report generation.

## Environment
<pre>
  <code>
    python=3.6.9
    pytorch=1.10.0
    torchvision=0.11.1
  </code>
</pre>


## Training and Testing
The source code for training is tools/train.py. To run this code, please use the following command:
<pre>
  <code>
    python tools/train.py --checkpoint_path output/model
  </code>
</pre>

The source code for testing is tools/test.py. To run this code, please use the following command:
<pre>
  <code>
    python tools/eval.py
  </code>
</pre>

## Contact
If you are interested in work or have any questions, please connect us: zhangxiaodan@bjut.edu.cn

## Acknowledges
We thank <a href="https://github.com/ruotianluo/ImageCaptioning.pytorch">Up-Down</a> for their open source works.
