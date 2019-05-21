# SGL_v1

MATLAB implementation of Synergistic Generic Learning (SGL) [1]

[1] M. Pang, Y.M. Cheung, B. Wang, and J. Lou, “Synergistic Generic Learning for Face Recognition from a Contaminated Single Sample per Person”, IEEE Transactions on Information Forensics and Security, 2019.

This package provides the codes of: 
1) SGL w/o PL for SSPP with a standard biometric enrolment database (SSPP-se), run testSGL_ssppse.m; 
The AR_VariationDictionary_SGL.mat can be downloaded in https://drive.google.com/open?id=11_JKSC6iDo2vObCxsmgRMLEmgjov7qno

2) Variation dictionary learning, run GenerateVariation_SGL.m

The codes for the prototype learning are under optimization and will be released soon in the next version of code package. If you wish to get the full codes of SGL or have any other questions, please contact the author with the email: mengpang@comp.hkbu.edu.hk or pangmeng@mail.dlut.edu.cn. 

The software is free for academic use, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.



Notes:

Contributions:

1. To the best of our knowledge, this work is the first attmpt to study the new and practical problem in SSPP FR, i.e., SSPP-ce FR, where the biometric enrolment database is contaminated by nuisance facial variations.
2. We develop a learned prototypes plus learned variation dictionary (learned P+learned V) framework to address the SSPP-ce FR problem. Moreover, under this framework, we propose the SGL method.
3. We present a new way to learn the variation dictionary by extracting the less discriminative parts (LDPs) from an auxiliary generic set, and use low-rank factorization to solve it efficiently.


Limitations:

1. SGL aims to learns netural prototypes for contaminated enrolment samples by separating the nuisance variations and preserving the more discriminative subject-specific portions. Since this learning process is based on the linear-based feature regrouping, some nonlinear facial variations (e.g., expressions and poses) cannot be successfully removed under the circumstances.
2. The trained prototypes using the generic set may not represent the target enrolment persons correctly, especially in the case when some crucial regions (e.g., eyes) are corrupted. This is because that the individual information of the enrolment persons are probably not included in the generic set.  


© 2019 GitHub, Inc.
