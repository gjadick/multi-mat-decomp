
# Inputs

Multi-mat decomp was done using VMIs constructed from an initial two-mat decomp,
with different basis materials:
    matdecomp1* - tissue + bone VMI basis materials
    matdecomp2* - water + aluminum VMI basis materials

`old_matdecomp` was accidentally computed using water + aluminum basis material images
but with VMIs computed using the mass attenuation coefficients for tissue and bone.

An equal dose was used for both 80kVp and 140kVp acquisitions and is denoted in micro Gy.
This is the "per acquisition" dose. The total dual-energy CT dose is twice this value.
