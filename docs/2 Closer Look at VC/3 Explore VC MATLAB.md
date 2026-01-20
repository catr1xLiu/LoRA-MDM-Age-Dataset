>[!info|right]
> **Part of a series on**
> ###### [[Dataset Overview|Van Criekinge]]
> ---
> **Analysis**
> * [[2 Closer Look at VC/1 Current Pipeline Failures|Pipeline Failures]]
> * [[2 Closer Look at VC/2 Explore VC C3D|C3D Exploration]]
> * [[2 Closer Look at VC/3 Explore VC MATLAB|MATLAB Exploration]]

>[!info]
> ###### MATLAB Exploration
> [Type::Data Investigation]
> [Tool::MATLAB / Octave]
> [Format::.mat]

This document details the **exploration of the legacy MATLAB file structure** (`.mat`) from the Van Criekinge dataset. It documents the hierarchy of structs, fields, and parameters extracted using Octave/MATLAB.


## Data Structure Analysis
We inspected the file `MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat`.

### Top-Level Object
The file contains a single large struct array `Sub` (1x138), consuming ~8GB.

```octave
>> load('MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat')
>> whos
  Name      Size                  Bytes  Class     Attributes

  Sub       1x138            8098475331  struct
```

### Subject Structure (`Sub`)
Each element in `Sub` represents a subject and contains 9 fields organizing the data by body side and segment.

```octave
>> Sub(1)

ans = 

  struct with fields:

    LsideSegm_BsideData: [1×1 struct]  % Left Side Segment, Both Sides Data
    LsideSegm_LsideData: [1×1 struct]  % Left Side Segment, Left Side Data
    LsideSegm_RsideData: [1×1 struct]  % Left Side Segment, Right Side Data
                 events: [1×1 struct]  % Gait Events (Heel Strike, Toe Off)
    RsideSegm_BsideData: [1×1 struct]
    RsideSegm_LsideData: [1×1 struct]
    RsideSegm_RsideData: [1×1 struct]
              meas_char: [1×1 struct]  % Measurement Characteristics
               sub_char: [1×1 struct]  % Subject Characteristics
```

## Detailed Field Inspection
We inspected the nested structures to understand the data schema.


```octave fold:"Struts within s"
>> s.sub_char

ans = 

  struct with fields:

       Weight: 63.6000
          Age: 86
       Height: 1580
         Male: 1
    LegLength: 850

>> s.meas_char

ans = 

  struct with fields:

        Direction_L: [-1 -1 -1 1 1 -1 -1]
        Direction_R: [-1 -1 1 1 -1 -1]
    AnalogFrameRate: 1000
     VideoFrameRate: 100

>> s.events

ans = 

  struct with fields:

                   L_IC_cnt: [79.0000 192.0000 297.0000 151.0000 254.0000 101.0000 208.0000]
               L_IC_cntnorm: [522.9358 549.5496 486.7256 528.3019 504.7620 485.7144 504.7616]
                   L_TO_cnt: [31.0000 140.0000 254.0000 107.0000 213.0000 59.0000 166.0000]
               L_TO_cntnorm: [82.5687 81.0812 106.1946 113.2074 114.2856 85.7144 104.7616]
                    L_Trial: [4 4 4 2 2 3 3]
    L_GoodForcePlateLanding: [0 0 0 1 0 0 1]
                   R_IC_cnt: [131.0000 242.0000 95.0000 201.0000 155.0000 260.0000]
               R_IC_cntnorm: [460.1769 476.1906 486.2385 485.4368 504.6730 504.8545]
                   R_TO_cnt: [89.0000 201.0000 56.0000 162.0000 114.0000 219.0000]
               R_TO_cntnorm: [88.4955 85.7140 128.4403 106.7962 121.4955 106.7966]
                    R_Trial: [4 4 2 2 3 3]
    R_GoodForcePlateLanding: [0 0 1 0 0 1]
                       L_TO: [92 202 315 164 269 115 220]
                   L_TOnorm: [642.2018 639.6395 646.0178 650.9434 647.6189 619.0477 619.0475]
                  L_ICstart: [22.0000 131.0000 242.0000 95.0000 201.0000 50.0000 154.0000]
                   L_ICstop: [131.0000 242.0000 355.0000 201.0000 306.0000 155.0000 260.0000]
                       R_TO: [145 256 109 215 168 271]
                   R_TOnorm: [584.0707 609.5238 614.6789 621.3591 626.1684 611.6504]
                  R_ICstart: [79.0000 192.0000 41.0000 151.0000 101.0000 207.0000]
                   R_ICstop: [192.0000 297.0000 151.0000 254.0000 208.0000 311.0000]

>> s.LsideSegm_BsideData

ans = 

  struct with fields:

            HEDO: [1×1 struct]
            HEDA: [1×1 struct]
            HEDL: [1×1 struct]
            HEDP: [1×1 struct]
            PELO: [1×1 struct]
            PELA: [1×1 struct]
            PELL: [1×1 struct]
            PELP: [1×1 struct]
            TRXO: [1×1 struct]
            TRXA: [1×1 struct]
            TRXL: [1×1 struct]
            TRXP: [1×1 struct]
              C7: [1×1 struct]
             T10: [1×1 struct]
            CLAV: [1×1 struct]
            STRN: [1×1 struct]
            SACR: [1×1 struct]
    CentreOfMass: [1×1 struct]

>> s.LsideSegm_LsideData

ans = 

  struct with fields:

                     CLO: [1×1 struct]
                     CLA: [1×1 struct]
                     CLL: [1×1 struct]
                     CLP: [1×1 struct]
                     FEO: [1×1 struct]
                     FEA: [1×1 struct]
                     FEL: [1×1 struct]
                     FEP: [1×1 struct]
                     FOO: [1×1 struct]
                     FOA: [1×1 struct]
                     FOL: [1×1 struct]
                     FOP: [1×1 struct]
                     HNO: [1×1 struct]
                     HNA: [1×1 struct]
                     HNL: [1×1 struct]
                     HNP: [1×1 struct]
                     HUO: [1×1 struct]
                     HUA: [1×1 struct]
                     HUL: [1×1 struct]
                     HUP: [1×1 struct]
                     RAO: [1×1 struct]
                     RAA: [1×1 struct]
                     RAL: [1×1 struct]
                     RAP: [1×1 struct]
                     TIO: [1×1 struct]
                     TIA: [1×1 struct]
                     TIL: [1×1 struct]
                     TIP: [1×1 struct]
                     TOO: [1×1 struct]
                     TOA: [1×1 struct]
                     TOL: [1×1 struct]
                     TOP: [1×1 struct]
                     FHD: [1×1 struct]
                     BHD: [1×1 struct]
                     SHO: [1×1 struct]
                     ELB: [1×1 struct]
                     WRA: [1×1 struct]
                     WRB: [1×1 struct]
                     FIN: [1×1 struct]
                     ASI: [1×1 struct]
                     THI: [1×1 struct]
                     KNE: [1×1 struct]
                     TIB: [1×1 struct]
                     ANK: [1×1 struct]
                     HEE: [1×1 struct]
                     TOE: [1×1 struct]
               HipAngles: [1×1 struct]
              KneeAngles: [1×1 struct]
             AnkleAngles: [1×1 struct]
            PelvisAngles: [1×1 struct]
      FootProgressAngles: [1×1 struct]
          ShoulderAngles: [1×1 struct]
             ElbowAngles: [1×1 struct]
             WristAngles: [1×1 struct]
              NeckAngles: [1×1 struct]
             SpineAngles: [1×1 struct]
              HeadAngles: [1×1 struct]
            ThoraxAngles: [1×1 struct]
              AnklePower: [1×1 struct]
               KneePower: [1×1 struct]
                HipPower: [1×1 struct]
              WaistPower: [1×1 struct]
               NeckPower: [1×1 struct]
           ShoulderPower: [1×1 struct]
              ElbowPower: [1×1 struct]
              WristPower: [1×1 struct]
     GroundReactionForce: [1×1 struct]
              AnkleForce: [1×1 struct]
               KneeForce: [1×1 struct]
                HipForce: [1×1 struct]
              WaistForce: [1×1 struct]
               NeckForce: [1×1 struct]
           ShoulderForce: [1×1 struct]
              ElbowForce: [1×1 struct]
              WristForce: [1×1 struct]
    GroundReactionMoment: [1×1 struct]
             AnkleMoment: [1×1 struct]
              KneeMoment: [1×1 struct]
               HipMoment: [1×1 struct]
             WaistMoment: [1×1 struct]
              NeckMoment: [1×1 struct]
          ShoulderMoment: [1×1 struct]
             ElbowMoment: [1×1 struct]
             WristMoment: [1×1 struct]
                     GAS: [1×1 struct]
                      RF: [1×1 struct]
                      VL: [1×1 struct]
                      BF: [1×1 struct]
                      ST: [1×1 struct]
                      TA: [1×1 struct]
                     ERS: [1×1 struct]
                 GASnorm: [1×1 struct]
                  RFnorm: [1×1 struct]
                  VLnorm: [1×1 struct]
                  BFnorm: [1×1 struct]
                  STnorm: [1×1 struct]
                  TAnorm: [1×1 struct]
                 ERSnorm: [1×1 struct]

>> s.LsideSegm_RsideData

ans = 

  struct with fields:

                     CLO: [1×1 struct]
                     CLA: [1×1 struct]
                     CLL: [1×1 struct]
                     CLP: [1×1 struct]
                     FEO: [1×1 struct]
                     FEA: [1×1 struct]
                     FEL: [1×1 struct]
                     FEP: [1×1 struct]
                     FOO: [1×1 struct]
                     FOA: [1×1 struct]
                     FOL: [1×1 struct]
                     FOP: [1×1 struct]
                     HNO: [1×1 struct]
                     HNA: [1×1 struct]
                     HNL: [1×1 struct]
                     HNP: [1×1 struct]
                     HUO: [1×1 struct]
                     HUA: [1×1 struct]
                     HUL: [1×1 struct]
                     HUP: [1×1 struct]
                     RAO: [1×1 struct]
                     RAA: [1×1 struct]
                     RAL: [1×1 struct]
                     RAP: [1×1 struct]
                     TIO: [1×1 struct]
                     TIA: [1×1 struct]
                     TIL: [1×1 struct]
                     TIP: [1×1 struct]
                     TOO: [1×1 struct]
                     TOA: [1×1 struct]
                     TOL: [1×1 struct]
                     TOP: [1×1 struct]
                     FHD: [1×1 struct]
                     BHD: [1×1 struct]
                     SHO: [1×1 struct]
                     ELB: [1×1 struct]
                     WRA: [1×1 struct]
                     WRB: [1×1 struct]
                     FIN: [1×1 struct]
                     ASI: [1×1 struct]
                     THI: [1×1 struct]
                     KNE: [1×1 struct]
                     TIB: [1×1 struct]
                     ANK: [1×1 struct]
                     HEE: [1×1 struct]
                     TOE: [1×1 struct]
               HipAngles: [1×1 struct]
              KneeAngles: [1×1 struct]
             AnkleAngles: [1×1 struct]
            PelvisAngles: [1×1 struct]
      FootProgressAngles: [1×1 struct]
          ShoulderAngles: [1×1 struct]
             ElbowAngles: [1×1 struct]
             WristAngles: [1×1 struct]
              NeckAngles: [1×1 struct]
             SpineAngles: [1×1 struct]
              HeadAngles: [1×1 struct]
            ThoraxAngles: [1×1 struct]
              AnklePower: [1×1 struct]
               KneePower: [1×1 struct]
                HipPower: [1×1 struct]
              WaistPower: [1×1 struct]
               NeckPower: [1×1 struct]
           ShoulderPower: [1×1 struct]
              ElbowPower: [1×1 struct]
              WristPower: [1×1 struct]
     GroundReactionForce: [1×1 struct]
              AnkleForce: [1×1 struct]
               KneeForce: [1×1 struct]
                HipForce: [1×1 struct]
              WaistForce: [1×1 struct]
               NeckForce: [1×1 struct]
           ShoulderForce: [1×1 struct]
              ElbowForce: [1×1 struct]
              WristForce: [1×1 struct]
    GroundReactionMoment: [1×1 struct]
             AnkleMoment: [1×1 struct]
              KneeMoment: [1×1 struct]
               HipMoment: [1×1 struct]
             WaistMoment: [1×1 struct]
              NeckMoment: [1×1 struct]
          ShoulderMoment: [1×1 struct]
             ElbowMoment: [1×1 struct]
             WristMoment: [1×1 struct]
                     GAS: [1×1 struct]
                      RF: [1×1 struct]
                      VL: [1×1 struct]
                      BF: [1×1 struct]
                      ST: [1×1 struct]
                      TA: [1×1 struct]
                     ERS: [1×1 struct]
                 GASnorm: [1×1 struct]
                  RFnorm: [1×1 struct]
                  VLnorm: [1×1 struct]
                  BFnorm: [1×1 struct]
                  STnorm: [1×1 struct]
                  TAnorm: [1×1 struct]
                 ERSnorm: [1×1 struct]

>> s.RsideSegm_BsideData

ans = 

  struct with fields:

            HEDO: [1×1 struct]
            HEDA: [1×1 struct]
            HEDL: [1×1 struct]
            HEDP: [1×1 struct]
            PELO: [1×1 struct]
            PELA: [1×1 struct]
            PELL: [1×1 struct]
            PELP: [1×1 struct]
            TRXO: [1×1 struct]
            TRXA: [1×1 struct]
            TRXL: [1×1 struct]
            TRXP: [1×1 struct]
              C7: [1×1 struct]
             T10: [1×1 struct]
            CLAV: [1×1 struct]
            STRN: [1×1 struct]
            SACR: [1×1 struct]
    CentreOfMass: [1×1 struct]

>> s.RsideSegm_LsideData

ans = 

  struct with fields:

                     CLO: [1×1 struct]
                     CLA: [1×1 struct]
                     CLL: [1×1 struct]
                     CLP: [1×1 struct]
                     FEO: [1×1 struct]
                     FEA: [1×1 struct]
                     FEL: [1×1 struct]
                     FEP: [1×1 struct]
                     FOO: [1×1 struct]
                     FOA: [1×1 struct]
                     FOL: [1×1 struct]
                     FOP: [1×1 struct]
                     HNO: [1×1 struct]
                     HNA: [1×1 struct]
                     HNL: [1×1 struct]
                     HNP: [1×1 struct]
                     HUO: [1×1 struct]
                     HUA: [1×1 struct]
                     HUL: [1×1 struct]
                     HUP: [1×1 struct]
                     RAO: [1×1 struct]
                     RAA: [1×1 struct]
                     RAL: [1×1 struct]
                     RAP: [1×1 struct]
                     TIO: [1×1 struct]
                     TIA: [1×1 struct]
                     TIL: [1×1 struct]
                     TIP: [1×1 struct]
                     TOO: [1×1 struct]
                     TOA: [1×1 struct]
                     TOL: [1×1 struct]
                     TOP: [1×1 struct]
                     FHD: [1×1 struct]
                     BHD: [1×1 struct]
                     SHO: [1×1 struct]
                     ELB: [1×1 struct]
                     WRA: [1×1 struct]
                     WRB: [1×1 struct]
                     FIN: [1×1 struct]
                     ASI: [1×1 struct]
                     THI: [1×1 struct]
                     KNE: [1×1 struct]
                     TIB: [1×1 struct]
                     ANK: [1×1 struct]
                     HEE: [1×1 struct]
                     TOE: [1×1 struct]
               HipAngles: [1×1 struct]
              KneeAngles: [1×1 struct]
             AnkleAngles: [1×1 struct]
            PelvisAngles: [1×1 struct]
      FootProgressAngles: [1×1 struct]
          ShoulderAngles: [1×1 struct]
             ElbowAngles: [1×1 struct]
             WristAngles: [1×1 struct]
              NeckAngles: [1×1 struct]
             SpineAngles: [1×1 struct]
              HeadAngles: [1×1 struct]
            ThoraxAngles: [1×1 struct]
              AnklePower: [1×1 struct]
               KneePower: [1×1 struct]
                HipPower: [1×1 struct]
              WaistPower: [1×1 struct]
               NeckPower: [1×1 struct]
           ShoulderPower: [1×1 struct]
              ElbowPower: [1×1 struct]
              WristPower: [1×1 struct]
     GroundReactionForce: [1×1 struct]
              AnkleForce: [1×1 struct]
               KneeForce: [1×1 struct]
                HipForce: [1×1 struct]
              WaistForce: [1×1 struct]
               NeckForce: [1×1 struct]
           ShoulderForce: [1×1 struct]
              ElbowForce: [1×1 struct]
              WristForce: [1×1 struct]
    GroundReactionMoment: [1×1 struct]
             AnkleMoment: [1×1 struct]
              KneeMoment: [1×1 struct]
               HipMoment: [1×1 struct]
             WaistMoment: [1×1 struct]
              NeckMoment: [1×1 struct]
          ShoulderMoment: [1×1 struct]
             ElbowMoment: [1×1 struct]
             WristMoment: [1×1 struct]
                     GAS: [1×1 struct]
                      RF: [1×1 struct]
                      VL: [1×1 struct]
                      BF: [1×1 struct]
                      ST: [1×1 struct]
                      TA: [1×1 struct]
                     ERS: [1×1 struct]
                 GASnorm: [1×1 struct]
                  RFnorm: [1×1 struct]
                  VLnorm: [1×1 struct]
                  BFnorm: [1×1 struct]
                  STnorm: [1×1 struct]
                  TAnorm: [1×1 struct]
                 ERSnorm: [1×1 struct]

>> s.RsideSegm_RsideData

ans = 

  struct with fields:

                     CLO: [1×1 struct]
                     CLA: [1×1 struct]
                     CLL: [1×1 struct]
                     CLP: [1×1 struct]
                     FEO: [1×1 struct]
                     FEA: [1×1 struct]
                     FEL: [1×1 struct]
                     FEP: [1×1 struct]
                     FOO: [1×1 struct]
                     FOA: [1×1 struct]
                     FOL: [1×1 struct]
                     FOP: [1×1 struct]
                     HNO: [1×1 struct]
                     HNA: [1×1 struct]
                     HNL: [1×1 struct]
                     HNP: [1×1 struct]
                     HUO: [1×1 struct]
                     HUA: [1×1 struct]
                     HUL: [1×1 struct]
                     HUP: [1×1 struct]
                     RAO: [1×1 struct]
                     RAA: [1×1 struct]
                     RAL: [1×1 struct]
                     RAP: [1×1 struct]
                     TIO: [1×1 struct]
                     TIA: [1×1 struct]
                     TIL: [1×1 struct]
                     TIP: [1×1 struct]
                     TOO: [1×1 struct]
                     TOA: [1×1 struct]
                     TOL: [1×1 struct]
                     TOP: [1×1 struct]
                     FHD: [1×1 struct]
                     BHD: [1×1 struct]
                     SHO: [1×1 struct]
                     ELB: [1×1 struct]
                     WRA: [1×1 struct]
                     WRB: [1×1 struct]
                     FIN: [1×1 struct]
                     ASI: [1×1 struct]
                     THI: [1×1 struct]
                     KNE: [1×1 struct]
                     TIB: [1×1 struct]
                     ANK: [1×1 struct]
                     HEE: [1×1 struct]
                     TOE: [1×1 struct]
               HipAngles: [1×1 struct]
              KneeAngles: [1×1 struct]
             AnkleAngles: [1×1 struct]
            PelvisAngles: [1×1 struct]
      FootProgressAngles: [1×1 struct]
          ShoulderAngles: [1×1 struct]
             ElbowAngles: [1×1 struct]
             WristAngles: [1×1 struct]
              NeckAngles: [1×1 struct]
             SpineAngles: [1×1 struct]
              HeadAngles: [1×1 struct]
            ThoraxAngles: [1×1 struct]
              AnklePower: [1×1 struct]
               KneePower: [1×1 struct]
                HipPower: [1×1 struct]
              WaistPower: [1×1 struct]
               NeckPower: [1×1 struct]
           ShoulderPower: [1×1 struct]
              ElbowPower: [1×1 struct]
              WristPower: [1×1 struct]
     GroundReactionForce: [1×1 struct]
              AnkleForce: [1×1 struct]
               KneeForce: [1×1 struct]
                HipForce: [1×1 struct]
              WaistForce: [1×1 struct]
               NeckForce: [1×1 struct]
           ShoulderForce: [1×1 struct]
              ElbowForce: [1×1 struct]
              WristForce: [1×1 struct]
    GroundReactionMoment: [1×1 struct]
             AnkleMoment: [1×1 struct]
              KneeMoment: [1×1 struct]
               HipMoment: [1×1 struct]
             WaistMoment: [1×1 struct]
              NeckMoment: [1×1 struct]
          ShoulderMoment: [1×1 struct]
             ElbowMoment: [1×1 struct]
             WristMoment: [1×1 struct]
                     GAS: [1×1 struct]
                      RF: [1×1 struct]
                      VL: [1×1 struct]
                      BF: [1×1 struct]
                      ST: [1×1 struct]
                      TA: [1×1 struct]
                     ERS: [1×1 struct]
                 GASnorm: [1×1 struct]
                  RFnorm: [1×1 struct]
                  VLnorm: [1×1 struct]
                  BFnorm: [1×1 struct]
                  STnorm: [1×1 struct]
                  TAnorm: [1×1 struct]
                 ERSnorm: [1×1 struct]
```

These are the Fields under each strut. Interesting.

Inspecting 2 random files gives this.


```octave fold:s.LsideSegm_BsideData.HEDO.x
>> s.LsideSegm_BsideData.HEDO.x

ans =

   1.0e+03 *

   -2.0154   -0.9182    0.2034   -0.0854    1.0732   -1.7937   -0.6489
   -2.0142   -0.9170    0.2046   -0.0841    1.0745   -1.7923   -0.6476
   -2.0131   -0.9157    0.2058   -0.0829    1.0757   -1.7910   -0.6463
   -2.0120   -0.9145    0.2071   -0.0816    1.0770   -1.7897   -0.6450
   -2.0108   -0.9133    0.2083   -0.0803    1.0782   -1.7883   -0.6437
   -2.0097   -0.9121    0.2095   -0.0790    1.0795   -1.7870   -0.6424
   -2.0085   -0.9109    0.2107   -0.0778    1.0807   -1.7856   -0.6411
   -2.0074   -0.9096    0.2119   -0.0765    1.0820   -1.7843   -0.6398
   -2.0062   -0.9084    0.2132   -0.0752    1.0832   -1.7829   -0.6385
   -2.0051   -0.9072    0.2144   -0.0740    1.0845   -1.7816   -0.6372
   -2.0039   -0.9060    0.2156   -0.0727    1.0857   -1.7802   -0.6359
   -2.0028   -0.9047    0.2168   -0.0714    1.0870   -1.7789   -0.6345
   -2.0017   -0.9035    0.2181   -0.0701    1.0882   -1.7775   -0.6332
   -2.0005   -0.9023    0.2193   -0.0688    1.0895   -1.7762   -0.6319
   -1.9993   -0.9010    0.2205   -0.0676    1.0907   -1.7748   -0.6306
   -1.9982   -0.8998    0.2217   -0.0663    1.0920   -1.7735   -0.6293
   -1.9970   -0.8986    0.2230   -0.0650    1.0933   -1.7721   -0.6280
   -1.9959   -0.8973    0.2242   -0.0637    1.0945   -1.7708   -0.6267
   -1.9947   -0.8961    0.2254   -0.0624    1.0958   -1.7694   -0.6254
   -1.9936   -0.8949    0.2267   -0.0611    1.0970   -1.7680   -0.6241
   -1.9924   -0.8936    0.2279   -0.0599    1.0983   -1.7667   -0.6228
   -1.9913   -0.8924    0.2291   -0.0586    1.0996   -1.7653   -0.6214
   -1.9901   -0.8912    0.2304   -0.0573    1.1008   -1.7640   -0.6201
   -1.9890   -0.8899    0.2316   -0.0560    1.1021   -1.7626   -0.6188
   -1.9878   -0.8887    0.2328   -0.0547    1.1033   -1.7613   -0.6175
   -1.9866   -0.8874    0.2341   -0.0534    1.1046   -1.7599   -0.6162
   -1.9855   -0.8862    0.2353   -0.0521    1.1059   -1.7586   -0.6149
   -1.9843   -0.8850    0.2365   -0.0508    1.1071   -1.7572   -0.6135
   -1.9832   -0.8837    0.2378   -0.0496    1.1084   -1.7558   -0.6122
   -1.9820   -0.8825    0.2390   -0.0483    1.1097   -1.7545   -0.6109
   -1.9808   -0.8812    0.2402   -0.0470    1.1109   -1.7531   -0.6096
   -1.9797   -0.8800    0.2415   -0.0457    1.1122   -1.7518   -0.6083
   -1.9785   -0.8788    0.2427   -0.0444    1.1135   -1.7504   -0.6070
   -1.9774   -0.8775    0.2440   -0.0431    1.1147   -1.7491   -0.6056
   -1.9762   -0.8763    0.2452   -0.0418    1.1160   -1.7477   -0.6043
   -1.9750   -0.8750    0.2464   -0.0405    1.1173   -1.7464   -0.6030
   -1.9739   -0.8738    0.2477   -0.0392    1.1185   -1.7450   -0.6017
   -1.9727   -0.8726    0.2489   -0.0379    1.1198   -1.7436   -0.6004
   -1.9716   -0.8713    0.2501   -0.0367    1.1210   -1.7423   -0.5991
   -1.9704   -0.8701    0.2514   -0.0354    1.1223   -1.7409   -0.5977
   -1.9692   -0.8689    0.2526   -0.0341    1.1236   -1.7396   -0.5964
   -1.9681   -0.8676    0.2538   -0.0328    1.1248   -1.7382   -0.5951
   -1.9669   -0.8664    0.2551   -0.0315    1.1261   -1.7369   -0.5938
   -1.9658   -0.8651    0.2563   -0.0302    1.1274   -1.7356   -0.5925
   -1.9646   -0.8639    0.2575   -0.0289    1.1286   -1.7342   -0.5912
   -1.9634   -0.8627    0.2587   -0.0276    1.1299   -1.7329   -0.5899
   -1.9623   -0.8614    0.2600   -0.0264    1.1312   -1.7315   -0.5885
   -1.9611   -0.8602    0.2612   -0.0251    1.1324   -1.7302   -0.5872
   -1.9600   -0.8590    0.2624   -0.0238    1.1337   -1.7288   -0.5859
   -1.9588   -0.8577    0.2636   -0.0225    1.1349   -1.7275   -0.5846
   -1.9577   -0.8565    0.2649   -0.0212    1.1362   -1.7262   -0.5833
   -1.9565   -0.8553    0.2661   -0.0199    1.1375   -1.7248   -0.5820
   -1.9553   -0.8540    0.2673   -0.0187    1.1387   -1.7235   -0.5807
   -1.9542   -0.8528    0.2685   -0.0174    1.1400   -1.7221   -0.5794
   -1.9530   -0.8516    0.2698   -0.0161    1.1412   -1.7208   -0.5781
   -1.9519   -0.8503    0.2710   -0.0148    1.1425   -1.7195   -0.5768
   -1.9507   -0.8491    0.2722   -0.0135    1.1437   -1.7181   -0.5755
   -1.9496   -0.8479    0.2734   -0.0123    1.1450   -1.7168   -0.5742
   -1.9484   -0.8467    0.2746   -0.0110    1.1462   -1.7155   -0.5729
   -1.9473   -0.8454    0.2759   -0.0097    1.1475   -1.7142   -0.5716
   -1.9461   -0.8442    0.2771   -0.0084    1.1488   -1.7128   -0.5703
   -1.9450   -0.8430    0.2783   -0.0072    1.1500   -1.7115   -0.5690
   -1.9438   -0.8417    0.2795   -0.0059    1.1513   -1.7102   -0.5677
   -1.9427   -0.8405    0.2807   -0.0046    1.1525   -1.7089   -0.5664
   -1.9415   -0.8393    0.2819   -0.0034    1.1538   -1.7075   -0.5651
   -1.9404   -0.8381    0.2831   -0.0021    1.1550   -1.7062   -0.5638
   -1.9392   -0.8369    0.2843   -0.0008    1.1563   -1.7049   -0.5625
   -1.9381   -0.8356    0.2856    0.0004    1.1575   -1.7036   -0.5612
   -1.9369   -0.8344    0.2868    0.0017    1.1587   -1.7023   -0.5599
   -1.9358   -0.8332    0.2880    0.0030    1.1600   -1.7010   -0.5586
   -1.9346   -0.8320    0.2892    0.0042    1.1612   -1.6997   -0.5573
   -1.9335   -0.8308    0.2904    0.0055    1.1625   -1.6983   -0.5560
   -1.9323   -0.8296    0.2916    0.0068    1.1637   -1.6970   -0.5548
   -1.9312   -0.8283    0.2928    0.0080    1.1650   -1.6957   -0.5535
   -1.9300   -0.8271    0.2940    0.0093    1.1662   -1.6944   -0.5522
   -1.9289   -0.8259    0.2952    0.0105    1.1674   -1.6931   -0.5509
   -1.9277   -0.8247    0.2964    0.0118    1.1687   -1.6918   -0.5496
   -1.9266   -0.8235    0.2976    0.0130    1.1699   -1.6905   -0.5483
   -1.9255   -0.8223    0.2988    0.0143    1.1712   -1.6892   -0.5471
   -1.9243   -0.8211    0.3000    0.0155    1.1724   -1.6879   -0.5458
   -1.9232   -0.8199    0.3012    0.0168    1.1736   -1.6866   -0.5445
   -1.9220   -0.8187    0.3024    0.0180    1.1749   -1.6853   -0.5432
   -1.9209   -0.8174    0.3035    0.0193    1.1761   -1.6840   -0.5420
   -1.9198   -0.8162    0.3047    0.0205    1.1773   -1.6828   -0.5407
   -1.9186   -0.8150    0.3059    0.0218    1.1786   -1.6815   -0.5394
   -1.9175   -0.8138    0.3071    0.0230    1.1798   -1.6802   -0.5382
   -1.9163   -0.8126    0.3083    0.0242    1.1810   -1.6789   -0.5369
   -1.9152   -0.8114    0.3095    0.0255    1.1822   -1.6776   -0.5356
   -1.9141   -0.8102    0.3107    0.0267    1.1835   -1.6763   -0.5343
   -1.9129   -0.8090    0.3119    0.0279    1.1847   -1.6751   -0.5331
   -1.9118   -0.8078    0.3130    0.0292    1.1859   -1.6738   -0.5318
   -1.9107   -0.8066    0.3142    0.0304    1.1871   -1.6725   -0.5306
   -1.9095   -0.8054    0.3154    0.0316    1.1884   -1.6712   -0.5293
   -1.9084   -0.8042    0.3166    0.0329    1.1896   -1.6699   -0.5280
   -1.9073   -0.8031    0.3177    0.0341    1.1908   -1.6687   -0.5268
   -1.9061   -0.8019    0.3189    0.0353    1.1920   -1.6674   -0.5255
   -1.9050   -0.8007    0.3201    0.0366    1.1933   -1.6661   -0.5243
   -1.9039   -0.7995    0.3213    0.0378    1.1945   -1.6649   -0.5230
   -1.9028   -0.7983    0.3224    0.0390    1.1957   -1.6636   -0.5218
   -1.9016   -0.7971    0.3236    0.0402    1.1969   -1.6623   -0.5205
   -1.9005   -0.7959    0.3248    0.0414    1.1981   -1.6611   -0.5192
   -1.8994   -0.7947    0.3259    0.0427    1.1993   -1.6598   -0.5180
   -1.8982   -0.7935    0.3271    0.0439    1.2006   -1.6586   -0.5167
   -1.8971   -0.7924    0.3283    0.0451    1.2018   -1.6573   -0.5155
   -1.8960   -0.7912    0.3294    0.0463    1.2030   -1.6560   -0.5142
   -1.8949   -0.7900    0.3306    0.0475    1.2042   -1.6548   -0.5130
   -1.8937   -0.7888    0.3318    0.0487    1.2054   -1.6535   -0.5118
   -1.8926   -0.7876    0.3329    0.0499    1.2066   -1.6523   -0.5105
   -1.8915   -0.7865    0.3341    0.0512    1.2078   -1.6510   -0.5093
   -1.8904   -0.7853    0.3352    0.0524    1.2090   -1.6498   -0.5080
   -1.8893   -0.7841    0.3364    0.0536    1.2102   -1.6485   -0.5068
   -1.8881   -0.7829    0.3375    0.0548    1.2115   -1.6473   -0.5056
   -1.8870   -0.7818    0.3387    0.0560    1.2127   -1.6461   -0.5043
   -1.8859   -0.7806    0.3398    0.0572    1.2139   -1.6448   -0.5031
   -1.8848   -0.7794    0.3410    0.0584    1.2151   -1.6436   -0.5018
   -1.8837   -0.7783    0.3421    0.0596    1.2163   -1.6423   -0.5006
   -1.8826   -0.7771    0.3433    0.0608    1.2175   -1.6411   -0.4994
   -1.8814   -0.7759    0.3444    0.0620    1.2187   -1.6399   -0.4981
   -1.8803   -0.7748    0.3456    0.0632    1.2199   -1.6386   -0.4969
   -1.8792   -0.7736    0.3467    0.0644    1.2211   -1.6374   -0.4957
   -1.8781   -0.7725    0.3478    0.0655    1.2223   -1.6362   -0.4945
   -1.8770   -0.7713    0.3490    0.0667    1.2235   -1.6350   -0.4932
   -1.8759   -0.7701    0.3501    0.0679    1.2247   -1.6337   -0.4920
   -1.8748   -0.7690    0.3513    0.0691    1.2258   -1.6325   -0.4908
   -1.8737   -0.7678    0.3524    0.0703    1.2270   -1.6313   -0.4896
   -1.8726   -0.7667    0.3535    0.0715    1.2282   -1.6301   -0.4883
   -1.8715   -0.7655    0.3546    0.0727    1.2294   -1.6289   -0.4871
   -1.8704   -0.7644    0.3558    0.0739    1.2306   -1.6277   -0.4859
   -1.8693   -0.7632    0.3569    0.0750    1.2318   -1.6264   -0.4847
   -1.8682   -0.7621    0.3580    0.0762    1.2330   -1.6252   -0.4835
   -1.8671   -0.7609    0.3592    0.0774    1.2342   -1.6240   -0.4822
   -1.8659   -0.7598    0.3603    0.0786    1.2354   -1.6228   -0.4810
   -1.8648   -0.7586    0.3614    0.0797    1.2365   -1.6216   -0.4798
   -1.8637   -0.7575    0.3625    0.0809    1.2377   -1.6204   -0.4786
   -1.8627   -0.7563    0.3636    0.0821    1.2389   -1.6192   -0.4774
   -1.8616   -0.7552    0.3647    0.0833    1.2401   -1.6180   -0.4762
   -1.8605   -0.7541    0.3659    0.0844    1.2413   -1.6168   -0.4750
   -1.8594   -0.7529    0.3670    0.0856    1.2425   -1.6156   -0.4738
   -1.8583   -0.7518    0.3681    0.0868    1.2436   -1.6144   -0.4726
   -1.8572   -0.7507    0.3692    0.0879    1.2448   -1.6132   -0.4714
   -1.8561   -0.7495    0.3703    0.0891    1.2460   -1.6121   -0.4702
   -1.8550   -0.7484    0.3714    0.0903    1.2472   -1.6109   -0.4690
   -1.8539   -0.7473    0.3725    0.0914    1.2483   -1.6097   -0.4678
   -1.8528   -0.7461    0.3736    0.0926    1.2495   -1.6085   -0.4666
   -1.8517   -0.7450    0.3747    0.0937    1.2507   -1.6073   -0.4654
   -1.8506   -0.7439    0.3758    0.0949    1.2518   -1.6061   -0.4642
   -1.8495   -0.7428    0.3769    0.0961    1.2530   -1.6050   -0.4630
   -1.8485   -0.7416    0.3780    0.0972    1.2542   -1.6038   -0.4618
   -1.8474   -0.7405    0.3791    0.0984    1.2554   -1.6026   -0.4606
   -1.8463   -0.7394    0.3802    0.0995    1.2565   -1.6015   -0.4594
   -1.8452   -0.7383    0.3813    0.1007    1.2577   -1.6003   -0.4582
   -1.8441   -0.7372    0.3824    0.1018    1.2588   -1.5991   -0.4570
   -1.8431   -0.7361    0.3835    0.1030    1.2600   -1.5980   -0.4558
   -1.8420   -0.7349    0.3846    0.1041    1.2612   -1.5968   -0.4546
   -1.8409   -0.7338    0.3857    0.1053    1.2623   -1.5956   -0.4535
   -1.8398   -0.7327    0.3867    0.1064    1.2635   -1.5945   -0.4523
   -1.8387   -0.7316    0.3878    0.1075    1.2646   -1.5933   -0.4511
   -1.8377   -0.7305    0.3889    0.1087    1.2658   -1.5922   -0.4499
   -1.8366   -0.7294    0.3900    0.1098    1.2670   -1.5910   -0.4487
   -1.8355   -0.7283    0.3911    0.1110    1.2681   -1.5899   -0.4476
   -1.8344   -0.7272    0.3921    0.1121    1.2693   -1.5887   -0.4464
   -1.8334   -0.7261    0.3932    0.1132    1.2704   -1.5876   -0.4452
   -1.8323   -0.7250    0.3943    0.1144    1.2716   -1.5864   -0.4440
   -1.8312   -0.7239    0.3954    0.1155    1.2727   -1.5853   -0.4429
   -1.8302   -0.7228    0.3964    0.1166    1.2739   -1.5842   -0.4417
   -1.8291   -0.7217    0.3975    0.1178    1.2750   -1.5830   -0.4405
   -1.8280   -0.7206    0.3986    0.1189    1.2762   -1.5819   -0.4394
   -1.8270   -0.7195    0.3997    0.1200    1.2773   -1.5807   -0.4382
   -1.8259   -0.7184    0.4007    0.1211    1.2785   -1.5796   -0.4370
   -1.8248   -0.7173    0.4018    0.1223    1.2796   -1.5785   -0.4359
   -1.8238   -0.7162    0.4028    0.1234    1.2807   -1.5774   -0.4347
   -1.8227   -0.7151    0.4039    0.1245    1.2819   -1.5762   -0.4335
   -1.8216   -0.7140    0.4050    0.1256    1.2830   -1.5751   -0.4324
   -1.8206   -0.7129    0.4060    0.1268    1.2842   -1.5740   -0.4312
   -1.8195   -0.7118    0.4071    0.1279    1.2853   -1.5729   -0.4301
   -1.8185   -0.7108    0.4081    0.1290    1.2865   -1.5717   -0.4289
   -1.8174   -0.7097    0.4092    0.1301    1.2876   -1.5706   -0.4277
   -1.8163   -0.7086    0.4103    0.1312    1.2887   -1.5695   -0.4266
   -1.8153   -0.7075    0.4113    0.1323    1.2899   -1.5684   -0.4254
   -1.8142   -0.7064    0.4124    0.1334    1.2910   -1.5673   -0.4243
   -1.8132   -0.7054    0.4134    0.1345    1.2921   -1.5662   -0.4231
   -1.8121   -0.7043    0.4145    0.1356    1.2933   -1.5651   -0.4220
   -1.8111   -0.7032    0.4155    0.1368    1.2944   -1.5640   -0.4208
   -1.8100   -0.7021    0.4166    0.1379    1.2955   -1.5629   -0.4197
   -1.8090   -0.7010    0.4176    0.1390    1.2966   -1.5618   -0.4185
   -1.8079   -0.7000    0.4187    0.1401    1.2978   -1.5607   -0.4174
   -1.8069   -0.6989    0.4197    0.1412    1.2989   -1.5596   -0.4163
   -1.8058   -0.6978    0.4207    0.1423    1.3000   -1.5585   -0.4151
   -1.8048   -0.6967    0.4218    0.1434    1.3012   -1.5574   -0.4140
   -1.8037   -0.6957    0.4228    0.1445    1.3023   -1.5563   -0.4128
   -1.8027   -0.6946    0.4239    0.1456    1.3034   -1.5552   -0.4117
   -1.8016   -0.6935    0.4249    0.1466    1.3045   -1.5541   -0.4106
   -1.8006   -0.6925    0.4259    0.1477    1.3056   -1.5530   -0.4094
   -1.7995   -0.6914    0.4270    0.1488    1.3068   -1.5519   -0.4083
   -1.7985   -0.6903    0.4280    0.1499    1.3079   -1.5508   -0.4072
   -1.7974   -0.6893    0.4290    0.1510    1.3090   -1.5498   -0.4060
   -1.7964   -0.6882    0.4301    0.1521    1.3101   -1.5487   -0.4049
   -1.7954   -0.6871    0.4311    0.1532    1.3112   -1.5476   -0.4038
   -1.7943   -0.6861    0.4321    0.1543    1.3124   -1.5465   -0.4026
   -1.7933   -0.6850    0.4332    0.1553    1.3135   -1.5454   -0.4015
   -1.7922   -0.6840    0.4342    0.1564    1.3146   -1.5444   -0.4004
   -1.7912   -0.6829    0.4352    0.1575    1.3157   -1.5433   -0.3992
   -1.7902   -0.6818    0.4363    0.1586    1.3168   -1.5422   -0.3981
   -1.7891   -0.6808    0.4373    0.1597    1.3179   -1.5411   -0.3970
   -1.7881   -0.6797    0.4383    0.1607    1.3190   -1.5401   -0.3959
   -1.7870   -0.6787    0.4393    0.1618    1.3202   -1.5390   -0.3948
   -1.7860   -0.6776    0.4404    0.1629    1.3213   -1.5379   -0.3936
   -1.7850   -0.6766    0.4414    0.1640    1.3224   -1.5369   -0.3925
   -1.7839   -0.6755    0.4424    0.1650    1.3235   -1.5358   -0.3914
   -1.7829   -0.6745    0.4434    0.1661    1.3246   -1.5347   -0.3903
   -1.7819   -0.6734    0.4444    0.1672    1.3257   -1.5337   -0.3892
   -1.7808   -0.6723    0.4455    0.1683    1.3268   -1.5326   -0.3880
   -1.7798   -0.6713    0.4465    0.1693    1.3279   -1.5316   -0.3869
   -1.7788   -0.6702    0.4475    0.1704    1.3290   -1.5305   -0.3858
   -1.7777   -0.6692    0.4485    0.1715    1.3301   -1.5294   -0.3847
   -1.7767   -0.6681    0.4495    0.1725    1.3312   -1.5284   -0.3836
   -1.7757   -0.6671    0.4505    0.1736    1.3323   -1.5273   -0.3825
   -1.7746   -0.6660    0.4516    0.1746    1.3335   -1.5263   -0.3814
   -1.7736   -0.6650    0.4526    0.1757    1.3346   -1.5252   -0.3803
   -1.7726   -0.6639    0.4536    0.1768    1.3357   -1.5242   -0.3791
   -1.7716   -0.6629    0.4546    0.1778    1.3368   -1.5231   -0.3780
   -1.7705   -0.6619    0.4556    0.1789    1.3379   -1.5221   -0.3769
   -1.7695   -0.6608    0.4566    0.1799    1.3390   -1.5210   -0.3758
   -1.7685   -0.6598    0.4576    0.1810    1.3401   -1.5200   -0.3747
   -1.7675   -0.6587    0.4586    0.1821    1.3412   -1.5189   -0.3736
   -1.7664   -0.6577    0.4596    0.1831    1.3423   -1.5179   -0.3725
   -1.7654   -0.6566    0.4607    0.1842    1.3434   -1.5168   -0.3714
   -1.7644   -0.6556    0.4617    0.1852    1.3445   -1.5158   -0.3703
   -1.7634   -0.6545    0.4627    0.1863    1.3456   -1.5147   -0.3692
   -1.7623   -0.6535    0.4637    0.1873    1.3467   -1.5137   -0.3681
   -1.7613   -0.6525    0.4647    0.1884    1.3478   -1.5126   -0.3670
   -1.7603   -0.6514    0.4657    0.1894    1.3489   -1.5116   -0.3659
   -1.7593   -0.6504    0.4667    0.1905    1.3500   -1.5106   -0.3648
   -1.7582   -0.6493    0.4677    0.1915    1.3511   -1.5095   -0.3637
   -1.7572   -0.6483    0.4687    0.1926    1.3522   -1.5085   -0.3626
   -1.7562   -0.6472    0.4697    0.1936    1.3533   -1.5074   -0.3615
   -1.7552   -0.6462    0.4707    0.1947    1.3544   -1.5064   -0.3604
   -1.7542   -0.6452    0.4717    0.1957    1.3555   -1.5054   -0.3593
   -1.7531   -0.6441    0.4727    0.1968    1.3566   -1.5043   -0.3582
   -1.7521   -0.6431    0.4737    0.1978    1.3577   -1.5033   -0.3571
   -1.7511   -0.6420    0.4747    0.1988    1.3587   -1.5023   -0.3560
   -1.7501   -0.6410    0.4757    0.1999    1.3598   -1.5012   -0.3549
   -1.7491   -0.6400    0.4767    0.2009    1.3609   -1.5002   -0.3538
   -1.7480   -0.6389    0.4777    0.2020    1.3620   -1.4991   -0.3527
   -1.7470   -0.6379    0.4787    0.2030    1.3631   -1.4981   -0.3516
   -1.7460   -0.6369    0.4797    0.2041    1.3642   -1.4971   -0.3505
   -1.7450   -0.6358    0.4807    0.2051    1.3653   -1.4960   -0.3494
   -1.7440   -0.6348    0.4817    0.2061    1.3664   -1.4950   -0.3484
   -1.7429   -0.6337    0.4827    0.2072    1.3675   -1.4940   -0.3473
   -1.7419   -0.6327    0.4837    0.2082    1.3686   -1.4929   -0.3462
   -1.7409   -0.6317    0.4847    0.2093    1.3697   -1.4919   -0.3451
   -1.7399   -0.6306    0.4857    0.2103    1.3708   -1.4909   -0.3440
   -1.7389   -0.6296    0.4867    0.2113    1.3719   -1.4899   -0.3429
   -1.7378   -0.6285    0.4877    0.2124    1.3730   -1.4888   -0.3418
   -1.7368   -0.6275    0.4887    0.2134    1.3741   -1.4878   -0.3407
   -1.7358   -0.6265    0.4897    0.2144    1.3752   -1.4868   -0.3396
   -1.7348   -0.6254    0.4907    0.2155    1.3763   -1.4857   -0.3385
   -1.7338   -0.6244    0.4917    0.2165    1.3774   -1.4847   -0.3374
   -1.7328   -0.6234    0.4926    0.2175    1.3785   -1.4837   -0.3363
   -1.7317   -0.6223    0.4936    0.2186    1.3796   -1.4826   -0.3352
   -1.7307   -0.6213    0.4946    0.2196    1.3807   -1.4816   -0.3342
   -1.7297   -0.6202    0.4956    0.2207    1.3818   -1.4806   -0.3331
   -1.7287   -0.6192    0.4966    0.2217    1.3829   -1.4795   -0.3320
   -1.7277   -0.6182    0.4976    0.2227    1.3840   -1.4785   -0.3309
   -1.7267   -0.6171    0.4986    0.2238    1.3851   -1.4775   -0.3298
   -1.7257   -0.6161    0.4996    0.2248    1.3862   -1.4765   -0.3287
   -1.7246   -0.6151    0.5006    0.2258    1.3873   -1.4754   -0.3276
   -1.7236   -0.6140    0.5016    0.2269    1.3884   -1.4744   -0.3265
   -1.7226   -0.6130    0.5026    0.2279    1.3895   -1.4734   -0.3254
   -1.7216   -0.6119    0.5036    0.2289    1.3906   -1.4723   -0.3243
   -1.7206   -0.6109    0.5046    0.2300    1.3917   -1.4713   -0.3232
   -1.7196   -0.6099    0.5056    0.2310    1.3928   -1.4703   -0.3222
   -1.7185   -0.6088    0.5065    0.2320    1.3939   -1.4693   -0.3211
   -1.7175   -0.6078    0.5075    0.2331    1.3950   -1.4682   -0.3200
   -1.7165   -0.6067    0.5085    0.2341    1.3961   -1.4672   -0.3189
   -1.7155   -0.6057    0.5095    0.2351    1.3972   -1.4662   -0.3178
   -1.7145   -0.6047    0.5105    0.2362    1.3983   -1.4651   -0.3167
   -1.7135   -0.6036    0.5115    0.2372    1.3994   -1.4641   -0.3156
   -1.7124   -0.6026    0.5125    0.2382    1.4005   -1.4631   -0.3145
   -1.7114   -0.6015    0.5135    0.2393    1.4016   -1.4620   -0.3134
   -1.7104   -0.6005    0.5145    0.2403    1.4027   -1.4610   -0.3123
   -1.7094   -0.5994    0.5155    0.2413    1.4038   -1.4600   -0.3112
   -1.7084   -0.5984    0.5165    0.2424    1.4049   -1.4589   -0.3101
   -1.7074   -0.5974    0.5175    0.2434    1.4060   -1.4579   -0.3091
   -1.7063   -0.5963    0.5185    0.2445    1.4071   -1.4569   -0.3080
   -1.7053   -0.5953    0.5195    0.2455    1.4082   -1.4558   -0.3069
   -1.7043   -0.5942    0.5204    0.2465    1.4093   -1.4548   -0.3058
   -1.7033   -0.5932    0.5214    0.2476    1.4104   -1.4538   -0.3047
   -1.7023   -0.5921    0.5224    0.2486    1.4115   -1.4527   -0.3036
   -1.7013   -0.5911    0.5234    0.2496    1.4126   -1.4517   -0.3025
   -1.7002   -0.5900    0.5244    0.2507    1.4137   -1.4507   -0.3014
   -1.6992   -0.5890    0.5254    0.2517    1.4148   -1.4496   -0.3003
   -1.6982   -0.5880    0.5264    0.2527    1.4159   -1.4486   -0.2992
   -1.6972   -0.5869    0.5274    0.2538    1.4170   -1.4476   -0.2981
   -1.6962   -0.5859    0.5284    0.2548    1.4182   -1.4465   -0.2970
   -1.6951   -0.5848    0.5294    0.2559    1.4193   -1.4455   -0.2959
   -1.6941   -0.5838    0.5304    0.2569    1.4204   -1.4445   -0.2948
   -1.6931   -0.5827    0.5314    0.2579    1.4215   -1.4434   -0.2937
   -1.6921   -0.5817    0.5324    0.2590    1.4226   -1.4424   -0.2926
   -1.6911   -0.5806    0.5334    0.2600    1.4237   -1.4413   -0.2915
   -1.6900   -0.5796    0.5344    0.2611    1.4248   -1.4403   -0.2904
   -1.6890   -0.5785    0.5354    0.2621    1.4259   -1.4393   -0.2893
   -1.6880   -0.5775    0.5364    0.2631    1.4270   -1.4382   -0.2882
   -1.6870   -0.5764    0.5374    0.2642    1.4282   -1.4372   -0.2871
   -1.6860   -0.5754    0.5383    0.2652    1.4293   -1.4361   -0.2860
   -1.6849   -0.5743    0.5393    0.2663    1.4304   -1.4351   -0.2849
   -1.6839   -0.5733    0.5403    0.2673    1.4315   -1.4341   -0.2838
   -1.6829   -0.5722    0.5413    0.2684    1.4326   -1.4330   -0.2827
   -1.6819   -0.5711    0.5423    0.2694    1.4337   -1.4320   -0.2816
   -1.6808   -0.5701    0.5433    0.2705    1.4348   -1.4309   -0.2805
   -1.6798   -0.5690    0.5443    0.2715    1.4360   -1.4299   -0.2794
   -1.6788   -0.5680    0.5453    0.2725    1.4371   -1.4288   -0.2783
   -1.6778   -0.5669    0.5463    0.2736    1.4382   -1.4278   -0.2772
   -1.6767   -0.5659    0.5473    0.2746    1.4393   -1.4267   -0.2760
   -1.6757   -0.5648    0.5483    0.2757    1.4404   -1.4257   -0.2749
   -1.6747   -0.5637    0.5493    0.2767    1.4416   -1.4246   -0.2738
   -1.6737   -0.5627    0.5503    0.2778    1.4427   -1.4236   -0.2727
   -1.6726   -0.5616    0.5513    0.2788    1.4438   -1.4226   -0.2716
   -1.6716   -0.5606    0.5523    0.2799    1.4449   -1.4215   -0.2705
   -1.6706   -0.5595    0.5533    0.2809    1.4460   -1.4205   -0.2694
   -1.6696   -0.5584    0.5543    0.2820    1.4472   -1.4194   -0.2683
   -1.6685   -0.5574    0.5553    0.2831    1.4483   -1.4184   -0.2672
   -1.6675   -0.5563    0.5563    0.2841    1.4494   -1.4173   -0.2661
   -1.6665   -0.5553    0.5573    0.2852    1.4505   -1.4162   -0.2649
   -1.6654   -0.5542    0.5583    0.2862    1.4516   -1.4152   -0.2638
   -1.6644   -0.5531    0.5594    0.2873    1.4528   -1.4141   -0.2627
   -1.6634   -0.5521    0.5604    0.2883    1.4539   -1.4131   -0.2616
   -1.6623   -0.5510    0.5614    0.2894    1.4550   -1.4120   -0.2605
   -1.6613   -0.5499    0.5624    0.2904    1.4561   -1.4110   -0.2594
   -1.6603   -0.5489    0.5634    0.2915    1.4573   -1.4099   -0.2582
   -1.6592   -0.5478    0.5644    0.2926    1.4584   -1.4089   -0.2571
   -1.6582   -0.5467    0.5654    0.2936    1.4595   -1.4078   -0.2560
   -1.6572   -0.5456    0.5664    0.2947    1.4606   -1.4067   -0.2549
   -1.6561   -0.5446    0.5674    0.2957    1.4618   -1.4057   -0.2538
   -1.6551   -0.5435    0.5684    0.2968    1.4629   -1.4046   -0.2526
   -1.6541   -0.5424    0.5694    0.2979    1.4640   -1.4036   -0.2515
   -1.6530   -0.5414    0.5704    0.2989    1.4651   -1.4025   -0.2504
   -1.6520   -0.5403    0.5714    0.3000    1.4663   -1.4014   -0.2493
   -1.6510   -0.5392    0.5724    0.3011    1.4674   -1.4004   -0.2481
   -1.6499   -0.5381    0.5735    0.3021    1.4685   -1.3993   -0.2470
   -1.6489   -0.5371    0.5745    0.3032    1.4697   -1.3982   -0.2459
   -1.6478   -0.5360    0.5755    0.3043    1.4708   -1.3972   -0.2448
   -1.6468   -0.5349    0.5765    0.3053    1.4719   -1.3961   -0.2436
   -1.6458   -0.5338    0.5775    0.3064    1.4730   -1.3951   -0.2425
   -1.6447   -0.5327    0.5785    0.3075    1.4742   -1.3940   -0.2414
   -1.6437   -0.5317    0.5795    0.3085    1.4753   -1.3929   -0.2402
   -1.6426   -0.5306    0.5805    0.3096    1.4764   -1.3918   -0.2391
   -1.6416   -0.5295    0.5816    0.3107    1.4776   -1.3908   -0.2380
   -1.6406   -0.5284    0.5826    0.3118    1.4787   -1.3897   -0.2368
   -1.6395   -0.5273    0.5836    0.3128    1.4798   -1.3886   -0.2357
   -1.6385   -0.5262    0.5846    0.3139    1.4810   -1.3876   -0.2346
   -1.6374   -0.5252    0.5856    0.3150    1.4821   -1.3865   -0.2334
   -1.6364   -0.5241    0.5866    0.3161    1.4832   -1.3854   -0.2323
   -1.6353   -0.5230    0.5877    0.3171    1.4844   -1.3843   -0.2312
   -1.6343   -0.5219    0.5887    0.3182    1.4855   -1.3833   -0.2300
   -1.6332   -0.5208    0.5897    0.3193    1.4866   -1.3822   -0.2289
   -1.6322   -0.5197    0.5907    0.3204    1.4878   -1.3811   -0.2277
   -1.6312   -0.5186    0.5917    0.3214    1.4889   -1.3800   -0.2266
   -1.6301   -0.5175    0.5928    0.3225    1.4900   -1.3790   -0.2255
   -1.6291   -0.5164    0.5938    0.3236    1.4912   -1.3779   -0.2243
   -1.6280   -0.5154    0.5948    0.3247    1.4923   -1.3768   -0.2232
   -1.6270   -0.5143    0.5958    0.3258    1.4934   -1.3757   -0.2220
   -1.6259   -0.5132    0.5969    0.3269    1.4946   -1.3747   -0.2209
   -1.6248   -0.5121    0.5979    0.3279    1.4957   -1.3736   -0.2197
   -1.6238   -0.5110    0.5989    0.3290    1.4968   -1.3725   -0.2186
   -1.6227   -0.5099    0.5999    0.3301    1.4980   -1.3714   -0.2175
   -1.6217   -0.5088    0.6010    0.3312    1.4991   -1.3703   -0.2163
   -1.6206   -0.5077    0.6020    0.3323    1.5002   -1.3692   -0.2152
   -1.6196   -0.5066    0.6030    0.3334    1.5014   -1.3682   -0.2140
   -1.6185   -0.5055    0.6041    0.3345    1.5025   -1.3671   -0.2129
   -1.6175   -0.5044    0.6051    0.3356    1.5037   -1.3660   -0.2117
   -1.6164   -0.5033    0.6061    0.3367    1.5048   -1.3649   -0.2106
   -1.6154   -0.5022    0.6071    0.3378    1.5059   -1.3638   -0.2094
   -1.6143   -0.5011    0.6082    0.3388    1.5071   -1.3627   -0.2083
   -1.6132   -0.5000    0.6092    0.3399    1.5082   -1.3616   -0.2071
   -1.6122   -0.4988    0.6102    0.3410    1.5093   -1.3605   -0.2059
   -1.6111   -0.4977    0.6113    0.3421    1.5105   -1.3595   -0.2048
   -1.6101   -0.4966    0.6123    0.3432    1.5116   -1.3584   -0.2036
   -1.6090   -0.4955    0.6134    0.3443    1.5128   -1.3573   -0.2025
   -1.6079   -0.4944    0.6144    0.3454    1.5139   -1.3562   -0.2013
   -1.6069   -0.4933    0.6154    0.3465    1.5150   -1.3551   -0.2002
   -1.6058   -0.4922    0.6165    0.3476    1.5162   -1.3540   -0.1990
   -1.6047   -0.4911    0.6175    0.3487    1.5173   -1.3529   -0.1978
   -1.6037   -0.4900    0.6185    0.3498    1.5185   -1.3518   -0.1967
   -1.6026   -0.4888    0.6196    0.3509    1.5196   -1.3507   -0.1955
   -1.6015   -0.4877    0.6206    0.3520    1.5207   -1.3496   -0.1944
   -1.6005   -0.4866    0.6217    0.3531    1.5219   -1.3485   -0.1932
   -1.5994   -0.4855    0.6227    0.3542    1.5230   -1.3474   -0.1920
   -1.5983   -0.4844    0.6238    0.3553    1.5242   -1.3463   -0.1909
   -1.5973   -0.4833    0.6248    0.3564    1.5253   -1.3452   -0.1897
   -1.5962   -0.4821    0.6258    0.3576    1.5264   -1.3441   -0.1885
   -1.5951   -0.4810    0.6269    0.3587    1.5276   -1.3430   -0.1874
   -1.5940   -0.4799    0.6279    0.3598    1.5287   -1.3419   -0.1862
   -1.5930   -0.4788    0.6290    0.3609    1.5299   -1.3408   -0.1850
   -1.5919   -0.4776    0.6300    0.3620    1.5310   -1.3397   -0.1838
   -1.5908   -0.4765    0.6311    0.3631    1.5322   -1.3386   -0.1827
   -1.5898   -0.4754    0.6321    0.3642    1.5333   -1.3375   -0.1815
   -1.5887   -0.4743    0.6332    0.3653    1.5344   -1.3364   -0.1803
   -1.5876   -0.4731    0.6342    0.3664    1.5356   -1.3353   -0.1792
   -1.5865   -0.4720    0.6353    0.3676    1.5367   -1.3342   -0.1780
   -1.5854   -0.4709    0.6363    0.3687    1.5379   -1.3330   -0.1768
   -1.5844   -0.4697    0.6374    0.3698    1.5390   -1.3319   -0.1756
   -1.5833   -0.4686    0.6385    0.3709    1.5402   -1.3308   -0.1745
   -1.5822   -0.4675    0.6395    0.3720    1.5413   -1.3297   -0.1733
   -1.5811   -0.4663    0.6406    0.3731    1.5425   -1.3286   -0.1721
   -1.5800   -0.4652    0.6416    0.3743    1.5436   -1.3275   -0.1709
   -1.5790   -0.4641    0.6427    0.3754    1.5447   -1.3264   -0.1697
   -1.5779   -0.4629    0.6438    0.3765    1.5459   -1.3253   -0.1686
   -1.5768   -0.4618    0.6448    0.3776    1.5470   -1.3241   -0.1674
   -1.5757   -0.4606    0.6459    0.3787    1.5482   -1.3230   -0.1662
   -1.5746   -0.4595    0.6469    0.3799    1.5493   -1.3219   -0.1650
   -1.5735   -0.4584    0.6480    0.3810    1.5505   -1.3208   -0.1638
   -1.5724   -0.4572    0.6491    0.3821    1.5516   -1.3197   -0.1626
   -1.5714   -0.4561    0.6501    0.3832    1.5528   -1.3185   -0.1614
   -1.5703   -0.4549    0.6512    0.3844    1.5539   -1.3174   -0.1603
   -1.5692   -0.4538    0.6523    0.3855    1.5551   -1.3163   -0.1591
   -1.5681   -0.4526    0.6533    0.3866    1.5562   -1.3152   -0.1579
   -1.5670   -0.4515    0.6544    0.3877    1.5574   -1.3140   -0.1567
   -1.5659   -0.4503    0.6555    0.3889    1.5585   -1.3129   -0.1555
   -1.5648   -0.4492    0.6566    0.3900    1.5597   -1.3118   -0.1543
   -1.5637   -0.4480    0.6576    0.3911    1.5608   -1.3107   -0.1531
   -1.5626   -0.4469    0.6587    0.3923    1.5620   -1.3095   -0.1519
   -1.5615   -0.4457    0.6598    0.3934    1.5632   -1.3084   -0.1507
   -1.5604   -0.4446    0.6609    0.3945    1.5643   -1.3073   -0.1495
   -1.5593   -0.4434    0.6619    0.3957    1.5655   -1.3061   -0.1483
   -1.5582   -0.4422    0.6630    0.3968    1.5666   -1.3050   -0.1471
   -1.5571   -0.4411    0.6641    0.3979    1.5678   -1.3039   -0.1459
   -1.5561   -0.4399    0.6652    0.3991    1.5689   -1.3027   -0.1447
   -1.5550   -0.4388    0.6663    0.4002    1.5701   -1.3016   -0.1435
   -1.5539   -0.4376    0.6673    0.4013    1.5712   -1.3005   -0.1423
   -1.5528   -0.4364    0.6684    0.4025    1.5724   -1.2993   -0.1411
   -1.5516   -0.4353    0.6695    0.4036    1.5736   -1.2982   -0.1399
   -1.5505   -0.4341    0.6706    0.4047    1.5747   -1.2971   -0.1387
   -1.5494   -0.4329    0.6717    0.4059    1.5759   -1.2959   -0.1375
   -1.5483   -0.4318    0.6728    0.4070    1.5770   -1.2948   -0.1363
   -1.5472   -0.4306    0.6739    0.4082    1.5782   -1.2936   -0.1351
   -1.5461   -0.4294    0.6749    0.4093    1.5794   -1.2925   -0.1339
   -1.5450   -0.4283    0.6760    0.4105    1.5805   -1.2913   -0.1327
   -1.5439   -0.4271    0.6771    0.4116    1.5817   -1.2902   -0.1314
   -1.5428   -0.4259    0.6782    0.4127    1.5829   -1.2890   -0.1302
   -1.5417   -0.4248    0.6793    0.4139    1.5840   -1.2879   -0.1290
   -1.5406   -0.4236    0.6804    0.4150    1.5852   -1.2867   -0.1278
   -1.5395   -0.4224    0.6815    0.4162    1.5864   -1.2856   -0.1266
   -1.5384   -0.4212    0.6826    0.4173    1.5875   -1.2844   -0.1254
   -1.5373   -0.4200    0.6837    0.4185    1.5887   -1.2833   -0.1241
   -1.5361   -0.4189    0.6848    0.4196    1.5899   -1.2821   -0.1229
   -1.5350   -0.4177    0.6859    0.4208    1.5910   -1.2810   -0.1217
   -1.5339   -0.4165    0.6870    0.4219    1.5922   -1.2798   -0.1205
   -1.5328   -0.4153    0.6881    0.4231    1.5934   -1.2786   -0.1193
   -1.5317   -0.4141    0.6892    0.4242    1.5946   -1.2775   -0.1180
   -1.5306   -0.4130    0.6903    0.4254    1.5957   -1.2763   -0.1168
   -1.5295   -0.4118    0.6914    0.4266    1.5969   -1.2752   -0.1156
   -1.5283   -0.4106    0.6925    0.4277    1.5981   -1.2740   -0.1144
   -1.5272   -0.4094    0.6937    0.4289    1.5993   -1.2728   -0.1131
   -1.5261   -0.4082    0.6948    0.4300    1.6004   -1.2717   -0.1119
   -1.5250   -0.4070    0.6959    0.4312    1.6016   -1.2705   -0.1107
   -1.5239   -0.4058    0.6970    0.4323    1.6028   -1.2693   -0.1094
   -1.5227   -0.4046    0.6981    0.4335    1.6040   -1.2682   -0.1082
   -1.5216   -0.4034    0.6992    0.4347    1.6052   -1.2670   -0.1070
   -1.5205   -0.4023    0.7003    0.4358    1.6063   -1.2658   -0.1057
   -1.5194   -0.4011    0.7015    0.4370    1.6075   -1.2646   -0.1045
   -1.5182   -0.3999    0.7026    0.4381    1.6087   -1.2635   -0.1032
   -1.5171   -0.3987    0.7037    0.4393    1.6099   -1.2623   -0.1020
   -1.5160   -0.3975    0.7048    0.4405    1.6111   -1.2611   -0.1008
   -1.5149   -0.3963    0.7059    0.4416    1.6123   -1.2599   -0.0995
   -1.5137   -0.3951    0.7071    0.4428    1.6134   -1.2587   -0.0983
   -1.5126   -0.3939    0.7082    0.4440    1.6146   -1.2576   -0.0970
   -1.5115   -0.3927    0.7093    0.4451    1.6158   -1.2564   -0.0958
   -1.5104   -0.3915    0.7104    0.4463    1.6170   -1.2552   -0.0945
   -1.5092   -0.3903    0.7116    0.4475    1.6182   -1.2540   -0.0933
   -1.5081   -0.3891    0.7127    0.4487    1.6194   -1.2528   -0.0920
   -1.5070   -0.3879    0.7138    0.4498    1.6206   -1.2516   -0.0908
   -1.5058   -0.3867    0.7150    0.4510    1.6218   -1.2504   -0.0895
   -1.5047   -0.3855    0.7161    0.4522    1.6230   -1.2493   -0.0883
   -1.5036   -0.3842    0.7172    0.4534    1.6242   -1.2481   -0.0870
   -1.5024   -0.3830    0.7184    0.4545    1.6254   -1.2469   -0.0858
   -1.5013   -0.3818    0.7195    0.4557    1.6266   -1.2457   -0.0845
   -1.5002   -0.3806    0.7207    0.4569    1.6278   -1.2445   -0.0833
   -1.4990   -0.3794    0.7218    0.4581    1.6290   -1.2433   -0.0820
   -1.4979   -0.3782    0.7229    0.4593    1.6302   -1.2421   -0.0808
   -1.4968   -0.3770    0.7241    0.4604    1.6314   -1.2409   -0.0795
   -1.4956   -0.3758    0.7252    0.4616    1.6326   -1.2397   -0.0782
   -1.4945   -0.3746    0.7264    0.4628    1.6338   -1.2385   -0.0770
   -1.4934   -0.3733    0.7275    0.4640    1.6350   -1.2373   -0.0757
   -1.4922   -0.3721    0.7286    0.4652    1.6362   -1.2361   -0.0744
   -1.4911   -0.3709    0.7298    0.4664    1.6374   -1.2349   -0.0732
   -1.4899   -0.3697    0.7309    0.4675    1.6386   -1.2337   -0.0719
   -1.4888   -0.3685    0.7321    0.4687    1.6398   -1.2325   -0.0706
   -1.4877   -0.3673    0.7332    0.4699    1.6410   -1.2313   -0.0694
   -1.4865   -0.3660    0.7344    0.4711    1.6422   -1.2301   -0.0681
   -1.4854   -0.3648    0.7355    0.4723    1.6435   -1.2288   -0.0668
   -1.4842   -0.3636    0.7367    0.4735    1.6447   -1.2276   -0.0656
   -1.4831   -0.3624    0.7379    0.4747    1.6459   -1.2264   -0.0643
   -1.4819   -0.3612    0.7390    0.4759    1.6471   -1.2252   -0.0630
   -1.4808   -0.3599    0.7402    0.4771    1.6483   -1.2240   -0.0617
   -1.4797   -0.3587    0.7413    0.4783    1.6495   -1.2228   -0.0605
   -1.4785   -0.3575    0.7425    0.4795    1.6508   -1.2216   -0.0592
   -1.4774   -0.3563    0.7436    0.4807    1.6520   -1.2204   -0.0579
   -1.4762   -0.3550    0.7448    0.4819    1.6532   -1.2191   -0.0566
   -1.4751   -0.3538    0.7460    0.4831    1.6544   -1.2179   -0.0553
   -1.4739   -0.3526    0.7471    0.4843    1.6556   -1.2167   -0.0541
   -1.4728   -0.3514    0.7483    0.4855    1.6569   -1.2155   -0.0528
   -1.4716   -0.3501    0.7495    0.4867    1.6581   -1.2143   -0.0515
   -1.4705   -0.3489    0.7506    0.4879    1.6593   -1.2131   -0.0502
   -1.4694   -0.3477    0.7518    0.4891    1.6605   -1.2118   -0.0489
   -1.4682   -0.3465    0.7530    0.4903    1.6618   -1.2106   -0.0476
   -1.4671   -0.3452    0.7541    0.4915    1.6630   -1.2094   -0.0464
   -1.4659   -0.3440    0.7553    0.4927    1.6642   -1.2082   -0.0451
   -1.4648   -0.3428    0.7565    0.4939    1.6655   -1.2069   -0.0438
   -1.4636   -0.3415    0.7576    0.4951    1.6667   -1.2057   -0.0425
   -1.4625   -0.3403    0.7588    0.4963    1.6679   -1.2045   -0.0412
   -1.4613   -0.3391    0.7600    0.4976    1.6692   -1.2033   -0.0399
   -1.4602   -0.3378    0.7612    0.4988    1.6704   -1.2020   -0.0386
   -1.4590   -0.3366    0.7623    0.5000    1.6716   -1.2008   -0.0373
   -1.4579   -0.3354    0.7635    0.5012    1.6729   -1.1996   -0.0360
   -1.4567   -0.3342    0.7647    0.5024    1.6741   -1.1984   -0.0348
   -1.4556   -0.3329    0.7659    0.5036    1.6754   -1.1971   -0.0335
   -1.4544   -0.3317    0.7671    0.5048    1.6766   -1.1959   -0.0322
   -1.4533   -0.3305    0.7682    0.5061    1.6778   -1.1947   -0.0309
   -1.4521   -0.3292    0.7694    0.5073    1.6791   -1.1935   -0.0296
   -1.4510   -0.3280    0.7706    0.5085    1.6803   -1.1922   -0.0283
   -1.4498   -0.3268    0.7718    0.5097    1.6816   -1.1910   -0.0270
   -1.4487   -0.3255    0.7730    0.5109    1.6828   -1.1898   -0.0257
   -1.4475   -0.3243    0.7741    0.5122    1.6841   -1.1886   -0.0244
   -1.4464   -0.3231    0.7753    0.5134    1.6853   -1.1873   -0.0231
   -1.4452   -0.3218    0.7765    0.5146    1.6866   -1.1861   -0.0218
   -1.4441   -0.3206    0.7777    0.5158    1.6878   -1.1849   -0.0205
   -1.4429   -0.3194    0.7789    0.5170    1.6891   -1.1836   -0.0192
   -1.4418   -0.3181    0.7801    0.5183    1.6903   -1.1824   -0.0179
   -1.4406   -0.3169    0.7812    0.5195    1.6916   -1.1812   -0.0166
   -1.4395   -0.3157    0.7824    0.5207    1.6928   -1.1800   -0.0153
   -1.4383   -0.3144    0.7836    0.5219    1.6941   -1.1787   -0.0140
   -1.4372   -0.3132    0.7848    0.5232    1.6953   -1.1775   -0.0127
   -1.4360   -0.3120    0.7860    0.5244    1.6966   -1.1763   -0.0114
   -1.4349   -0.3107    0.7872    0.5256    1.6978   -1.1751   -0.0101
   -1.4337   -0.3095    0.7884    0.5269    1.6991   -1.1738   -0.0089
   -1.4325   -0.3083    0.7896    0.5281    1.7003   -1.1726   -0.0076
   -1.4314   -0.3070    0.7908    0.5293    1.7016   -1.1714   -0.0063
   -1.4302   -0.3058    0.7919    0.5305    1.7029   -1.1701   -0.0050
   -1.4291   -0.3046    0.7931    0.5318    1.7041   -1.1689   -0.0037
   -1.4279   -0.3033    0.7943    0.5330    1.7054   -1.1677   -0.0024
   -1.4268   -0.3021    0.7955    0.5342    1.7066   -1.1665   -0.0011
   -1.4256   -0.3009    0.7967    0.5355    1.7079   -1.1652    0.0002
   -1.4245   -0.2997    0.7979    0.5367    1.7092   -1.1640    0.0015
   -1.4233   -0.2984    0.7991    0.5379    1.7104   -1.1628    0.0028
   -1.4222   -0.2972    0.8003    0.5392    1.7117   -1.1616    0.0041
   -1.4210   -0.2960    0.8015    0.5404    1.7129   -1.1603    0.0054
   -1.4199   -0.2947    0.8027    0.5416    1.7142   -1.1591    0.0067
   -1.4187   -0.2935    0.8039    0.5429    1.7155   -1.1579    0.0080
   -1.4176   -0.2923    0.8051    0.5441    1.7167   -1.1567    0.0093
   -1.4164   -0.2910    0.8062    0.5453    1.7180   -1.1554    0.0106
   -1.4153   -0.2898    0.8074    0.5465    1.7193   -1.1542    0.0118
   -1.4142   -0.2886    0.8086    0.5478    1.7205   -1.1530    0.0131
   -1.4130   -0.2874    0.8098    0.5490    1.7218   -1.1518    0.0144
   -1.4119   -0.2861    0.8110    0.5502    1.7231   -1.1506    0.0157
   -1.4107   -0.2849    0.8122    0.5515    1.7243   -1.1493    0.0170
   -1.4096   -0.2837    0.8134    0.5527    1.7256   -1.1481    0.0183
   -1.4084   -0.2825    0.8146    0.5539    1.7269   -1.1469    0.0196
   -1.4073   -0.2812    0.8158    0.5552    1.7281   -1.1457    0.0209
   -1.4061   -0.2800    0.8170    0.5564    1.7294   -1.1445    0.0221
   -1.4050   -0.2788    0.8182    0.5576    1.7307   -1.1433    0.0234
   -1.4038   -0.2776    0.8194    0.5589    1.7319   -1.1420    0.0247
   -1.4027   -0.2764    0.8206    0.5601    1.7332   -1.1408    0.0260
   -1.4015   -0.2751    0.8217    0.5613    1.7344   -1.1396    0.0273
   -1.4004   -0.2739    0.8229    0.5625    1.7357   -1.1384    0.0286
   -1.3993   -0.2727    0.8241    0.5638    1.7370   -1.1372    0.0298
   -1.3981   -0.2715    0.8253    0.5650    1.7382   -1.1360    0.0311
   -1.3970   -0.2703    0.8265    0.5662    1.7395   -1.1348    0.0324
   -1.3958   -0.2690    0.8277    0.5675    1.7408   -1.1336    0.0337
   -1.3947   -0.2678    0.8289    0.5687    1.7420   -1.1323    0.0349
   -1.3935   -0.2666    0.8301    0.5699    1.7433   -1.1311    0.0362
   -1.3924   -0.2654    0.8313    0.5711    1.7446   -1.1299    0.0375
   -1.3913   -0.2642    0.8325    0.5724    1.7458   -1.1287    0.0388
   -1.3901   -0.2630    0.8336    0.5736    1.7471   -1.1275    0.0400
   -1.3890   -0.2618    0.8348    0.5748    1.7484   -1.1263    0.0413
   -1.3878   -0.2605    0.8360    0.5760    1.7496   -1.1251    0.0426
   -1.3867   -0.2593    0.8372    0.5772    1.7509   -1.1239    0.0438
   -1.3856   -0.2581    0.8384    0.5785    1.7521   -1.1227    0.0451
   -1.3844   -0.2569    0.8396    0.5797    1.7534   -1.1215    0.0464
   -1.3833   -0.2557    0.8408    0.5809    1.7547   -1.1203    0.0476
   -1.3822   -0.2545    0.8419    0.5821    1.7559   -1.1191    0.0489
   -1.3810   -0.2533    0.8431    0.5833    1.7572   -1.1179    0.0501
   -1.3799   -0.2521    0.8443    0.5846    1.7585   -1.1167    0.0514
   -1.3787   -0.2509    0.8455    0.5858    1.7597   -1.1155    0.0527
   -1.3776   -0.2497    0.8467    0.5870    1.7610   -1.1143    0.0539
   -1.3765   -0.2485    0.8478    0.5882    1.7622   -1.1131    0.0552
   -1.3753   -0.2473    0.8490    0.5894    1.7635   -1.1119    0.0564
   -1.3742   -0.2461    0.8502    0.5906    1.7647   -1.1107    0.0577
   -1.3731   -0.2449    0.8514    0.5918    1.7660   -1.1095    0.0589
   -1.3720   -0.2437    0.8526    0.5930    1.7673   -1.1083    0.0602
   -1.3708   -0.2425    0.8537    0.5943    1.7685   -1.1071    0.0614
   -1.3697   -0.2413    0.8549    0.5955    1.7698   -1.1059    0.0627
   -1.3686   -0.2401    0.8561    0.5967    1.7710   -1.1047    0.0639
   -1.3674   -0.2389    0.8573    0.5979    1.7723   -1.1035    0.0652
   -1.3663   -0.2377    0.8584    0.5991    1.7735   -1.1023    0.0664
   -1.3652   -0.2365    0.8596    0.6003    1.7748   -1.1012    0.0676
   -1.3641   -0.2354    0.8608    0.6015    1.7760   -1.1000    0.0689
   -1.3629   -0.2342    0.8619    0.6027    1.7773   -1.0988    0.0701
   -1.3618   -0.2330    0.8631    0.6039    1.7785   -1.0976    0.0713
   -1.3607   -0.2318    0.8643    0.6051    1.7798   -1.0964    0.0726
   -1.3596   -0.2306    0.8654    0.6063    1.7810   -1.0952    0.0738
   -1.3584   -0.2294    0.8666    0.6075    1.7822   -1.0940    0.0750
   -1.3573   -0.2282    0.8678    0.6087    1.7835   -1.0928    0.0763
   -1.3562   -0.2271    0.8689    0.6099    1.7847   -1.0917    0.0775
   -1.3551   -0.2259    0.8701    0.6111    1.7860   -1.0905    0.0787
   -1.3539   -0.2247    0.8712    0.6123    1.7872   -1.0893    0.0800
   -1.3528   -0.2235    0.8724    0.6135    1.7884   -1.0881    0.0812
   -1.3517   -0.2224    0.8736    0.6147    1.7897   -1.0869    0.0824
   -1.3506   -0.2212    0.8747    0.6158    1.7909   -1.0858    0.0836
   -1.3495   -0.2200    0.8759    0.6170    1.7922   -1.0846    0.0848
   -1.3483   -0.2188    0.8770    0.6182    1.7934   -1.0834    0.0861
   -1.3472   -0.2177    0.8782    0.6194    1.7946   -1.0822    0.0873
   -1.3461   -0.2165    0.8793    0.6206    1.7959   -1.0811    0.0885
   -1.3450   -0.2153    0.8805    0.6218    1.7971   -1.0799    0.0897
   -1.3439   -0.2142    0.8816    0.6230    1.7983   -1.0787    0.0909
   -1.3428   -0.2130    0.8828    0.6241    1.7995   -1.0775    0.0921
   -1.3417   -0.2118    0.8839    0.6253    1.8008   -1.0764    0.0933
   -1.3405   -0.2107    0.8851    0.6265    1.8020   -1.0752    0.0946
   -1.3394   -0.2095    0.8862    0.6277    1.8032   -1.0740    0.0958
   -1.3383   -0.2083    0.8874    0.6289    1.8044   -1.0729    0.0970
   -1.3372   -0.2072    0.8885    0.6300    1.8057   -1.0717    0.0982
   -1.3361   -0.2060    0.8896    0.6312    1.8069   -1.0705    0.0994
   -1.3350   -0.2049    0.8908    0.6324    1.8081   -1.0694    0.1006
   -1.3339   -0.2037    0.8919    0.6336    1.8093   -1.0682    0.1018
   -1.3328   -0.2026    0.8930    0.6347    1.8105   -1.0670    0.1030
   -1.3317   -0.2014    0.8942    0.6359    1.8117   -1.0659    0.1042
   -1.3306   -0.2003    0.8953    0.6371    1.8130   -1.0647    0.1054
   -1.3295   -0.1991    0.8964    0.6382    1.8142   -1.0635    0.1066
   -1.3284   -0.1980    0.8976    0.6394    1.8154   -1.0624    0.1078
   -1.3273   -0.1968    0.8987    0.6406    1.8166   -1.0612    0.1089
   -1.3262   -0.1957    0.8998    0.6417    1.8178   -1.0601    0.1101
   -1.3251   -0.1945    0.9009    0.6429    1.8190   -1.0589    0.1113
   -1.3240   -0.1934    0.9021    0.6441    1.8202   -1.0578    0.1125
   -1.3229   -0.1923    0.9032    0.6452    1.8214   -1.0566    0.1137
   -1.3218   -0.1911    0.9043    0.6464    1.8226   -1.0554    0.1149
   -1.3207   -0.1900    0.9054    0.6476    1.8238   -1.0543    0.1161
   -1.3196   -0.1888    0.9065    0.6487    1.8250   -1.0531    0.1172
   -1.3185   -0.1877    0.9076    0.6499    1.8262   -1.0520    0.1184
   -1.3174   -0.1866    0.9088    0.6510    1.8274   -1.0508    0.1196
   -1.3163   -0.1854    0.9099    0.6522    1.8286   -1.0497    0.1208
   -1.3152   -0.1843    0.9110    0.6533    1.8298   -1.0485    0.1220
   -1.3141   -0.1832    0.9121    0.6545    1.8310   -1.0474    0.1231
   -1.3130   -0.1820    0.9132    0.6557    1.8322   -1.0462    0.1243
   -1.3119   -0.1809    0.9143    0.6568    1.8334   -1.0451    0.1255
   -1.3108   -0.1798    0.9154    0.6580    1.8346   -1.0439    0.1266
   -1.3097   -0.1787    0.9165    0.6591    1.8357   -1.0428    0.1278
   -1.3086   -0.1775    0.9176    0.6603    1.8369   -1.0417    0.1290
   -1.3075   -0.1764    0.9187    0.6614    1.8381   -1.0405    0.1302
   -1.3064   -0.1753    0.9198    0.6626    1.8393   -1.0394    0.1313
   -1.3053   -0.1742    0.9209    0.6637    1.8405   -1.0382    0.1325
   -1.3043   -0.1731    0.9220    0.6649    1.8417   -1.0371    0.1336
   -1.3032   -0.1720    0.9231    0.6660    1.8428   -1.0360    0.1348
   -1.3021   -0.1708    0.9242    0.6671    1.8440   -1.0348    0.1360
   -1.3010   -0.1697    0.9252    0.6683    1.8452   -1.0337    0.1371
   -1.2999   -0.1686    0.9263    0.6694    1.8464   -1.0325    0.1383
   -1.2988   -0.1675    0.9274    0.6706    1.8475   -1.0314    0.1394
   -1.2977   -0.1664    0.9285    0.6717    1.8487   -1.0303    0.1406
   -1.2967   -0.1653    0.9296    0.6729    1.8499   -1.0291    0.1418
   -1.2956   -0.1642    0.9307    0.6740    1.8510   -1.0280    0.1429
   -1.2945   -0.1631    0.9317    0.6751    1.8522   -1.0269    0.1441
   -1.2934   -0.1620    0.9328    0.6763    1.8534   -1.0257    0.1452
   -1.2923   -0.1609    0.9339    0.6774    1.8545   -1.0246    0.1464
   -1.2912   -0.1598    0.9350    0.6786    1.8557   -1.0235    0.1475
   -1.2902   -0.1587    0.9360    0.6797    1.8569   -1.0224    0.1487
   -1.2891   -0.1576    0.9371    0.6808    1.8580   -1.0212    0.1498
   -1.2880   -0.1565    0.9382    0.6820    1.8592   -1.0201    0.1510
   -1.2869   -0.1554    0.9392    0.6831    1.8603   -1.0190    0.1521
   -1.2859   -0.1543    0.9403    0.6842    1.8615   -1.0179    0.1532
   -1.2848   -0.1532    0.9414    0.6854    1.8626   -1.0167    0.1544
   -1.2837   -0.1521    0.9424    0.6865    1.8638   -1.0156    0.1555
   -1.2826   -0.1510    0.9435    0.6876    1.8649   -1.0145    0.1567
   -1.2816   -0.1499    0.9445    0.6888    1.8661   -1.0134    0.1578
   -1.2805   -0.1488    0.9456    0.6899    1.8672   -1.0123    0.1589
   -1.2794   -0.1477    0.9466    0.6910    1.8684   -1.0111    0.1601
   -1.2783   -0.1466    0.9477    0.6922    1.8695   -1.0100    0.1612
   -1.2773   -0.1455    0.9487    0.6933    1.8707   -1.0089    0.1623
   -1.2762   -0.1445    0.9498    0.6944    1.8718   -1.0078    0.1635
   -1.2751   -0.1434    0.9508    0.6955    1.8730   -1.0067    0.1646
   -1.2741   -0.1423    0.9519    0.6967    1.8741   -1.0056    0.1657
   -1.2730   -0.1412    0.9529    0.6978    1.8752   -1.0045    0.1669
   -1.2719   -0.1401    0.9540    0.6989    1.8764   -1.0033    0.1680
   -1.2708   -0.1391    0.9550    0.7000    1.8775   -1.0022    0.1691
   -1.2698   -0.1380    0.9561    0.7012    1.8786   -1.0011    0.1703
   -1.2687   -0.1369    0.9571    0.7023    1.8798   -1.0000    0.1714
   -1.2676   -0.1358    0.9581    0.7034    1.8809   -0.9989    0.1725
   -1.2666   -0.1348    0.9592    0.7045    1.8820   -0.9978    0.1736
   -1.2655   -0.1337    0.9602    0.7057    1.8832   -0.9967    0.1748
   -1.2645   -0.1326    0.9612    0.7068    1.8843   -0.9956    0.1759
   -1.2634   -0.1315    0.9623    0.7079    1.8854   -0.9945    0.1770
   -1.2623   -0.1305    0.9633    0.7090    1.8865   -0.9934    0.1781
   -1.2613   -0.1294    0.9643    0.7101    1.8877   -0.9923    0.1792
   -1.2602   -0.1283    0.9654    0.7113    1.8888   -0.9912    0.1804
   -1.2591   -0.1273    0.9664    0.7124    1.8899   -0.9901    0.1815
   -1.2581   -0.1262    0.9674    0.7135    1.8910   -0.9890    0.1826
   -1.2570   -0.1251    0.9684    0.7146    1.8922   -0.9879    0.1837
   -1.2560   -0.1241    0.9694    0.7157    1.8933   -0.9868    0.1848
   -1.2549   -0.1230    0.9705    0.7168    1.8944   -0.9857    0.1859
   -1.2538   -0.1219    0.9715    0.7179    1.8955   -0.9846    0.1871
   -1.2528   -0.1209    0.9725    0.7191    1.8966   -0.9835    0.1882
   -1.2517   -0.1198    0.9735    0.7202    1.8977   -0.9824    0.1893
   -1.2507   -0.1188    0.9745    0.7213    1.8988   -0.9813    0.1904
   -1.2496   -0.1177    0.9755    0.7224    1.9000   -0.9802    0.1915
   -1.2485   -0.1167    0.9765    0.7235    1.9011   -0.9791    0.1926
   -1.2475   -0.1156    0.9775    0.7246    1.9022   -0.9780    0.1937
   -1.2464   -0.1145    0.9786    0.7257    1.9033   -0.9769    0.1948
   -1.2454   -0.1135    0.9796    0.7268    1.9044   -0.9758    0.1959
   -1.2443   -0.1124    0.9806    0.7280    1.9055   -0.9748    0.1970
   -1.2433   -0.1114    0.9816    0.7291    1.9066   -0.9737    0.1982
   -1.2422   -0.1103    0.9826    0.7302    1.9077   -0.9726    0.1993
   -1.2412   -0.1093    0.9836    0.7313    1.9088   -0.9715    0.2004
   -1.2401   -0.1082    0.9846    0.7324    1.9099   -0.9704    0.2015
   -1.2391   -0.1072    0.9856    0.7335    1.9110   -0.9693    0.2026
   -1.2380   -0.1061    0.9866    0.7346    1.9121   -0.9682    0.2037
   -1.2370   -0.1051    0.9876    0.7357    1.9132   -0.9671    0.2048
   -1.2359   -0.1041    0.9886    0.7368    1.9143   -0.9661    0.2059
   -1.2349   -0.1030    0.9896    0.7379    1.9154   -0.9650    0.2070
   -1.2338   -0.1020    0.9905    0.7390    1.9165   -0.9639    0.2081
   -1.2328   -0.1009    0.9915    0.7401    1.9176   -0.9628    0.2092
   -1.2317   -0.0999    0.9925    0.7412    1.9187   -0.9617    0.2103
   -1.2307   -0.0988    0.9935    0.7423    1.9198   -0.9606    0.2114
   -1.2296   -0.0978    0.9945    0.7434    1.9209   -0.9596    0.2125
   -1.2286   -0.0968    0.9955    0.7445    1.9220   -0.9585    0.2136
   -1.2275   -0.0957    0.9965    0.7456    1.9230   -0.9574    0.2147
   -1.2265   -0.0947    0.9975    0.7467    1.9241   -0.9563    0.2158
   -1.2254   -0.0936    0.9984    0.7478    1.9252   -0.9552    0.2169
   -1.2244   -0.0926    0.9994    0.7489    1.9263   -0.9542    0.2180
   -1.2233   -0.0916    1.0004    0.7500    1.9274   -0.9531    0.2190
   -1.2223   -0.0905    1.0014    0.7511    1.9285   -0.9520    0.2201
   -1.2213   -0.0895    1.0024    0.7522    1.9296   -0.9509    0.2212
   -1.2202   -0.0885    1.0033    0.7533    1.9307   -0.9499    0.2223
   -1.2192   -0.0874    1.0043    0.7544    1.9317   -0.9488    0.2234
   -1.2181   -0.0864    1.0053    0.7555    1.9328   -0.9477    0.2245
   -1.2171   -0.0854    1.0063    0.7566    1.9339   -0.9466    0.2256
   -1.2160   -0.0843    1.0072    0.7577    1.9350   -0.9456    0.2267
   -1.2150   -0.0833    1.0082    0.7588    1.9361   -0.9445    0.2278
   -1.2140   -0.0823    1.0092    0.7599    1.9372   -0.9434    0.2289
   -1.2129   -0.0813    1.0101    0.7610    1.9382   -0.9423    0.2300
   -1.2119   -0.0802    1.0111    0.7621    1.9393   -0.9413    0.2311
   -1.2108   -0.0792    1.0121    0.7632    1.9404   -0.9402    0.2322
   -1.2098   -0.0782    1.0130    0.7643    1.9415   -0.9391    0.2332
   -1.2088   -0.0771    1.0140    0.7654    1.9425   -0.9380    0.2343
   -1.2077   -0.0761    1.0150    0.7665    1.9436   -0.9370    0.2354
   -1.2067   -0.0751    1.0159    0.7676    1.9447   -0.9359    0.2365
   -1.2056   -0.0741    1.0169    0.7687    1.9458   -0.9348    0.2376
   -1.2046   -0.0730    1.0179    0.7698    1.9469   -0.9337    0.2387
   -1.2036   -0.0720    1.0188    0.7709    1.9479   -0.9327    0.2398
   -1.2025   -0.0710    1.0198    0.7719    1.9490   -0.9316    0.2409
   -1.2015   -0.0700    1.0208    0.7730    1.9501   -0.9305    0.2420
   -1.2004   -0.0690    1.0217    0.7741    1.9512   -0.9295    0.2430
   -1.1994   -0.0679    1.0227    0.7752    1.9522   -0.9284    0.2441
   -1.1984   -0.0669    1.0236    0.7763    1.9533   -0.9273    0.2452
   -1.1973   -0.0659    1.0246    0.7774    1.9544   -0.9262    0.2463
   -1.1963   -0.0649    1.0255    0.7785    1.9554   -0.9252    0.2474
   -1.1953   -0.0639    1.0265    0.7796    1.9565   -0.9241    0.2485
   -1.1942   -0.0628    1.0275    0.7807    1.9576   -0.9230    0.2496
   -1.1932   -0.0618    1.0284    0.7818    1.9587   -0.9220    0.2507
   -1.1922   -0.0608    1.0294    0.7828    1.9597   -0.9209    0.2518
   -1.1911   -0.0598    1.0303    0.7839    1.9608   -0.9198    0.2528
   -1.1901   -0.0588    1.0313    0.7850    1.9619   -0.9188    0.2539
   -1.1891   -0.0577    1.0322    0.7861    1.9629   -0.9177    0.2550
   -1.1880   -0.0567    1.0332    0.7872    1.9640   -0.9166    0.2561
   -1.1870   -0.0557    1.0341    0.7883    1.9651   -0.9155    0.2572
   -1.1860   -0.0547    1.0351    0.7894    1.9662   -0.9145    0.2583
   -1.1849   -0.0537    1.0360    0.7905    1.9672   -0.9134    0.2594
   -1.1839   -0.0527    1.0370    0.7916    1.9683   -0.9123    0.2605
   -1.1829   -0.0516    1.0379    0.7926    1.9694   -0.9113    0.2615
   -1.1818   -0.0506    1.0389    0.7937    1.9704   -0.9102    0.2626
   -1.1808   -0.0496    1.0398    0.7948    1.9715   -0.9091    0.2637
   -1.1798   -0.0486    1.0408    0.7959    1.9726   -0.9080    0.2648
   -1.1787   -0.0476    1.0417    0.7970    1.9737   -0.9070    0.2659
   -1.1777   -0.0466    1.0426    0.7981    1.9747   -0.9059    0.2670
   -1.1767   -0.0456    1.0436    0.7992    1.9758   -0.9048    0.2681
   -1.1756   -0.0445    1.0445    0.8003    1.9769   -0.9038    0.2692
   -1.1746   -0.0435    1.0455    0.8013    1.9779   -0.9027    0.2703
   -1.1736   -0.0425    1.0464    0.8024    1.9790   -0.9016    0.2713
   -1.1726   -0.0415    1.0474    0.8035    1.9801   -0.9005    0.2724
   -1.1715   -0.0405    1.0483    0.8046    1.9811   -0.8995    0.2735
   -1.1705   -0.0395    1.0492    0.8057    1.9822   -0.8984    0.2746
   -1.1695   -0.0385    1.0502    0.8068    1.9833   -0.8973    0.2757
   -1.1684   -0.0375    1.0511    0.8079    1.9844   -0.8962    0.2768
   -1.1674   -0.0364    1.0521    0.8089    1.9854   -0.8952    0.2779
   -1.1664   -0.0354    1.0530    0.8100    1.9865   -0.8941    0.2790
   -1.1653   -0.0344    1.0540    0.8111    1.9876   -0.8930    0.2801
   -1.1643   -0.0334    1.0549    0.8122    1.9886   -0.8919    0.2812
   -1.1633   -0.0324    1.0558    0.8133    1.9897   -0.8909    0.2822
   -1.1623   -0.0314    1.0568    0.8144    1.9908   -0.8898    0.2833
   -1.1612   -0.0304    1.0577    0.8155    1.9919   -0.8887    0.2844
   -1.1602   -0.0294    1.0586    0.8166    1.9929   -0.8876    0.2855
   -1.1592   -0.0283    1.0596    0.8176    1.9940   -0.8866    0.2866
   -1.1581   -0.0273    1.0605    0.8187    1.9951   -0.8855    0.2877
   -1.1571   -0.0263    1.0615    0.8198    1.9961   -0.8844    0.2888
   -1.1561   -0.0253    1.0624    0.8209    1.9972   -0.8833    0.2899
   -1.1551   -0.0243    1.0633    0.8220    1.9983   -0.8822    0.2910
   -1.1540   -0.0233    1.0643    0.8231    1.9994   -0.8812    0.2921
   -1.1530   -0.0223    1.0652    0.8242    2.0004   -0.8801    0.2932
   -1.1520   -0.0213    1.0661    0.8253    2.0015   -0.8790    0.2943
   -1.1510   -0.0202    1.0671    0.8263    2.0026   -0.8779    0.2954
   -1.1499   -0.0192    1.0680    0.8274    2.0037   -0.8768    0.2965
   -1.1489   -0.0182    1.0689    0.8285    2.0047   -0.8757    0.2976
   -1.1479   -0.0172    1.0699    0.8296    2.0058   -0.8747    0.2987
   -1.1468   -0.0162    1.0708    0.8307    2.0069   -0.8736    0.2998
   -1.1458   -0.0152    1.0717    0.8318    2.0080   -0.8725    0.3009
   -1.1448   -0.0142    1.0727    0.8329    2.0090   -0.8714    0.3020
   -1.1438   -0.0132    1.0736    0.8340    2.0101   -0.8703    0.3031
   -1.1427   -0.0121    1.0745    0.8351    2.0112   -0.8692    0.3042
   -1.1417   -0.0111    1.0755    0.8361    2.0123   -0.8681    0.3053
   -1.1407   -0.0101    1.0764    0.8372    2.0134   -0.8670    0.3064
   -1.1396   -0.0091    1.0773    0.8383    2.0144   -0.8660    0.3075
   -1.1386   -0.0081    1.0783    0.8394    2.0155   -0.8649    0.3086
   -1.1376   -0.0071    1.0792    0.8405    2.0166   -0.8638    0.3097
   -1.1365   -0.0061    1.0801    0.8416    2.0177   -0.8627    0.3108
   -1.1355   -0.0050    1.0811    0.8427    2.0188   -0.8616    0.3119
   -1.1345   -0.0040    1.0820    0.8438    2.0199   -0.8605    0.3130
   -1.1335   -0.0030    1.0830    0.8449    2.0209   -0.8594    0.3141
   -1.1324   -0.0020    1.0839    0.8460    2.0220   -0.8583    0.3152
   -1.1314   -0.0010    1.0848    0.8471    2.0231   -0.8572    0.3163
   -1.1304    0.0000    1.0858    0.8482    2.0242   -0.8561    0.3174
   -1.1293    0.0011    1.0867    0.8492    2.0253   -0.8550    0.3185
   -1.1283    0.0021    1.0876    0.8503    2.0264   -0.8539    0.3196
   -1.1273    0.0031    1.0886    0.8514    2.0274   -0.8528    0.3207
   -1.1262    0.0041    1.0895    0.8525    2.0285   -0.8517    0.3218
   -1.1252    0.0051    1.0904    0.8536    2.0296   -0.8506    0.3229
   -1.1242    0.0062    1.0914    0.8547    2.0307   -0.8495    0.3240
   -1.1231    0.0072    1.0923    0.8558    2.0318   -0.8484    0.3251
   -1.1221    0.0082    1.0932    0.8569    2.0329   -0.8473    0.3262
   -1.1211    0.0092    1.0942    0.8580    2.0340   -0.8462    0.3273
   -1.1200    0.0102    1.0951    0.8591    2.0351   -0.8451    0.3285
   -1.1190    0.0113    1.0960    0.8602    2.0362   -0.8440    0.3296
   -1.1180    0.0123    1.0970    0.8613    2.0373   -0.8429    0.3307
   -1.1169    0.0133    1.0979    0.8624    2.0384   -0.8418    0.3318
   -1.1159    0.0143    1.0988    0.8635    2.0394   -0.8407    0.3329
   -1.1149    0.0154    1.0998    0.8646    2.0405   -0.8395    0.3340
   -1.1138    0.0164    1.1007    0.8657    2.0416   -0.8384    0.3351
   -1.1128    0.0174    1.1016    0.8668    2.0427   -0.8373    0.3362
   -1.1118    0.0184    1.1026    0.8679    2.0438   -0.8362    0.3374
   -1.1107    0.0195    1.1035    0.8690    2.0449   -0.8351    0.3385
   -1.1097    0.0205    1.1045    0.8701    2.0460   -0.8340    0.3396
   -1.1087    0.0215    1.1054    0.8712    2.0471   -0.8329    0.3407
   -1.1076    0.0225    1.1063    0.8723    2.0482   -0.8317    0.3418
   -1.1066    0.0236    1.1073    0.8734    2.0493   -0.8306    0.3429
   -1.1055    0.0246    1.1082    0.8745    2.0504   -0.8295    0.3441
   -1.1045    0.0256    1.1091    0.8756    2.0515   -0.8284    0.3452
   -1.1035    0.0267    1.1101    0.8768    2.0526   -0.8273    0.3463
   -1.1024    0.0277    1.1110    0.8779    2.0537   -0.8261    0.3474
   -1.1014    0.0287    1.1120    0.8790    2.0548   -0.8250    0.3485
   -1.1004    0.0298    1.1129    0.8801    2.0559   -0.8239    0.3496
   -1.0993    0.0308    1.1138    0.8812    2.0570   -0.8228    0.3508
   -1.0983    0.0318    1.1148    0.8823    2.0581   -0.8216    0.3519
   -1.0972    0.0329    1.1157    0.8834    2.0592   -0.8205    0.3530
   -1.0962    0.0339    1.1166    0.8845    2.0603   -0.8194    0.3541
   -1.0952    0.0349    1.1176    0.8856    2.0614   -0.8183    0.3552
   -1.0941    0.0360    1.1185    0.8867    2.0625   -0.8171    0.3564
   -1.0931    0.0370    1.1195    0.8878    2.0637   -0.8160    0.3575
   -1.0920    0.0381    1.1204    0.8890    2.0648   -0.8149    0.3586
   -1.0910    0.0391    1.1214    0.8901    2.0659   -0.8138    0.3597
   -1.0899    0.0401    1.1223    0.8912    2.0670   -0.8126    0.3609
   -1.0889    0.0412    1.1232    0.8923    2.0681   -0.8115    0.3620
   -1.0878    0.0422    1.1242    0.8934    2.0692   -0.8104    0.3631
   -1.0868    0.0433    1.1251    0.8945    2.0703   -0.8092    0.3642
   -1.0858    0.0443    1.1261    0.8957    2.0714   -0.8081    0.3654
   -1.0847    0.0453    1.1270    0.8968    2.0725   -0.8069    0.3665
   -1.0837    0.0464    1.1280    0.8979    2.0736   -0.8058    0.3676
   -1.0826    0.0474    1.1289    0.8990    2.0747   -0.8047    0.3687
   -1.0816    0.0485    1.1298    0.9001    2.0758   -0.8035    0.3699
   -1.0805    0.0495    1.1308    0.9012    2.0770   -0.8024    0.3710
   -1.0795    0.0506    1.1317    0.9024    2.0781   -0.8013    0.3721
   -1.0784    0.0516    1.1327    0.9035    2.0792   -0.8001    0.3732
   -1.0774    0.0527    1.1336    0.9046    2.0803   -0.7990    0.3744
   -1.0763    0.0537    1.1346    0.9057    2.0814   -0.7978    0.3755
   -1.0753    0.0548    1.1355    0.9068    2.0825   -0.7967    0.3766
   -1.0742    0.0558    1.1365    0.9080    2.0836   -0.7955    0.3777
   -1.0732    0.0569    1.1374    0.9091    2.0847   -0.7944    0.3789
   -1.0721    0.0579    1.1384    0.9102    2.0858   -0.7932    0.3800
   -1.0711    0.0590    1.1393    0.9113    2.0870   -0.7921    0.3811
   -1.0700    0.0600    1.1403    0.9125    2.0881   -0.7910    0.3823
   -1.0690    0.0611    1.1412    0.9136    2.0892   -0.7898    0.3834
   -1.0679    0.0622    1.1422    0.9147    2.0903   -0.7887    0.3845
   -1.0668    0.0632    1.1431    0.9159    2.0914   -0.7875    0.3857
   -1.0658    0.0643    1.1441    0.9170    2.0925   -0.7863    0.3868
   -1.0647    0.0653    1.1450    0.9181    2.0936   -0.7852    0.3879
   -1.0637    0.0664    1.1460    0.9192    2.0948   -0.7840    0.3891
   -1.0626    0.0675    1.1469    0.9204    2.0959   -0.7829    0.3902
   -1.0616    0.0685    1.1479    0.9215    2.0970   -0.7817    0.3913
   -1.0605    0.0696    1.1488    0.9226    2.0981   -0.7806    0.3925
   -1.0594    0.0707    1.1498    0.9238    2.0992   -0.7794    0.3936
   -1.0584    0.0717    1.1507    0.9249    2.1004   -0.7783    0.3947
   -1.0573    0.0728    1.1517    0.9260    2.1015   -0.7771    0.3959
   -1.0563    0.0739    1.1526    0.9272    2.1026   -0.7759    0.3970
   -1.0552    0.0749    1.1536    0.9283    2.1037   -0.7748    0.3981
   -1.0541    0.0760    1.1546    0.9294    2.1048   -0.7736    0.3993
   -1.0531    0.0771    1.1555    0.9306    2.1060   -0.7724    0.4004
   -1.0520    0.0781    1.1565    0.9317    2.1071   -0.7713    0.4016
   -1.0509    0.0792    1.1574    0.9329    2.1082   -0.7701    0.4027
   -1.0499    0.0803    1.1584    0.9340    2.1093   -0.7689    0.4038
   -1.0488    0.0814    1.1594    0.9352    2.1104   -0.7678    0.4050
   -1.0477    0.0825    1.1603    0.9363    2.1116   -0.7666    0.4061
   -1.0466    0.0835    1.1613    0.9374    2.1127   -0.7654    0.4073
   -1.0456    0.0846    1.1623    0.9386    2.1138   -0.7642    0.4084
   -1.0445    0.0857    1.1632    0.9397    2.1149   -0.7631    0.4096
   -1.0434    0.0868    1.1642    0.9409    2.1161   -0.7619    0.4107
   -1.0423    0.0879    1.1652    0.9420    2.1172   -0.7607    0.4118
   -1.0413    0.0890    1.1661    0.9432    2.1183   -0.7595    0.4130
   -1.0402    0.0901    1.1671    0.9443    2.1194   -0.7583    0.4141
   -1.0391    0.0911    1.1681    0.9455    2.1206   -0.7571    0.4153
   -1.0380    0.0922    1.1690    0.9466    2.1217   -0.7560    0.4164
   -1.0369    0.0933    1.1700    0.9478    2.1228   -0.7548    0.4176
   -1.0359    0.0944    1.1710    0.9490    2.1240   -0.7536    0.4188
   -1.0348    0.0955    1.1720    0.9501    2.1251   -0.7524    0.4199
   -1.0337    0.0966    1.1729    0.9513    2.1262   -0.7512    0.4211
   -1.0326    0.0977    1.1739    0.9524    2.1274   -0.7500    0.4222
   -1.0315    0.0988    1.1749    0.9536    2.1285   -0.7488    0.4234
   -1.0304    0.0999    1.1759    0.9548    2.1296   -0.7476    0.4245
   -1.0293    0.1011    1.1769    0.9559    2.1308   -0.7464    0.4257
   -1.0282    0.1022    1.1778    0.9571    2.1319   -0.7452    0.4269
   -1.0271    0.1033    1.1788    0.9583    2.1331   -0.7440    0.4280
   -1.0260    0.1044    1.1798    0.9594    2.1342   -0.7428    0.4292
   -1.0249    0.1055    1.1808    0.9606    2.1354   -0.7416    0.4304
   -1.0238    0.1066    1.1818    0.9618    2.1365   -0.7404    0.4315
   -1.0227    0.1077    1.1828    0.9629    2.1376   -0.7392    0.4327
   -1.0216    0.1089    1.1838    0.9641    2.1388   -0.7379    0.4339
   -1.0205    0.1100    1.1848    0.9653    2.1399   -0.7367    0.4350
   -1.0194    0.1111    1.1858    0.9665    2.1411   -0.7355    0.4362
   -1.0183    0.1122    1.1868    0.9677    2.1422   -0.7343    0.4374
   -1.0172    0.1134    1.1878    0.9688    2.1434   -0.7331    0.4386
   -1.0161    0.1145    1.1887    0.9700    2.1445   -0.7319    0.4397
   -1.0150    0.1156    1.1898    0.9712    2.1457   -0.7306    0.4409
   -1.0138    0.1168    1.1908    0.9724    2.1468   -0.7294    0.4421
   -1.0127    0.1179    1.1918    0.9736    2.1480   -0.7282    0.4433
   -1.0116    0.1190    1.1928    0.9747    2.1491   -0.7270    0.4445
   -1.0105    0.1202    1.1938    0.9759    2.1503   -0.7257    0.4456
   -1.0094    0.1213    1.1948    0.9771    2.1515   -0.7245    0.4468
   -1.0082    0.1225    1.1958    0.9783    2.1526   -0.7233    0.4480
   -1.0071    0.1236    1.1968    0.9795    2.1538   -0.7220    0.4492
   -1.0060    0.1247    1.1978    0.9807    2.1549   -0.7208    0.4504
   -1.0049    0.1259    1.1988    0.9819    2.1561   -0.7196    0.4516
   -1.0037    0.1270    1.1998    0.9831    2.1573   -0.7183    0.4528
   -1.0026    0.1282    1.2008    0.9843    2.1584   -0.7171    0.4539
   -1.0015    0.1293    1.2019    0.9854    2.1596   -0.7159    0.4551
   -1.0004    0.1305    1.2029    0.9866    2.1607   -0.7146    0.4563
   -0.9992    0.1316    1.2039    0.9878    2.1619   -0.7134    0.4575
   -0.9981    0.1328    1.2049    0.9890    2.1631   -0.7121    0.4587
   -0.9970    0.1340    1.2059    0.9902    2.1642   -0.7109    0.4599
   -0.9958    0.1351    1.2069    0.9914    2.1654   -0.7097    0.4611
   -0.9947    0.1363    1.2080    0.9926    2.1666   -0.7084    0.4623
   -0.9936    0.1374    1.2090    0.9938    2.1677   -0.7072    0.4635
   -0.9924    0.1386    1.2100    0.9950    2.1689   -0.7059    0.4647
   -0.9913    0.1397    1.2110    0.9962    2.1701   -0.7047    0.4659
   -0.9902    0.1409    1.2120    0.9974    2.1713   -0.7034    0.4670
   -0.9890    0.1421    1.2131    0.9986    2.1724   -0.7022    0.4682
   -0.9879    0.1432    1.2141    0.9998    2.1736   -0.7010    0.4694
   -0.9868    0.1444    1.2151    1.0010    2.1748   -0.6997    0.4706
   -0.9856    0.1456    1.2161    1.0022    2.1759   -0.6985    0.4718
   -0.9845    0.1467    1.2172    1.0033    2.1771   -0.6972    0.4730
   -0.9834    0.1479    1.2182    1.0045    2.1783   -0.6960    0.4742
   -0.9822    0.1490    1.2192    1.0057    2.1795   -0.6947    0.4754
   -0.9811    0.1502    1.2202    1.0069    2.1806   -0.6935    0.4766
   -0.9800    0.1514    1.2213    1.0081    2.1818   -0.6923    0.4778
   -0.9788    0.1525    1.2223    1.0093    2.1830   -0.6910    0.4790
   -0.9777    0.1537    1.2233    1.0105    2.1841   -0.6898    0.4802
   -0.9766    0.1548    1.2243    1.0117    2.1853   -0.6885    0.4814
   -0.9754    0.1560    1.2254    1.0129    2.1865   -0.6873    0.4825
   -0.9743    0.1572    1.2264    1.0141    2.1877   -0.6861    0.4837
   -0.9732    0.1583    1.2274    1.0152    2.1888   -0.6848    0.4849
   -0.9720    0.1595    1.2284    1.0164    2.1900   -0.6836    0.4861
   -0.9709    0.1606    1.2295    1.0176    2.1912   -0.6823    0.4873
   -0.9698    0.1618    1.2305    1.0188    2.1923   -0.6811    0.4885
   -0.9687    0.1630    1.2315    1.0200    2.1935   -0.6799    0.4897
   -0.9675    0.1641    1.2325    1.0212    2.1947   -0.6786    0.4908
   -0.9664    0.1653    1.2335    1.0223    2.1959   -0.6774    0.4920
   -0.9653    0.1664    1.2346    1.0235    2.1970   -0.6762    0.4932
   -0.9642    0.1676    1.2356    1.0247    2.1982   -0.6749    0.4944
   -0.9630    0.1687    1.2366    1.0259    2.1994   -0.6737    0.4956
   -0.9619    0.1699    1.2376    1.0271    2.2005   -0.6725    0.4968
   -0.9608    0.1710    1.2386    1.0283    2.2017   -0.6712    0.4979
   -0.9596    0.1722    1.2397    1.0294    2.2029   -0.6700    0.4991
   -0.9585    0.1733    1.2407    1.0306    2.2041   -0.6688    0.5003
   -0.9574    0.1745    1.2417    1.0318    2.2052   -0.6675    0.5015
   -0.9563    0.1757    1.2427    1.0330    2.2064   -0.6663    0.5027
   -0.9551    0.1768    1.2437    1.0342    2.2076   -0.6651    0.5039
   -0.9540    0.1780    1.2448    1.0353    2.2088   -0.6638    0.5050
   -0.9529    0.1791    1.2458    1.0365    2.2100   -0.6626    0.5062
   -0.9518    0.1803    1.2468    1.0377    2.2111   -0.6614    0.5074
   -0.9506    0.1814    1.2478    1.0389    2.2123   -0.6601    0.5086
   -0.9495    0.1826    1.2489    1.0401    2.2135   -0.6589    0.5098
   -0.9484    0.1838    1.2499    1.0413    2.2147   -0.6576    0.5110
   -0.9472    0.1849    1.2509    1.0424    2.2159   -0.6564    0.5122
   -0.9461    0.1861    1.2519    1.0436    2.2171   -0.6552    0.5134
   -0.9450    0.1872    1.2530    1.0448    2.2182   -0.6539    0.5145
   -0.9438    0.1884    1.2540    1.0460    2.2194   -0.6527    0.5157
   -0.9427    0.1896    1.2550    1.0472    2.2206   -0.6514    0.5169
   -0.9416    0.1907    1.2561    1.0484    2.2218   -0.6502    0.5181
   -0.9404    0.1919    1.2571    1.0496    2.2230   -0.6489    0.5193
   -0.9393    0.1931    1.2582    1.0508    2.2242   -0.6477    0.5206
   -0.9381    0.1943    1.2592    1.0520    2.2255   -0.6464    0.5218
   -0.9370    0.1954    1.2603    1.0532    2.2267   -0.6451    0.5230
   -0.9358    0.1966    1.2613    1.0544    2.2279   -0.6439    0.5242
   -0.9346    0.1978    1.2624    1.0556    2.2291   -0.6426    0.5254
   -0.9335    0.1990    1.2634    1.0569    2.2303   -0.6413    0.5266
   -0.9323    0.2002    1.2645    1.0581    2.2315   -0.6400    0.5279
   -0.9311    0.2014    1.2655    1.0593    2.2328   -0.6388    0.5291
   -0.9300    0.2026    1.2666    1.0605    2.2340   -0.6375    0.5303
```

```octave fold:s.RsideSegm_BsideData.HEDO.x
>> s.RsideSegm_BsideData.HEDO.x

ans =

   1.0e+03 *

   -1.4421   -0.2917   -0.6664    0.5247   -1.2341   -0.0542
   -1.4409   -0.2906   -0.6651    0.5259   -1.2329   -0.0529
   -1.4397   -0.2894   -0.6639    0.5271   -1.2317   -0.0517
   -1.4385   -0.2883   -0.6627    0.5283   -1.2305   -0.0504
   -1.4373   -0.2871   -0.6615    0.5295   -1.2293   -0.0492
   -1.4361   -0.2860   -0.6603    0.5307   -1.2281   -0.0479
   -1.4349   -0.2849   -0.6590    0.5319   -1.2268   -0.0466
   -1.4337   -0.2837   -0.6578    0.5331   -1.2256   -0.0454
   -1.4325   -0.2826   -0.6566    0.5343   -1.2244   -0.0441
   -1.4313   -0.2814   -0.6554    0.5355   -1.2232   -0.0428
   -1.4301   -0.2803   -0.6541    0.5367   -1.2219   -0.0416
   -1.4289   -0.2792   -0.6529    0.5379   -1.2207   -0.0403
   -1.4277   -0.2780   -0.6517    0.5391   -1.2195   -0.0391
   -1.4265   -0.2769   -0.6505    0.5403   -1.2182   -0.0378
   -1.4253   -0.2757   -0.6492    0.5416   -1.2170   -0.0365
   -1.4241   -0.2746   -0.6480    0.5428   -1.2158   -0.0353
   -1.4229   -0.2734   -0.6468    0.5440   -1.2146   -0.0340
   -1.4217   -0.2723   -0.6455    0.5452   -1.2133   -0.0327
   -1.4205   -0.2712   -0.6443    0.5464   -1.2121   -0.0314
   -1.4193   -0.2700   -0.6431    0.5476   -1.2108   -0.0302
   -1.4181   -0.2689   -0.6418    0.5488   -1.2096   -0.0289
   -1.4169   -0.2677   -0.6406    0.5500   -1.2084   -0.0276
   -1.4157   -0.2666   -0.6393    0.5512   -1.2071   -0.0264
   -1.4145   -0.2654   -0.6381    0.5525   -1.2059   -0.0251
   -1.4132   -0.2643   -0.6369    0.5537   -1.2047   -0.0238
   -1.4120   -0.2631   -0.6356    0.5549   -1.2034   -0.0225
   -1.4108   -0.2620   -0.6344    0.5561   -1.2022   -0.0212
   -1.4096   -0.2609   -0.6331    0.5573   -1.2009   -0.0200
   -1.4084   -0.2597   -0.6319    0.5585   -1.1997   -0.0187
   -1.4072   -0.2586   -0.6307    0.5597   -1.1985   -0.0174
   -1.4060   -0.2574   -0.6294    0.5609   -1.1972   -0.0161
   -1.4048   -0.2563   -0.6282    0.5621   -1.1960   -0.0149
   -1.4036   -0.2552   -0.6269    0.5634   -1.1947   -0.0136
   -1.4024   -0.2540   -0.6257    0.5646   -1.1935   -0.0123
   -1.4012   -0.2529   -0.6245    0.5658   -1.1922   -0.0110
   -1.4000   -0.2518   -0.6232    0.5670   -1.1910   -0.0098
   -1.3988   -0.2506   -0.6220    0.5682   -1.1898   -0.0085
   -1.3976   -0.2495   -0.6207    0.5694   -1.1885   -0.0072
   -1.3964   -0.2483   -0.6195    0.5706   -1.1873   -0.0059
   -1.3952   -0.2472   -0.6183    0.5718   -1.1860   -0.0047
   -1.3940   -0.2461   -0.6170    0.5730   -1.1848   -0.0034
   -1.3928   -0.2450   -0.6158    0.5742   -1.1835   -0.0021
   -1.3916   -0.2438   -0.6145    0.5754   -1.1823   -0.0008
   -1.3904   -0.2427   -0.6133    0.5766   -1.1810    0.0004
   -1.3892   -0.2416   -0.6120    0.5778   -1.1798    0.0017
   -1.3880   -0.2405   -0.6108    0.5790   -1.1786    0.0030
   -1.3868   -0.2393   -0.6096    0.5802   -1.1773    0.0043
   -1.3856   -0.2382   -0.6083    0.5814   -1.1761    0.0055
   -1.3844   -0.2371   -0.6071    0.5826   -1.1748    0.0068
   -1.3832   -0.2360   -0.6059    0.5838   -1.1736    0.0081
   -1.3820   -0.2348   -0.6046    0.5850   -1.1724    0.0094
   -1.3808   -0.2337   -0.6034    0.5862   -1.1711    0.0106
   -1.3796   -0.2326   -0.6022    0.5874   -1.1699    0.0119
   -1.3784   -0.2315   -0.6009    0.5886   -1.1686    0.0132
   -1.3773   -0.2304   -0.5997    0.5898   -1.1674    0.0144
   -1.3761   -0.2293   -0.5985    0.5910   -1.1661    0.0157
   -1.3749   -0.2282   -0.5972    0.5922   -1.1649    0.0170
   -1.3737   -0.2271   -0.5960    0.5934   -1.1637    0.0182
   -1.3725   -0.2260   -0.5948    0.5946   -1.1624    0.0195
   -1.3713   -0.2249   -0.5935    0.5958   -1.1612    0.0207
   -1.3701   -0.2237   -0.5923    0.5969   -1.1600    0.0220
   -1.3690   -0.2226   -0.5911    0.5981   -1.1587    0.0233
   -1.3678   -0.2215   -0.5899    0.5993   -1.1575    0.0245
   -1.3666   -0.2204   -0.5886    0.6005   -1.1563    0.0258
   -1.3654   -0.2193   -0.5874    0.6017   -1.1550    0.0270
   -1.3642   -0.2182   -0.5862    0.6029   -1.1538    0.0283
   -1.3631   -0.2172   -0.5850    0.6040   -1.1526    0.0296
   -1.3619   -0.2161   -0.5837    0.6052   -1.1513    0.0308
   -1.3607   -0.2150   -0.5825    0.6064   -1.1501    0.0321
   -1.3595   -0.2139   -0.5813    0.6076   -1.1489    0.0333
   -1.3584   -0.2128   -0.5801    0.6087   -1.1476    0.0346
   -1.3572   -0.2117   -0.5789    0.6099   -1.1464    0.0358
   -1.3560   -0.2106   -0.5777    0.6111   -1.1452    0.0371
   -1.3549   -0.2095   -0.5765    0.6122   -1.1439    0.0383
   -1.3537   -0.2084   -0.5752    0.6134   -1.1427    0.0396
   -1.3525   -0.2074   -0.5740    0.6146   -1.1415    0.0408
   -1.3513   -0.2063   -0.5728    0.6157   -1.1403    0.0420
   -1.3502   -0.2052   -0.5716    0.6169   -1.1390    0.0433
   -1.3490   -0.2041   -0.5704    0.6181   -1.1378    0.0445
   -1.3478   -0.2030   -0.5692    0.6192   -1.1366    0.0458
   -1.3467   -0.2020   -0.5680    0.6204   -1.1354    0.0470
   -1.3455   -0.2009   -0.5668    0.6216   -1.1342    0.0482
   -1.3444   -0.1998   -0.5656    0.6227   -1.1329    0.0495
   -1.3432   -0.1987   -0.5644    0.6239   -1.1317    0.0507
   -1.3420   -0.1977   -0.5632    0.6250   -1.1305    0.0519
   -1.3409   -0.1966   -0.5620    0.6262   -1.1293    0.0532
   -1.3397   -0.1955   -0.5608    0.6273   -1.1281    0.0544
   -1.3385   -0.1944   -0.5596    0.6285   -1.1268    0.0556
   -1.3374   -0.1934   -0.5584    0.6296   -1.1256    0.0569
   -1.3362   -0.1923   -0.5572    0.6308   -1.1244    0.0581
   -1.3351   -0.1912   -0.5560    0.6319   -1.1232    0.0593
   -1.3339   -0.1902   -0.5548    0.6331   -1.1220    0.0605
   -1.3327   -0.1891   -0.5536    0.6342   -1.1208    0.0618
   -1.3316   -0.1881   -0.5524    0.6354   -1.1196    0.0630
   -1.3304   -0.1870   -0.5512    0.6365   -1.1183    0.0642
   -1.3293   -0.1859   -0.5500    0.6377   -1.1171    0.0654
   -1.3281   -0.1849   -0.5488    0.6388   -1.1159    0.0666
   -1.3270   -0.1838   -0.5476    0.6400   -1.1147    0.0678
   -1.3258   -0.1828   -0.5465    0.6411   -1.1135    0.0691
   -1.3247   -0.1817   -0.5453    0.6423   -1.1123    0.0703
   -1.3235   -0.1807   -0.5441    0.6434   -1.1111    0.0715
   -1.3224   -0.1796   -0.5429    0.6445   -1.1099    0.0727
   -1.3212   -0.1785   -0.5417    0.6457   -1.1087    0.0739
   -1.3201   -0.1775   -0.5405    0.6468   -1.1075    0.0751
   -1.3189   -0.1764   -0.5394    0.6480   -1.1063    0.0763
   -1.3178   -0.1754   -0.5382    0.6491   -1.1051    0.0775
   -1.3166   -0.1744   -0.5370    0.6502   -1.1039    0.0787
   -1.3155   -0.1733   -0.5358    0.6514   -1.1027    0.0799
   -1.3143   -0.1723   -0.5347    0.6525   -1.1015    0.0811
   -1.3132   -0.1712   -0.5335    0.6536   -1.1003    0.0823
   -1.3120   -0.1702   -0.5323    0.6548   -1.0991    0.0835
   -1.3109   -0.1691   -0.5311    0.6559   -1.0979    0.0847
   -1.3098   -0.1681   -0.5300    0.6570   -1.0967    0.0859
   -1.3086   -0.1670   -0.5288    0.6582   -1.0955    0.0871
   -1.3075   -0.1660   -0.5276    0.6593   -1.0943    0.0883
   -1.3063   -0.1650   -0.5264    0.6604   -1.0931    0.0895
   -1.3052   -0.1639   -0.5253    0.6615   -1.0919    0.0907
   -1.3041   -0.1629   -0.5241    0.6627   -1.0907    0.0919
   -1.3029   -0.1619   -0.5229    0.6638   -1.0895    0.0931
   -1.3018   -0.1608   -0.5218    0.6649   -1.0883    0.0942
   -1.3006   -0.1598   -0.5206    0.6660   -1.0871    0.0954
   -1.2995   -0.1588   -0.5194    0.6672   -1.0859    0.0966
   -1.2984   -0.1577   -0.5183    0.6683   -1.0847    0.0978
   -1.2972   -0.1567   -0.5171    0.6694   -1.0836    0.0990
   -1.2961   -0.1557   -0.5159    0.6705   -1.0824    0.1002
   -1.2950   -0.1546   -0.5148    0.6717   -1.0812    0.1013
   -1.2938   -0.1536   -0.5136    0.6728   -1.0800    0.1025
   -1.2927   -0.1526   -0.5125    0.6739   -1.0788    0.1037
   -1.2916   -0.1516   -0.5113    0.6750   -1.0776    0.1049
   -1.2904   -0.1505   -0.5101    0.6761   -1.0764    0.1060
   -1.2893   -0.1495   -0.5090    0.6772   -1.0753    0.1072
   -1.2882   -0.1485   -0.5078    0.6784   -1.0741    0.1084
   -1.2871   -0.1475   -0.5067    0.6795   -1.0729    0.1095
   -1.2859   -0.1465   -0.5055    0.6806   -1.0717    0.1107
   -1.2848   -0.1454   -0.5043    0.6817   -1.0705    0.1119
   -1.2837   -0.1444   -0.5032    0.6828   -1.0694    0.1130
   -1.2826   -0.1434   -0.5020    0.6839   -1.0682    0.1142
   -1.2814   -0.1424   -0.5009    0.6851   -1.0670    0.1154
   -1.2803   -0.1414   -0.4997    0.6862   -1.0658    0.1165
   -1.2792   -0.1404   -0.4986    0.6873   -1.0647    0.1177
   -1.2781   -0.1394   -0.4974    0.6884   -1.0635    0.1188
   -1.2769   -0.1383   -0.4963    0.6895   -1.0623    0.1200
   -1.2758   -0.1373   -0.4951    0.6906   -1.0612    0.1212
   -1.2747   -0.1363   -0.4940    0.6917   -1.0600    0.1223
   -1.2736   -0.1353   -0.4928    0.6928   -1.0588    0.1235
   -1.2725   -0.1343   -0.4917    0.6939   -1.0576    0.1246
   -1.2713   -0.1333   -0.4905    0.6950   -1.0565    0.1258
   -1.2702   -0.1323   -0.4894    0.6961   -1.0553    0.1269
   -1.2691   -0.1313   -0.4882    0.6972   -1.0541    0.1281
   -1.2680   -0.1303   -0.4871    0.6983   -1.0530    0.1292
   -1.2669   -0.1293   -0.4859    0.6995   -1.0518    0.1303
   -1.2658   -0.1283   -0.4848    0.7006   -1.0507    0.1315
   -1.2646   -0.1273   -0.4836    0.7017   -1.0495    0.1326
   -1.2635   -0.1263   -0.4825    0.7028   -1.0483    0.1338
   -1.2624   -0.1253   -0.4813    0.7039   -1.0472    0.1349
   -1.2613   -0.1243   -0.4802    0.7050   -1.0460    0.1361
   -1.2602   -0.1233   -0.4791    0.7061   -1.0449    0.1372
   -1.2591   -0.1223   -0.4779    0.7072   -1.0437    0.1383
   -1.2580   -0.1213   -0.4768    0.7083   -1.0425    0.1395
   -1.2569   -0.1203   -0.4756    0.7094   -1.0414    0.1406
   -1.2558   -0.1193   -0.4745    0.7105   -1.0402    0.1417
   -1.2546   -0.1183   -0.4733    0.7116   -1.0391    0.1429
   -1.2535   -0.1174   -0.4722    0.7126   -1.0379    0.1440
   -1.2524   -0.1164   -0.4711    0.7137   -1.0368    0.1451
   -1.2513   -0.1154   -0.4699    0.7148   -1.0356    0.1463
   -1.2502   -0.1144   -0.4688    0.7159   -1.0345    0.1474
   -1.2491   -0.1134   -0.4676    0.7170   -1.0333    0.1485
   -1.2480   -0.1124   -0.4665    0.7181   -1.0322    0.1496
   -1.2469   -0.1114   -0.4654    0.7192   -1.0310    0.1508
   -1.2458   -0.1104   -0.4642    0.7203   -1.0299    0.1519
   -1.2447   -0.1095   -0.4631    0.7214   -1.0288    0.1530
   -1.2436   -0.1085   -0.4620    0.7225   -1.0276    0.1541
   -1.2425   -0.1075   -0.4608    0.7236   -1.0265    0.1552
   -1.2414   -0.1065   -0.4597    0.7247   -1.0253    0.1564
   -1.2403   -0.1055   -0.4586    0.7258   -1.0242    0.1575
   -1.2392   -0.1046   -0.4574    0.7268   -1.0231    0.1586
   -1.2381   -0.1036   -0.4563    0.7279   -1.0219    0.1597
   -1.2370   -0.1026   -0.4551    0.7290   -1.0208    0.1608
   -1.2359   -0.1016   -0.4540    0.7301   -1.0196    0.1619
   -1.2348   -0.1006   -0.4529    0.7312   -1.0185    0.1631
   -1.2337   -0.0997   -0.4517    0.7323   -1.0174    0.1642
   -1.2326   -0.0987   -0.4506    0.7334   -1.0162    0.1653
   -1.2315   -0.0977   -0.4495    0.7345   -1.0151    0.1664
   -1.2304   -0.0968   -0.4484    0.7355   -1.0140    0.1675
   -1.2293   -0.0958   -0.4472    0.7366   -1.0129    0.1686
   -1.2282   -0.0948   -0.4461    0.7377   -1.0117    0.1697
   -1.2271   -0.0938   -0.4450    0.7388   -1.0106    0.1708
   -1.2260   -0.0929   -0.4438    0.7399   -1.0095    0.1719
   -1.2249   -0.0919   -0.4427    0.7410   -1.0083    0.1730
   -1.2238   -0.0909   -0.4416    0.7420   -1.0072    0.1741
   -1.2227   -0.0900   -0.4404    0.7431   -1.0061    0.1752
   -1.2216   -0.0890   -0.4393    0.7442   -1.0050    0.1763
   -1.2206   -0.0880   -0.4382    0.7453   -1.0038    0.1774
   -1.2195   -0.0871   -0.4371    0.7464   -1.0027    0.1785
   -1.2184   -0.0861   -0.4359    0.7474   -1.0016    0.1796
   -1.2173   -0.0851   -0.4348    0.7485   -1.0005    0.1807
   -1.2162   -0.0842   -0.4337    0.7496   -0.9994    0.1818
   -1.2151   -0.0832   -0.4326    0.7507   -0.9983    0.1829
   -1.2140   -0.0822   -0.4314    0.7518   -0.9971    0.1840
   -1.2129   -0.0813   -0.4303    0.7528   -0.9960    0.1851
   -1.2118   -0.0803   -0.4292    0.7539   -0.9949    0.1862
   -1.2107   -0.0793   -0.4281    0.7550   -0.9938    0.1873
   -1.2097   -0.0784   -0.4269    0.7561   -0.9927    0.1884
   -1.2086   -0.0774   -0.4258    0.7571   -0.9916    0.1895
   -1.2075   -0.0764   -0.4247    0.7582   -0.9905    0.1906
   -1.2064   -0.0755   -0.4236    0.7593   -0.9893    0.1917
   -1.2053   -0.0745   -0.4225    0.7604   -0.9882    0.1927
   -1.2042   -0.0736   -0.4213    0.7614   -0.9871    0.1938
   -1.2031   -0.0726   -0.4202    0.7625   -0.9860    0.1949
   -1.2021   -0.0717   -0.4191    0.7636   -0.9849    0.1960
   -1.2010   -0.0707   -0.4180    0.7647   -0.9838    0.1971
   -1.1999   -0.0697   -0.4169    0.7657   -0.9827    0.1982
   -1.1988   -0.0688   -0.4157    0.7668   -0.9816    0.1993
   -1.1977   -0.0678   -0.4146    0.7679   -0.9805    0.2003
   -1.1966   -0.0669   -0.4135    0.7689   -0.9794    0.2014
   -1.1955   -0.0659   -0.4124    0.7700   -0.9783    0.2025
   -1.1945   -0.0650   -0.4113    0.7711   -0.9772    0.2036
   -1.1934   -0.0640   -0.4101    0.7722   -0.9761    0.2047
   -1.1923   -0.0630   -0.4090    0.7732   -0.9750    0.2058
   -1.1912   -0.0621   -0.4079    0.7743   -0.9739    0.2068
   -1.1901   -0.0611   -0.4068    0.7754   -0.9728    0.2079
   -1.1891   -0.0602   -0.4057    0.7764   -0.9717    0.2090
   -1.1880   -0.0592   -0.4046    0.7775   -0.9706    0.2101
   -1.1869   -0.0583   -0.4035    0.7786   -0.9695    0.2111
   -1.1858   -0.0573   -0.4023    0.7796   -0.9684    0.2122
   -1.1847   -0.0564   -0.4012    0.7807   -0.9673    0.2133
   -1.1837   -0.0554   -0.4001    0.7818   -0.9662    0.2144
   -1.1826   -0.0545   -0.3990    0.7829   -0.9651    0.2155
   -1.1815   -0.0535   -0.3979    0.7839   -0.9640    0.2165
   -1.1804   -0.0526   -0.3968    0.7850   -0.9629    0.2176
   -1.1793   -0.0516   -0.3957    0.7861   -0.9618    0.2187
   -1.1783   -0.0507   -0.3945    0.7871   -0.9607    0.2197
   -1.1772   -0.0497   -0.3934    0.7882   -0.9596    0.2208
   -1.1761   -0.0487   -0.3923    0.7893   -0.9586    0.2219
   -1.1750   -0.0478   -0.3912    0.7903   -0.9575    0.2230
   -1.1739   -0.0468   -0.3901    0.7914   -0.9564    0.2240
   -1.1729   -0.0459   -0.3890    0.7925   -0.9553    0.2251
   -1.1718   -0.0449   -0.3879    0.7935   -0.9542    0.2262
   -1.1707   -0.0440   -0.3868    0.7946   -0.9531    0.2273
   -1.1696   -0.0430   -0.3856    0.7957   -0.9520    0.2283
   -1.1685   -0.0421   -0.3845    0.7967   -0.9509    0.2294
   -1.1675   -0.0411   -0.3834    0.7978   -0.9498    0.2305
   -1.1664   -0.0402   -0.3823    0.7989   -0.9488    0.2315
   -1.1653   -0.0393   -0.3812    0.7999   -0.9477    0.2326
   -1.1642   -0.0383   -0.3801    0.8010   -0.9466    0.2337
   -1.1632   -0.0374   -0.3790    0.8021   -0.9455    0.2347
   -1.1621   -0.0364   -0.3779    0.8031   -0.9444    0.2358
   -1.1610   -0.0355   -0.3768    0.8042   -0.9433    0.2369
   -1.1599   -0.0345   -0.3756    0.8053   -0.9422    0.2380
   -1.1589   -0.0336   -0.3745    0.8063   -0.9412    0.2390
   -1.1578   -0.0326   -0.3734    0.8074   -0.9401    0.2401
   -1.1567   -0.0317   -0.3723    0.8085   -0.9390    0.2412
   -1.1556   -0.0307   -0.3712    0.8095   -0.9379    0.2422
   -1.1546   -0.0298   -0.3701    0.8106   -0.9368    0.2433
   -1.1535   -0.0288   -0.3690    0.8117   -0.9358    0.2444
   -1.1524   -0.0279   -0.3679    0.8127   -0.9347    0.2454
   -1.1513   -0.0269   -0.3668    0.8138   -0.9336    0.2465
   -1.1502   -0.0260   -0.3657    0.8149   -0.9325    0.2476
   -1.1492   -0.0250   -0.3645    0.8159   -0.9314    0.2486
   -1.1481   -0.0241   -0.3634    0.8170   -0.9303    0.2497
   -1.1470   -0.0231   -0.3623    0.8181   -0.9293    0.2508
   -1.1459   -0.0222   -0.3612    0.8191   -0.9282    0.2518
   -1.1449   -0.0212   -0.3601    0.8202   -0.9271    0.2529
   -1.1438   -0.0203   -0.3590    0.8213   -0.9260    0.2540
   -1.1427   -0.0193   -0.3579    0.8223   -0.9249    0.2550
   -1.1416   -0.0184   -0.3568    0.8234   -0.9239    0.2561
   -1.1406   -0.0174   -0.3557    0.8245   -0.9228    0.2572
   -1.1395   -0.0165   -0.3546    0.8255   -0.9217    0.2582
   -1.1384   -0.0155   -0.3534    0.8266   -0.9206    0.2593
   -1.1373   -0.0146   -0.3523    0.8277   -0.9195    0.2604
   -1.1363   -0.0136   -0.3512    0.8287   -0.9184    0.2614
   -1.1352   -0.0127   -0.3501    0.8298   -0.9174    0.2625
   -1.1341   -0.0117   -0.3490    0.8309   -0.9163    0.2636
   -1.1330   -0.0108   -0.3479    0.8319   -0.9152    0.2646
   -1.1319   -0.0098   -0.3468    0.8330   -0.9141    0.2657
   -1.1309   -0.0089   -0.3457    0.8341   -0.9130    0.2668
   -1.1298   -0.0079   -0.3446    0.8351   -0.9120    0.2678
   -1.1287   -0.0070   -0.3435    0.8362   -0.9109    0.2689
   -1.1276   -0.0060   -0.3423    0.8373   -0.9098    0.2700
   -1.1266   -0.0051   -0.3412    0.8383   -0.9087    0.2711
   -1.1255   -0.0041   -0.3401    0.8394   -0.9076    0.2721
   -1.1244   -0.0032   -0.3390    0.8405   -0.9066    0.2732
   -1.1233   -0.0022   -0.3379    0.8416   -0.9055    0.2743
   -1.1222   -0.0013   -0.3368    0.8426   -0.9044    0.2753
   -1.1212   -0.0003   -0.3357    0.8437   -0.9033    0.2764
   -1.1201    0.0006   -0.3346    0.8448   -0.9022    0.2775
   -1.1190    0.0016   -0.3334    0.8458   -0.9011    0.2785
   -1.1179    0.0025   -0.3323    0.8469   -0.9001    0.2796
   -1.1168    0.0035   -0.3312    0.8480   -0.8990    0.2807
   -1.1158    0.0044   -0.3301    0.8491   -0.8979    0.2818
   -1.1147    0.0054   -0.3290    0.8501   -0.8968    0.2828
   -1.1136    0.0064   -0.3279    0.8512   -0.8957    0.2839
   -1.1125    0.0073   -0.3268    0.8523   -0.8946    0.2850
   -1.1114    0.0083   -0.3257    0.8534   -0.8935    0.2860
   -1.1103    0.0092   -0.3245    0.8544   -0.8925    0.2871
   -1.1093    0.0102   -0.3234    0.8555   -0.8914    0.2882
   -1.1082    0.0111   -0.3223    0.8566   -0.8903    0.2893
   -1.1071    0.0121   -0.3212    0.8577   -0.8892    0.2903
   -1.1060    0.0131   -0.3201    0.8587   -0.8881    0.2914
   -1.1049    0.0140   -0.3190    0.8598   -0.8870    0.2925
   -1.1038    0.0150   -0.3179    0.8609   -0.8859    0.2936
   -1.1028    0.0159   -0.3167    0.8620   -0.8848    0.2946
   -1.1017    0.0169   -0.3156    0.8630   -0.8838    0.2957
   -1.1006    0.0179   -0.3145    0.8641   -0.8827    0.2968
   -1.0995    0.0188   -0.3134    0.8652   -0.8816    0.2979
   -1.0984    0.0198   -0.3123    0.8663   -0.8805    0.2989
   -1.0973    0.0207   -0.3111    0.8674   -0.8794    0.3000
   -1.0962    0.0217   -0.3100    0.8684   -0.8783    0.3011
   -1.0951    0.0227   -0.3089    0.8695   -0.8772    0.3022
   -1.0940    0.0236   -0.3078    0.8706   -0.8761    0.3032
   -1.0930    0.0246   -0.3067    0.8717   -0.8750    0.3043
   -1.0919    0.0256   -0.3056    0.8728   -0.8739    0.3054
   -1.0908    0.0265   -0.3044    0.8738   -0.8728    0.3065
   -1.0897    0.0275   -0.3033    0.8749   -0.8717    0.3076
   -1.0886    0.0284   -0.3022    0.8760   -0.8706    0.3086
   -1.0875    0.0294   -0.3011    0.8771   -0.8695    0.3097
   -1.0864    0.0304   -0.3000    0.8782   -0.8684    0.3108
   -1.0853    0.0314   -0.2988    0.8793   -0.8673    0.3119
   -1.0842    0.0323   -0.2977    0.8804   -0.8662    0.3130
   -1.0831    0.0333   -0.2966    0.8814   -0.8651    0.3141
   -1.0820    0.0343   -0.2955    0.8825   -0.8640    0.3151
   -1.0809    0.0352   -0.2943    0.8836   -0.8629    0.3162
   -1.0798    0.0362   -0.2932    0.8847   -0.8618    0.3173
   -1.0787    0.0372   -0.2921    0.8858   -0.8607    0.3184
   -1.0776    0.0381   -0.2910    0.8869   -0.8596    0.3195
   -1.0765    0.0391   -0.2898    0.8880   -0.8585    0.3206
   -1.0754    0.0401   -0.2887    0.8891   -0.8574    0.3216
   -1.0743    0.0411   -0.2876    0.8902   -0.8563    0.3227
   -1.0732    0.0420   -0.2865    0.8913   -0.8552    0.3238
   -1.0721    0.0430   -0.2853    0.8924   -0.8541    0.3249
   -1.0710    0.0440   -0.2842    0.8935   -0.8530    0.3260
   -1.0699    0.0450   -0.2831    0.8945   -0.8519    0.3271
   -1.0688    0.0460   -0.2820    0.8956   -0.8507    0.3282
   -1.0677    0.0469   -0.2808    0.8967   -0.8496    0.3292
   -1.0666    0.0479   -0.2797    0.8978   -0.8485    0.3303
   -1.0655    0.0489   -0.2786    0.8989   -0.8474    0.3314
   -1.0644    0.0499   -0.2774    0.9000   -0.8463    0.3325
   -1.0633    0.0509   -0.2763    0.9011   -0.8452    0.3336
   -1.0621    0.0519   -0.2752    0.9022   -0.8441    0.3347
   -1.0610    0.0528   -0.2740    0.9033   -0.8429    0.3358
   -1.0599    0.0538   -0.2729    0.9044   -0.8418    0.3369
   -1.0588    0.0548   -0.2718    0.9055   -0.8407    0.3380
   -1.0577    0.0558   -0.2706    0.9066   -0.8396    0.3391
   -1.0566    0.0568   -0.2695    0.9077   -0.8385    0.3402
   -1.0555    0.0578   -0.2684    0.9089   -0.8373    0.3413
   -1.0543    0.0588   -0.2672    0.9100   -0.8362    0.3423
   -1.0532    0.0598   -0.2661    0.9111   -0.8351    0.3434
   -1.0521    0.0607   -0.2650    0.9122   -0.8340    0.3445
   -1.0510    0.0617   -0.2638    0.9133   -0.8328    0.3456
   -1.0499    0.0627   -0.2627    0.9144   -0.8317    0.3467
   -1.0487    0.0637   -0.2616    0.9155   -0.8306    0.3478
   -1.0476    0.0647   -0.2604    0.9166   -0.8295    0.3489
   -1.0465    0.0657   -0.2593    0.9177   -0.8283    0.3500
   -1.0454    0.0667   -0.2581    0.9188   -0.8272    0.3511
   -1.0442    0.0677   -0.2570    0.9199   -0.8261    0.3522
   -1.0431    0.0687   -0.2559    0.9211   -0.8249    0.3533
   -1.0420    0.0697   -0.2547    0.9222   -0.8238    0.3544
   -1.0409    0.0707   -0.2536    0.9233   -0.8227    0.3555
   -1.0397    0.0717   -0.2524    0.9244   -0.8215    0.3566
   -1.0386    0.0727   -0.2513    0.9255   -0.8204    0.3577
   -1.0375    0.0737   -0.2501    0.9266   -0.8193    0.3588
   -1.0363    0.0747   -0.2490    0.9278   -0.8181    0.3599
   -1.0352    0.0757   -0.2478    0.9289   -0.8170    0.3610
   -1.0341    0.0768   -0.2467    0.9300   -0.8158    0.3621
   -1.0329    0.0778   -0.2456    0.9311   -0.8147    0.3632
   -1.0318    0.0788   -0.2444    0.9322   -0.8136    0.3643
   -1.0306    0.0798   -0.2433    0.9334   -0.8124    0.3654
   -1.0295    0.0808   -0.2421    0.9345   -0.8113    0.3665
   -1.0284    0.0818   -0.2410    0.9356   -0.8101    0.3676
   -1.0272    0.0828   -0.2398    0.9367   -0.8090    0.3688
   -1.0261    0.0838   -0.2387    0.9379   -0.8078    0.3699
   -1.0249    0.0849   -0.2375    0.9390   -0.8067    0.3710
   -1.0238    0.0859   -0.2364    0.9401   -0.8055    0.3721
   -1.0226    0.0869   -0.2352    0.9412   -0.8044    0.3732
   -1.0215    0.0879   -0.2340    0.9424   -0.8032    0.3743
   -1.0203    0.0889   -0.2329    0.9435   -0.8021    0.3754
   -1.0192    0.0900   -0.2317    0.9446   -0.8009    0.3765
   -1.0180    0.0910   -0.2306    0.9457   -0.7998    0.3776
   -1.0169    0.0920   -0.2294    0.9469   -0.7986    0.3787
   -1.0157    0.0930   -0.2283    0.9480   -0.7975    0.3798
   -1.0146    0.0941   -0.2271    0.9491   -0.7963    0.3810
   -1.0134    0.0951   -0.2259    0.9503   -0.7951    0.3821
   -1.0123    0.0961   -0.2248    0.9514   -0.7940    0.3832
   -1.0111    0.0971   -0.2236    0.9525   -0.7928    0.3843
   -1.0099    0.0982   -0.2225    0.9537   -0.7917    0.3854
   -1.0088    0.0992   -0.2213    0.9548   -0.7905    0.3865
   -1.0076    0.1002   -0.2201    0.9560   -0.7893    0.3876
   -1.0065    0.1013   -0.2190    0.9571   -0.7882    0.3887
   -1.0053    0.1023   -0.2178    0.9582   -0.7870    0.3899
   -1.0041    0.1034   -0.2166    0.9594   -0.7858    0.3910
   -1.0030    0.1044   -0.2155    0.9605   -0.7847    0.3921
   -1.0018    0.1054   -0.2143    0.9617   -0.7835    0.3932
   -1.0006    0.1065   -0.2131    0.9628   -0.7823    0.3943
   -0.9995    0.1075   -0.2120    0.9639   -0.7812    0.3954
   -0.9983    0.1086   -0.2108    0.9651   -0.7800    0.3966
   -0.9971    0.1096   -0.2096    0.9662   -0.7788    0.3977
   -0.9959    0.1106   -0.2084    0.9674   -0.7776    0.3988
   -0.9948    0.1117   -0.2073    0.9685   -0.7765    0.3999
   -0.9936    0.1127   -0.2061    0.9697   -0.7753    0.4010
   -0.9924    0.1138   -0.2049    0.9708   -0.7741    0.4022
   -0.9912    0.1148   -0.2037    0.9720   -0.7729    0.4033
   -0.9900    0.1159   -0.2026    0.9731   -0.7717    0.4044
   -0.9889    0.1170   -0.2014    0.9743   -0.7706    0.4055
   -0.9877    0.1180   -0.2002    0.9754   -0.7694    0.4067
   -0.9865    0.1191   -0.1990    0.9766   -0.7682    0.4078
   -0.9853    0.1201   -0.1978    0.9777   -0.7670    0.4089
   -0.9841    0.1212   -0.1967    0.9789   -0.7658    0.4100
   -0.9829    0.1222   -0.1955    0.9800   -0.7646    0.4112
   -0.9817    0.1233   -0.1943    0.9812   -0.7634    0.4123
   -0.9805    0.1244   -0.1931    0.9823   -0.7622    0.4134
   -0.9794    0.1254   -0.1919    0.9835   -0.7611    0.4145
   -0.9782    0.1265   -0.1907    0.9847   -0.7599    0.4157
   -0.9770    0.1276   -0.1895    0.9858   -0.7587    0.4168
   -0.9758    0.1286   -0.1883    0.9870   -0.7575    0.4179
   -0.9746    0.1297   -0.1872    0.9881   -0.7563    0.4191
   -0.9734    0.1308   -0.1860    0.9893   -0.7551    0.4202
   -0.9722    0.1318   -0.1848    0.9905   -0.7539    0.4213
   -0.9710    0.1329   -0.1836    0.9916   -0.7527    0.4224
   -0.9698    0.1340   -0.1824    0.9928   -0.7515    0.4236
   -0.9686    0.1351   -0.1812    0.9940   -0.7503    0.4247
   -0.9674    0.1361   -0.1800    0.9951   -0.7491    0.4258
   -0.9662    0.1372   -0.1788    0.9963   -0.7479    0.4270
   -0.9649    0.1383   -0.1776    0.9975   -0.7467    0.4281
   -0.9637    0.1394   -0.1764    0.9986   -0.7455    0.4293
   -0.9625    0.1404   -0.1752    0.9998   -0.7443    0.4304
   -0.9613    0.1415   -0.1740    1.0010   -0.7430    0.4315
   -0.9601    0.1426   -0.1728    1.0021   -0.7418    0.4327
   -0.9589    0.1437   -0.1715    1.0033   -0.7406    0.4338
   -0.9577    0.1448   -0.1703    1.0045   -0.7394    0.4350
   -0.9565    0.1459   -0.1691    1.0056   -0.7382    0.4361
   -0.9552    0.1470   -0.1679    1.0068   -0.7370    0.4372
   -0.9540    0.1480   -0.1667    1.0080   -0.7358    0.4384
   -0.9528    0.1491   -0.1655    1.0092   -0.7345    0.4395
   -0.9516    0.1502   -0.1643    1.0104   -0.7333    0.4407
   -0.9504    0.1513   -0.1631    1.0115   -0.7321    0.4418
   -0.9492    0.1524   -0.1618    1.0127   -0.7309    0.4430
   -0.9479    0.1535   -0.1606    1.0139   -0.7297    0.4441
   -0.9467    0.1546   -0.1594    1.0151   -0.7284    0.4453
   -0.9455    0.1557   -0.1582    1.0163   -0.7272    0.4464
   -0.9443    0.1568   -0.1569    1.0174   -0.7260    0.4476
   -0.9430    0.1579   -0.1557    1.0186   -0.7247    0.4487
   -0.9418    0.1590   -0.1545    1.0198   -0.7235    0.4499
   -0.9406    0.1601   -0.1533    1.0210   -0.7223    0.4510
   -0.9393    0.1612   -0.1520    1.0222   -0.7211    0.4522
   -0.9381    0.1623   -0.1508    1.0234   -0.7198    0.4533
   -0.9369    0.1634   -0.1496    1.0246   -0.7186    0.4545
   -0.9356    0.1645   -0.1483    1.0257   -0.7173    0.4556
   -0.9344    0.1656   -0.1471    1.0269   -0.7161    0.4568
   -0.9332    0.1667   -0.1459    1.0281   -0.7149    0.4579
   -0.9319    0.1678   -0.1446    1.0293   -0.7136    0.4591
   -0.9307    0.1689   -0.1434    1.0305   -0.7124    0.4603
   -0.9295    0.1701   -0.1421    1.0317   -0.7111    0.4614
   -0.9282    0.1712   -0.1409    1.0329   -0.7099    0.4626
   -0.9270    0.1723   -0.1396    1.0341   -0.7087    0.4637
   -0.9257    0.1734   -0.1384    1.0353   -0.7074    0.4649
   -0.9245    0.1745   -0.1371    1.0365   -0.7062    0.4661
   -0.9233    0.1756   -0.1359    1.0377   -0.7049    0.4672
   -0.9220    0.1767   -0.1346    1.0389   -0.7037    0.4684
   -0.9208    0.1779   -0.1334    1.0401   -0.7024    0.4696
   -0.9195    0.1790   -0.1321    1.0413   -0.7012    0.4707
   -0.9183    0.1801   -0.1309    1.0425   -0.6999    0.4719
   -0.9170    0.1812   -0.1296    1.0437   -0.6986    0.4731
   -0.9158    0.1823   -0.1283    1.0449   -0.6974    0.4743
   -0.9146    0.1835   -0.1271    1.0461   -0.6961    0.4754
   -0.9133    0.1846   -0.1258    1.0473   -0.6949    0.4766
   -0.9121    0.1857   -0.1246    1.0485   -0.6936    0.4778
   -0.9108    0.1868   -0.1233    1.0497   -0.6923    0.4790
   -0.9096    0.1879   -0.1220    1.0509   -0.6911    0.4801
   -0.9083    0.1891   -0.1207    1.0521   -0.6898    0.4813
   -0.9071    0.1902   -0.1195    1.0534   -0.6886    0.4825
   -0.9058    0.1913   -0.1182    1.0546   -0.6873    0.4837
   -0.9046    0.1925   -0.1169    1.0558   -0.6860    0.4849
   -0.9033    0.1936   -0.1156    1.0570   -0.6847    0.4860
   -0.9021    0.1947   -0.1144    1.0582   -0.6835    0.4872
   -0.9008    0.1958   -0.1131    1.0594   -0.6822    0.4884
   -0.8996    0.1970   -0.1118    1.0606   -0.6809    0.4896
   -0.8983    0.1981   -0.1105    1.0619   -0.6797    0.4908
   -0.8970    0.1992   -0.1092    1.0631   -0.6784    0.4920
   -0.8958    0.2004   -0.1079    1.0643   -0.6771    0.4932
   -0.8945    0.2015   -0.1066    1.0655   -0.6758    0.4944
   -0.8933    0.2026   -0.1054    1.0667   -0.6745    0.4956
   -0.8920    0.2038   -0.1041    1.0679   -0.6733    0.4967
   -0.8908    0.2049   -0.1028    1.0692   -0.6720    0.4979
   -0.8895    0.2060   -0.1015    1.0704   -0.6707    0.4991
   -0.8883    0.2072   -0.1002    1.0716   -0.6694    0.5003
   -0.8870    0.2083   -0.0989    1.0728   -0.6681    0.5015
   -0.8858    0.2094   -0.0976    1.0741   -0.6668    0.5027
   -0.8845    0.2106   -0.0963    1.0753   -0.6656    0.5039
   -0.8832    0.2117   -0.0950    1.0765   -0.6643    0.5051
   -0.8820    0.2129   -0.0937    1.0777   -0.6630    0.5063
   -0.8807    0.2140   -0.0924    1.0790   -0.6617    0.5075
   -0.8795    0.2151   -0.0911    1.0802   -0.6604    0.5087
   -0.8782    0.2163   -0.0897    1.0814   -0.6591    0.5099
   -0.8770    0.2174   -0.0884    1.0826   -0.6578    0.5111
   -0.8757    0.2185   -0.0871    1.0839   -0.6565    0.5123
   -0.8744    0.2197   -0.0858    1.0851   -0.6552    0.5136
   -0.8732    0.2208   -0.0845    1.0863   -0.6539    0.5148
   -0.8719    0.2220   -0.0832    1.0876   -0.6526    0.5160
   -0.8707    0.2231   -0.0819    1.0888   -0.6513    0.5172
   -0.8694    0.2242   -0.0805    1.0900   -0.6500    0.5184
   -0.8682    0.2254   -0.0792    1.0912   -0.6487    0.5196
   -0.8669    0.2265   -0.0779    1.0925   -0.6474    0.5208
   -0.8657    0.2277   -0.0766    1.0937   -0.6461    0.5220
   -0.8644    0.2288   -0.0753    1.0949   -0.6448    0.5232
   -0.8632    0.2300   -0.0739    1.0962   -0.6435    0.5245
   -0.8619    0.2311   -0.0726    1.0974   -0.6422    0.5257
   -0.8606    0.2322   -0.0713    1.0986   -0.6409    0.5269
   -0.8594    0.2334   -0.0700    1.0999   -0.6396    0.5281
   -0.8581    0.2345   -0.0686    1.1011   -0.6383    0.5293
   -0.8569    0.2357   -0.0673    1.1023   -0.6370    0.5305
   -0.8556    0.2368   -0.0660    1.1036   -0.6357    0.5318
   -0.8544    0.2379   -0.0647    1.1048   -0.6344    0.5330
   -0.8531    0.2391   -0.0633    1.1060   -0.6331    0.5342
   -0.8519    0.2402   -0.0620    1.1073   -0.6318    0.5354
   -0.8506    0.2414   -0.0607    1.1085   -0.6305    0.5366
   -0.8494    0.2425   -0.0593    1.1097   -0.6292    0.5379
   -0.8481    0.2437   -0.0580    1.1110   -0.6279    0.5391
   -0.8469    0.2448   -0.0567    1.1122   -0.6266    0.5403
   -0.8456    0.2459   -0.0553    1.1135   -0.6252    0.5415
   -0.8444    0.2471   -0.0540    1.1147   -0.6239    0.5428
   -0.8431    0.2482   -0.0527    1.1159   -0.6226    0.5440
   -0.8419    0.2494   -0.0513    1.1172   -0.6213    0.5452
   -0.8406    0.2505   -0.0500    1.1184   -0.6200    0.5464
   -0.8394    0.2516   -0.0487    1.1196   -0.6187    0.5477
   -0.8382    0.2528   -0.0473    1.1209   -0.6174    0.5489
   -0.8369    0.2539   -0.0460    1.1221   -0.6161    0.5501
   -0.8357    0.2551   -0.0447    1.1233   -0.6148    0.5514
   -0.8344    0.2562   -0.0433    1.1246   -0.6134    0.5526
   -0.8332    0.2573   -0.0420    1.1258   -0.6121    0.5538
   -0.8319    0.2585   -0.0407    1.1270   -0.6108    0.5550
   -0.8307    0.2596   -0.0393    1.1283   -0.6095    0.5563
   -0.8295    0.2607   -0.0380    1.1295   -0.6082    0.5575
   -0.8282    0.2619   -0.0367    1.1307   -0.6069    0.5587
   -0.8270    0.2630   -0.0353    1.1320   -0.6056    0.5599
   -0.8257    0.2642   -0.0340    1.1332   -0.6043    0.5612
   -0.8245    0.2653   -0.0327    1.1344   -0.6029    0.5624
   -0.8233    0.2664   -0.0313    1.1357   -0.6016    0.5636
   -0.8220    0.2676   -0.0300    1.1369   -0.6003    0.5649
   -0.8208    0.2687   -0.0287    1.1381   -0.5990    0.5661
   -0.8196    0.2698   -0.0273    1.1394   -0.5977    0.5673
   -0.8184    0.2710   -0.0260    1.1406   -0.5964    0.5686
   -0.8171    0.2721   -0.0247    1.1418   -0.5951    0.5698
   -0.8159    0.2732   -0.0233    1.1431   -0.5938    0.5710
   -0.8147    0.2743   -0.0220    1.1443   -0.5925    0.5722
   -0.8134    0.2755   -0.0207    1.1455   -0.5912    0.5735
   -0.8122    0.2766   -0.0193    1.1467   -0.5898    0.5747
   -0.8110    0.2777   -0.0180    1.1480   -0.5885    0.5759
   -0.8098    0.2789   -0.0167    1.1492   -0.5872    0.5772
   -0.8085    0.2800   -0.0154    1.1504   -0.5859    0.5784
   -0.8073    0.2811   -0.0140    1.1517   -0.5846    0.5796
   -0.8061    0.2822   -0.0127    1.1529   -0.5833    0.5808
   -0.8049    0.2834   -0.0114    1.1541   -0.5820    0.5821
   -0.8037    0.2845   -0.0101    1.1553   -0.5807    0.5833
   -0.8025    0.2856   -0.0087    1.1566   -0.5794    0.5845
   -0.8012    0.2867   -0.0074    1.1578   -0.5781    0.5857
   -0.8000    0.2879   -0.0061    1.1590   -0.5768    0.5870
   -0.7988    0.2890   -0.0048    1.1602   -0.5755    0.5882
   -0.7976    0.2901   -0.0035    1.1614   -0.5742    0.5894
   -0.7964    0.2912   -0.0022    1.1627   -0.5729    0.5906
   -0.7952    0.2923   -0.0008    1.1639   -0.5716    0.5919
   -0.7940    0.2934    0.0005    1.1651   -0.5703    0.5931
   -0.7928    0.2946    0.0018    1.1663   -0.5690    0.5943
   -0.7916    0.2957    0.0031    1.1675   -0.5677    0.5955
   -0.7904    0.2968    0.0044    1.1688   -0.5664    0.5968
   -0.7892    0.2979    0.0057    1.1700   -0.5651    0.5980
   -0.7880    0.2990    0.0070    1.1712   -0.5638    0.5992
   -0.7868    0.3001    0.0083    1.1724   -0.5625    0.6004
   -0.7856    0.3012    0.0096    1.1736   -0.5612    0.6016
   -0.7844    0.3023    0.0109    1.1748   -0.5599    0.6029
   -0.7832    0.3035    0.0122    1.1760   -0.5587    0.6041
   -0.7820    0.3046    0.0135    1.1772   -0.5574    0.6053
   -0.7808    0.3057    0.0148    1.1785   -0.5561    0.6065
   -0.7796    0.3068    0.0161    1.1797   -0.5548    0.6077
   -0.7784    0.3079    0.0174    1.1809   -0.5535    0.6089
   -0.7772    0.3090    0.0187    1.1821   -0.5522    0.6102
   -0.7761    0.3101    0.0200    1.1833   -0.5509    0.6114
   -0.7749    0.3112    0.0213    1.1845   -0.5497    0.6126
   -0.7737    0.3123    0.0226    1.1857   -0.5484    0.6138
   -0.7725    0.3134    0.0239    1.1869   -0.5471    0.6150
   -0.7713    0.3145    0.0252    1.1881   -0.5458    0.6162
   -0.7701    0.3156    0.0265    1.1893   -0.5445    0.6174
   -0.7690    0.3167    0.0277    1.1905   -0.5433    0.6186
   -0.7678    0.3178    0.0290    1.1917   -0.5420    0.6198
   -0.7666    0.3188    0.0303    1.1929   -0.5407    0.6210
   -0.7654    0.3199    0.0316    1.1941   -0.5394    0.6222
   -0.7643    0.3210    0.0329    1.1953   -0.5382    0.6234
   -0.7631    0.3221    0.0341    1.1965   -0.5369    0.6247
   -0.7619    0.3232    0.0354    1.1977   -0.5356    0.6259
   -0.7608    0.3243    0.0367    1.1989   -0.5344    0.6271
   -0.7596    0.3254    0.0380    1.2001   -0.5331    0.6283
   -0.7584    0.3265    0.0392    1.2013   -0.5318    0.6295
   -0.7573    0.3275    0.0405    1.2025   -0.5306    0.6306
   -0.7561    0.3286    0.0418    1.2036   -0.5293    0.6318
   -0.7550    0.3297    0.0430    1.2048   -0.5280    0.6330
   -0.7538    0.3308    0.0443    1.2060   -0.5268    0.6342
   -0.7526    0.3319    0.0456    1.2072   -0.5255    0.6354
   -0.7515    0.3329    0.0468    1.2084   -0.5243    0.6366
   -0.7503    0.3340    0.0481    1.2096   -0.5230    0.6378
   -0.7492    0.3351    0.0493    1.2108   -0.5217    0.6390
   -0.7480    0.3361    0.0506    1.2119   -0.5205    0.6402
   -0.7469    0.3372    0.0518    1.2131   -0.5192    0.6414
   -0.7457    0.3383    0.0531    1.2143   -0.5180    0.6426
   -0.7446    0.3394    0.0544    1.2155   -0.5167    0.6437
   -0.7434    0.3404    0.0556    1.2167   -0.5155    0.6449
   -0.7423    0.3415    0.0569    1.2178   -0.5142    0.6461
   -0.7412    0.3426    0.0581    1.2190   -0.5130    0.6473
   -0.7400    0.3436    0.0593    1.2202   -0.5117    0.6485
   -0.7389    0.3447    0.0606    1.2214   -0.5105    0.6496
   -0.7377    0.3457    0.0618    1.2225   -0.5093    0.6508
   -0.7366    0.3468    0.0631    1.2237   -0.5080    0.6520
   -0.7355    0.3479    0.0643    1.2249   -0.5068    0.6532
   -0.7343    0.3489    0.0655    1.2261   -0.5055    0.6543
   -0.7332    0.3500    0.0668    1.2272   -0.5043    0.6555
   -0.7321    0.3510    0.0680    1.2284   -0.5031    0.6567
   -0.7309    0.3521    0.0693    1.2296   -0.5018    0.6578
   -0.7298    0.3531    0.0705    1.2307   -0.5006    0.6590
   -0.7287    0.3542    0.0717    1.2319   -0.4994    0.6602
   -0.7276    0.3552    0.0729    1.2331   -0.4981    0.6613
   -0.7264    0.3563    0.0742    1.2342   -0.4969    0.6625
   -0.7253    0.3573    0.0754    1.2354   -0.4957    0.6636
   -0.7242    0.3584    0.0766    1.2366   -0.4944    0.6648
   -0.7231    0.3594    0.0778    1.2377   -0.4932    0.6660
   -0.7220    0.3604    0.0791    1.2389   -0.4920    0.6671
   -0.7208    0.3615    0.0803    1.2400   -0.4908    0.6683
   -0.7197    0.3625    0.0815    1.2412   -0.4895    0.6694
   -0.7186    0.3636    0.0827    1.2423   -0.4883    0.6706
   -0.7175    0.3646    0.0839    1.2435   -0.4871    0.6717
   -0.7164    0.3656    0.0851    1.2447   -0.4859    0.6729
   -0.7153    0.3667    0.0864    1.2458   -0.4847    0.6740
   -0.7142    0.3677    0.0876    1.2470   -0.4835    0.6751
   -0.7131    0.3687    0.0888    1.2481   -0.4822    0.6763
   -0.7120    0.3697    0.0900    1.2493   -0.4810    0.6774
   -0.7109    0.3708    0.0912    1.2504   -0.4798    0.6786
   -0.7097    0.3718    0.0924    1.2516   -0.4786    0.6797
   -0.7086    0.3728    0.0936    1.2527   -0.4774    0.6808
   -0.7075    0.3739    0.0948    1.2538   -0.4762    0.6820
   -0.7064    0.3749    0.0960    1.2550   -0.4750    0.6831
   -0.7053    0.3759    0.0972    1.2561   -0.4738    0.6842
   -0.7042    0.3769    0.0984    1.2573   -0.4726    0.6854
   -0.7031    0.3779    0.0996    1.2584   -0.4714    0.6865
   -0.7021    0.3790    0.1008    1.2596   -0.4702    0.6876
   -0.7010    0.3800    0.1020    1.2607   -0.4690    0.6887
   -0.6999    0.3810    0.1032    1.2618   -0.4678    0.6898
   -0.6988    0.3820    0.1044    1.2630   -0.4666    0.6910
   -0.6977    0.3830    0.1055    1.2641   -0.4654    0.6921
   -0.6966    0.3840    0.1067    1.2653   -0.4642    0.6932
   -0.6955    0.3850    0.1079    1.2664   -0.4630    0.6943
   -0.6944    0.3860    0.1091    1.2675   -0.4618    0.6954
   -0.6933    0.3870    0.1103    1.2687   -0.4606    0.6965
   -0.6922    0.3881    0.1115    1.2698   -0.4594    0.6976
   -0.6911    0.3891    0.1126    1.2709   -0.4582    0.6987
   -0.6901    0.3901    0.1138    1.2720   -0.4570    0.6998
   -0.6890    0.3911    0.1150    1.2732   -0.4558    0.7009
   -0.6879    0.3921    0.1162    1.2743   -0.4546    0.7020
   -0.6868    0.3931    0.1173    1.2754   -0.4535    0.7031
   -0.6857    0.3941    0.1185    1.2765   -0.4523    0.7042
   -0.6846    0.3951    0.1197    1.2777   -0.4511    0.7053
   -0.6836    0.3961    0.1209    1.2788   -0.4499    0.7064
   -0.6825    0.3971    0.1220    1.2799   -0.4487    0.7075
   -0.6814    0.3980    0.1232    1.2810   -0.4476    0.7086
   -0.6803    0.3990    0.1244    1.2822   -0.4464    0.7097
   -0.6793    0.4000    0.1255    1.2833   -0.4452    0.7108
   -0.6782    0.4010    0.1267    1.2844   -0.4440    0.7119
   -0.6771    0.4020    0.1278    1.2855   -0.4429    0.7130
   -0.6760    0.4030    0.1290    1.2866   -0.4417    0.7140
   -0.6750    0.4040    0.1302    1.2877   -0.4405    0.7151
   -0.6739    0.4050    0.1313    1.2889   -0.4394    0.7162
   -0.6728    0.4060    0.1325    1.2900   -0.4382    0.7173
   -0.6717    0.4069    0.1336    1.2911   -0.4370    0.7183
   -0.6707    0.4079    0.1348    1.2922   -0.4359    0.7194
   -0.6696    0.4089    0.1359    1.2933   -0.4347    0.7205
   -0.6685    0.4099    0.1371    1.2944   -0.4335    0.7216
   -0.6675    0.4109    0.1382    1.2955   -0.4324    0.7226
   -0.6664    0.4118    0.1394    1.2966   -0.4312    0.7237
   -0.6653    0.4128    0.1405    1.2977   -0.4301    0.7248
   -0.6643    0.4138    0.1416    1.2988   -0.4289    0.7258
   -0.6632    0.4148    0.1428    1.3000   -0.4277    0.7269
   -0.6621    0.4157    0.1439    1.3011   -0.4266    0.7279
   -0.6611    0.4167    0.1451    1.3022   -0.4254    0.7290
   -0.6600    0.4177    0.1462    1.3033   -0.4243    0.7301
   -0.6589    0.4187    0.1473    1.3044   -0.4231    0.7311
   -0.6579    0.4196    0.1485    1.3055   -0.4220    0.7322
   -0.6568    0.4206    0.1496    1.3066   -0.4208    0.7332
   -0.6558    0.4216    0.1507    1.3077   -0.4197    0.7343
   -0.6547    0.4225    0.1518    1.3088   -0.4185    0.7353
   -0.6536    0.4235    0.1530    1.3099   -0.4174    0.7364
   -0.6526    0.4245    0.1541    1.3110   -0.4163    0.7374
   -0.6515    0.4254    0.1552    1.3121   -0.4151    0.7384
   -0.6504    0.4264    0.1563    1.3131   -0.4140    0.7395
   -0.6494    0.4273    0.1575    1.3142   -0.4128    0.7405
   -0.6483    0.4283    0.1586    1.3153   -0.4117    0.7416
   -0.6473    0.4293    0.1597    1.3164   -0.4106    0.7426
   -0.6462    0.4302    0.1608    1.3175   -0.4094    0.7437
   -0.6451    0.4312    0.1619    1.3186   -0.4083    0.7447
   -0.6441    0.4321    0.1631    1.3197   -0.4072    0.7457
   -0.6430    0.4331    0.1642    1.3208   -0.4060    0.7468
   -0.6420    0.4341    0.1653    1.3219   -0.4049    0.7478
   -0.6409    0.4350    0.1664    1.3230   -0.4038    0.7488
   -0.6399    0.4360    0.1675    1.3241   -0.4026    0.7498
   -0.6388    0.4369    0.1686    1.3251   -0.4015    0.7509
   -0.6377    0.4379    0.1697    1.3262   -0.4004    0.7519
   -0.6367    0.4388    0.1708    1.3273   -0.3992    0.7529
   -0.6356    0.4398    0.1719    1.3284   -0.3981    0.7540
   -0.6346    0.4407    0.1730    1.3295   -0.3970    0.7550
   -0.6335    0.4417    0.1742    1.3306   -0.3959    0.7560
   -0.6325    0.4426    0.1753    1.3317   -0.3948    0.7570
   -0.6314    0.4436    0.1764    1.3327   -0.3936    0.7581
   -0.6303    0.4445    0.1775    1.3338   -0.3925    0.7591
   -0.6293    0.4455    0.1786    1.3349   -0.3914    0.7601
   -0.6282    0.4464    0.1797    1.3360   -0.3903    0.7611
   -0.6272    0.4474    0.1808    1.3371   -0.3892    0.7621
   -0.6261    0.4483    0.1818    1.3382   -0.3880    0.7631
   -0.6251    0.4492    0.1829    1.3392   -0.3869    0.7642
   -0.6240    0.4502    0.1840    1.3403   -0.3858    0.7652
   -0.6229    0.4511    0.1851    1.3414   -0.3847    0.7662
   -0.6219    0.4521    0.1862    1.3425   -0.3836    0.7672
   -0.6208    0.4530    0.1873    1.3436   -0.3825    0.7682
   -0.6198    0.4539    0.1884    1.3446   -0.3814    0.7692
   -0.6187    0.4549    0.1895    1.3457   -0.3803    0.7702
   -0.6177    0.4558    0.1906    1.3468   -0.3791    0.7713
   -0.6166    0.4568    0.1917    1.3479   -0.3780    0.7723
   -0.6155    0.4577    0.1928    1.3490   -0.3769    0.7733
   -0.6145    0.4586    0.1938    1.3500   -0.3758    0.7743
   -0.6134    0.4596    0.1949    1.3511   -0.3747    0.7753
   -0.6124    0.4605    0.1960    1.3522   -0.3736    0.7763
   -0.6113    0.4614    0.1971    1.3533   -0.3725    0.7773
   -0.6103    0.4624    0.1982    1.3543   -0.3714    0.7783
   -0.6092    0.4633    0.1993    1.3554   -0.3703    0.7793
   -0.6081    0.4642    0.2004    1.3565   -0.3692    0.7803
   -0.6071    0.4652    0.2014    1.3576   -0.3681    0.7813
   -0.6060    0.4661    0.2025    1.3586   -0.3670    0.7823
   -0.6050    0.4670    0.2036    1.3597   -0.3659    0.7833
   -0.6039    0.4680    0.2047    1.3608   -0.3648    0.7843
   -0.6028    0.4689    0.2058    1.3619   -0.3637    0.7853
   -0.6018    0.4698    0.2068    1.3629   -0.3626    0.7863
   -0.6007    0.4708    0.2079    1.3640   -0.3615    0.7874
   -0.5997    0.4717    0.2090    1.3651   -0.3604    0.7884
   -0.5986    0.4726    0.2101    1.3662   -0.3593    0.7894
   -0.5975    0.4736    0.2111    1.3672   -0.3582    0.7904
   -0.5965    0.4745    0.2122    1.3683   -0.3571    0.7914
   -0.5954    0.4754    0.2133    1.3694   -0.3560    0.7924
   -0.5943    0.4763    0.2144    1.3705   -0.3549    0.7934
   -0.5933    0.4773    0.2155    1.3716   -0.3538    0.7944
   -0.5922    0.4782    0.2165    1.3726   -0.3527    0.7954
   -0.5912    0.4791    0.2176    1.3737   -0.3516    0.7964
   -0.5901    0.4801    0.2187    1.3748   -0.3505    0.7974
   -0.5890    0.4810    0.2198    1.3759   -0.3494    0.7984
   -0.5880    0.4819    0.2208    1.3769   -0.3484    0.7994
   -0.5869    0.4828    0.2219    1.3780   -0.3473    0.8004
   -0.5858    0.4838    0.2230    1.3791   -0.3462    0.8014
   -0.5848    0.4847    0.2241    1.3802   -0.3451    0.8024
   -0.5837    0.4856    0.2251    1.3812   -0.3440    0.8034
   -0.5826    0.4865    0.2262    1.3823   -0.3429    0.8044
   -0.5816    0.4875    0.2273    1.3834   -0.3418    0.8054
   -0.5805    0.4884    0.2284    1.3845   -0.3407    0.8064
   -0.5794    0.4893    0.2294    1.3855   -0.3396    0.8074
   -0.5784    0.4902    0.2305    1.3866   -0.3385    0.8084
   -0.5773    0.4912    0.2316    1.3877   -0.3374    0.8094
   -0.5762    0.4921    0.2326    1.3888   -0.3363    0.8104
   -0.5751    0.4930    0.2337    1.3899   -0.3352    0.8114
   -0.5741    0.4939    0.2348    1.3909   -0.3342    0.8124
   -0.5730    0.4948    0.2359    1.3920   -0.3331    0.8134
   -0.5719    0.4958    0.2369    1.3931   -0.3320    0.8144
   -0.5708    0.4967    0.2380    1.3942   -0.3309    0.8154
   -0.5698    0.4976    0.2391    1.3952   -0.3298    0.8164
   -0.5687    0.4985    0.2402    1.3963   -0.3287    0.8174
   -0.5676    0.4995    0.2412    1.3974   -0.3276    0.8184
   -0.5665    0.5004    0.2423    1.3985   -0.3265    0.8194
   -0.5655    0.5013    0.2434    1.3996   -0.3254    0.8204
   -0.5644    0.5022    0.2445    1.4006   -0.3243    0.8214
   -0.5633    0.5031    0.2455    1.4017   -0.3233    0.8224
   -0.5622    0.5041    0.2466    1.4028   -0.3222    0.8234
   -0.5612    0.5050    0.2477    1.4039   -0.3211    0.8244
   -0.5601    0.5059    0.2488    1.4050   -0.3200    0.8254
   -0.5590    0.5068    0.2498    1.4061   -0.3189    0.8264
   -0.5579    0.5078    0.2509    1.4071   -0.3178    0.8274
   -0.5568    0.5087    0.2520    1.4082   -0.3167    0.8284
   -0.5557    0.5096    0.2531    1.4093   -0.3156    0.8294
   -0.5547    0.5105    0.2541    1.4104   -0.3145    0.8304
   -0.5536    0.5114    0.2552    1.4115   -0.3134    0.8314
   -0.5525    0.5124    0.2563    1.4126   -0.3123    0.8325
   -0.5514    0.5133    0.2574    1.4136   -0.3112    0.8335
   -0.5503    0.5142    0.2585    1.4147   -0.3101    0.8345
   -0.5492    0.5151    0.2595    1.4158   -0.3091    0.8355
   -0.5481    0.5160    0.2606    1.4169   -0.3080    0.8365
   -0.5471    0.5170    0.2617    1.4180   -0.3069    0.8375
   -0.5460    0.5179    0.2628    1.4191   -0.3058    0.8385
   -0.5449    0.5188    0.2639    1.4202   -0.3047    0.8395
   -0.5438    0.5197    0.2650    1.4213   -0.3036    0.8405
   -0.5427    0.5207    0.2660    1.4223   -0.3025    0.8415
   -0.5416    0.5216    0.2671    1.4234   -0.3014    0.8425
   -0.5405    0.5225    0.2682    1.4245   -0.3003    0.8436
   -0.5394    0.5234    0.2693    1.4256   -0.2992    0.8446
   -0.5383    0.5243    0.2704    1.4267   -0.2981    0.8456
   -0.5372    0.5253    0.2715    1.4278   -0.2970    0.8466
   -0.5361    0.5262    0.2725    1.4289   -0.2959    0.8476
   -0.5350    0.5271    0.2736    1.4300   -0.2948    0.8486
   -0.5339    0.5280    0.2747    1.4311   -0.2937    0.8496
   -0.5328    0.5290    0.2758    1.4322   -0.2926    0.8506
   -0.5317    0.5299    0.2769    1.4333   -0.2915    0.8517
   -0.5306    0.5308    0.2780    1.4344   -0.2904    0.8527
   -0.5295    0.5317    0.2791    1.4355   -0.2893    0.8537
   -0.5284    0.5327    0.2802    1.4366   -0.2882    0.8547
   -0.5273    0.5336    0.2813    1.4377   -0.2871    0.8557
   -0.5262    0.5345    0.2824    1.4387   -0.2860    0.8567
   -0.5251    0.5354    0.2834    1.4398   -0.2849    0.8578
   -0.5240    0.5364    0.2845    1.4409   -0.2838    0.8588
   -0.5229    0.5373    0.2856    1.4420   -0.2827    0.8598
   -0.5218    0.5382    0.2867    1.4431   -0.2816    0.8608
   -0.5207    0.5391    0.2878    1.4442   -0.2805    0.8618
   -0.5196    0.5401    0.2889    1.4453   -0.2794    0.8629
   -0.5185    0.5410    0.2900    1.4464   -0.2783    0.8639
   -0.5174    0.5419    0.2911    1.4475   -0.2772    0.8649
   -0.5163    0.5428    0.2922    1.4486   -0.2761    0.8659
   -0.5151    0.5438    0.2933    1.4497   -0.2749    0.8670
   -0.5140    0.5447    0.2944    1.4508   -0.2738    0.8680
   -0.5129    0.5456    0.2955    1.4519   -0.2727    0.8690
   -0.5118    0.5465    0.2966    1.4530   -0.2716    0.8700
   -0.5107    0.5475    0.2977    1.4541   -0.2705    0.8711
   -0.5096    0.5484    0.2988    1.4553   -0.2694    0.8721
   -0.5084    0.5493    0.2999    1.4564   -0.2683    0.8731
   -0.5073    0.5503    0.3010    1.4575   -0.2672    0.8741
   -0.5062    0.5512    0.3022    1.4586   -0.2660    0.8752
   -0.5051    0.5521    0.3033    1.4597   -0.2649    0.8762
   -0.5040    0.5531    0.3044    1.4608   -0.2638    0.8772
   -0.5028    0.5540    0.3055    1.4619   -0.2627    0.8783
   -0.5017    0.5549    0.3066    1.4630   -0.2616    0.8793
   -0.5006    0.5558    0.3077    1.4641   -0.2605    0.8803
   -0.4995    0.5568    0.3088    1.4652   -0.2593    0.8814
   -0.4983    0.5577    0.3099    1.4663   -0.2582    0.8824
   -0.4972    0.5586    0.3110    1.4674   -0.2571    0.8834
   -0.4961    0.5596    0.3122    1.4685   -0.2560    0.8845
   -0.4949    0.5605    0.3133    1.4696   -0.2549    0.8855
   -0.4938    0.5614    0.3144    1.4707   -0.2537    0.8865
   -0.4927    0.5624    0.3155    1.4719   -0.2526    0.8876
   -0.4915    0.5633    0.3166    1.4730   -0.2515    0.8886
   -0.4904    0.5643    0.3177    1.4741   -0.2504    0.8896
   -0.4893    0.5652    0.3189    1.4752   -0.2492    0.8907
   -0.4881    0.5661    0.3200    1.4763   -0.2481    0.8917
   -0.4870    0.5671    0.3211    1.4774   -0.2470    0.8928
   -0.4859    0.5680    0.3222    1.4785   -0.2459    0.8938
   -0.4847    0.5689    0.3234    1.4796   -0.2447    0.8948
   -0.4836    0.5699    0.3245    1.4807   -0.2436    0.8959
   -0.4824    0.5708    0.3256    1.4818   -0.2425    0.8969
   -0.4813    0.5717    0.3267    1.4830   -0.2413    0.8979
   -0.4801    0.5727    0.3279    1.4841   -0.2402    0.8990
   -0.4790    0.5736    0.3290    1.4852   -0.2391    0.9000
   -0.4779    0.5746    0.3301    1.4863   -0.2379    0.9011
   -0.4767    0.5755    0.3313    1.4874   -0.2368    0.9021
   -0.4756    0.5764    0.3324    1.4885   -0.2357    0.9032
   -0.4744    0.5774    0.3335    1.4896   -0.2345    0.9042
   -0.4733    0.5783    0.3346    1.4907   -0.2334    0.9052
   -0.4721    0.5793    0.3358    1.4918   -0.2323    0.9063
   -0.4710    0.5802    0.3369    1.4930   -0.2311    0.9073
   -0.4698    0.5811    0.3380    1.4941   -0.2300    0.9084
   -0.4686    0.5821    0.3392    1.4952   -0.2289    0.9094
   -0.4675    0.5830    0.3403    1.4963   -0.2277    0.9105
   -0.4663    0.5840    0.3415    1.4974   -0.2266    0.9115
   -0.4652    0.5849    0.3426    1.4985   -0.2254    0.9126
   -0.4640    0.5859    0.3437    1.4996   -0.2243    0.9136
   -0.4629    0.5868    0.3449    1.5007   -0.2232    0.9146
   -0.4617    0.5878    0.3460    1.5019   -0.2220    0.9157
   -0.4605    0.5887    0.3471    1.5030   -0.2209    0.9167
   -0.4594    0.5896    0.3483    1.5041   -0.2197    0.9178
   -0.4582    0.5906    0.3494    1.5052   -0.2186    0.9188
   -0.4571    0.5915    0.3506    1.5063   -0.2174    0.9199
   -0.4559    0.5925    0.3517    1.5074   -0.2163    0.9209
   -0.4547    0.5934    0.3529    1.5085   -0.2152    0.9220
   -0.4536    0.5944    0.3540    1.5097   -0.2140    0.9230
   -0.4524    0.5953    0.3552    1.5108   -0.2129    0.9241
   -0.4512    0.5963    0.3563    1.5119   -0.2117    0.9251
   -0.4501    0.5972    0.3574    1.5130   -0.2106    0.9262
   -0.4489    0.5982    0.3586    1.5141   -0.2094    0.9272
   -0.4477    0.5991    0.3597    1.5152   -0.2083    0.9283
   -0.4465    0.6001    0.3609    1.5163   -0.2071    0.9293
   -0.4454    0.6010    0.3620    1.5175   -0.2060    0.9304
   -0.4442    0.6020    0.3632    1.5186   -0.2048    0.9314
   -0.4430    0.6029    0.3643    1.5197   -0.2037    0.9325
   -0.4418    0.6039    0.3655    1.5208   -0.2025    0.9335
   -0.4407    0.6048    0.3666    1.5219   -0.2013    0.9346
   -0.4395    0.6058    0.3678    1.5230   -0.2002    0.9356
   -0.4383    0.6067    0.3690    1.5241   -0.1990    0.9367
   -0.4371    0.6077    0.3701    1.5253   -0.1979    0.9377
   -0.4360    0.6087    0.3713    1.5264   -0.1967    0.9388
   -0.4348    0.6096    0.3724    1.5275   -0.1956    0.9398
   -0.4336    0.6106    0.3736    1.5286   -0.1944    0.9409
   -0.4324    0.6115    0.3747    1.5297   -0.1932    0.9420
   -0.4312    0.6125    0.3759    1.5308   -0.1921    0.9430
   -0.4300    0.6134    0.3771    1.5320   -0.1909    0.9441
   -0.4288    0.6144    0.3782    1.5331   -0.1898    0.9451
   -0.4277    0.6154    0.3794    1.5342   -0.1886    0.9462
   -0.4265    0.6163    0.3805    1.5353   -0.1874    0.9472
   -0.4253    0.6173    0.3817    1.5364   -0.1863    0.9483
   -0.4241    0.6183    0.3829    1.5376   -0.1851    0.9494
   -0.4229    0.6192    0.3840    1.5387   -0.1839    0.9504
   -0.4217    0.6202    0.3852    1.5398   -0.1828    0.9515
   -0.4205    0.6212    0.3864    1.5409   -0.1816    0.9526
   -0.4193    0.6221    0.3875    1.5421   -0.1804    0.9536
   -0.4181    0.6231    0.3887    1.5432   -0.1792    0.9547
   -0.4169    0.6241    0.3899    1.5443   -0.1781    0.9557
   -0.4157    0.6250    0.3910    1.5454   -0.1769    0.9568
   -0.4145    0.6260    0.3922    1.5466   -0.1757    0.9579
   -0.4133    0.6270    0.3934    1.5477   -0.1745    0.9590
   -0.4121    0.6279    0.3946    1.5488   -0.1734    0.9600
   -0.4109    0.6289    0.3957    1.5499   -0.1722    0.9611
   -0.4097    0.6299    0.3969    1.5511   -0.1710    0.9622
   -0.4085    0.6309    0.3981    1.5522   -0.1698    0.9632
   -0.4073    0.6319    0.3993    1.5533   -0.1686    0.9643
   -0.4061    0.6328    0.4005    1.5545   -0.1674    0.9654
   -0.4048    0.6338    0.4016    1.5556   -0.1663    0.9665
   -0.4036    0.6348    0.4028    1.5567   -0.1651    0.9676
   -0.4024    0.6358    0.4040    1.5579   -0.1639    0.9686
   -0.4012    0.6368    0.4052    1.5590   -0.1627    0.9697
   -0.4000    0.6377    0.4064    1.5601   -0.1615    0.9708
   -0.3987    0.6387    0.4076    1.5613   -0.1603    0.9719
   -0.3975    0.6397    0.4088    1.5624   -0.1591    0.9730
   -0.3963    0.6407    0.4100    1.5636   -0.1579    0.9741
   -0.3951    0.6417    0.4112    1.5647   -0.1567    0.9751
   -0.3938    0.6427    0.4124    1.5658   -0.1555    0.9762
   -0.3926    0.6437    0.4136    1.5670   -0.1543    0.9773
   -0.3914    0.6447    0.4148    1.5681   -0.1531    0.9784
   -0.3901    0.6457    0.4160    1.5693   -0.1519    0.9795
   -0.3889    0.6467    0.4172    1.5704   -0.1507    0.9806
   -0.3877    0.6477    0.4184    1.5716   -0.1495    0.9817
   -0.3864    0.6487    0.4196    1.5727   -0.1483    0.9828
   -0.3852    0.6497    0.4208    1.5739   -0.1471    0.9839
   -0.3840    0.6507    0.4220    1.5750   -0.1459    0.9850
   -0.3827    0.6517    0.4232    1.5762   -0.1446    0.9861
   -0.3815    0.6527    0.4244    1.5773   -0.1434    0.9872
   -0.3802    0.6537    0.4256    1.5785   -0.1422    0.9883
   -0.3790    0.6547    0.4268    1.5796   -0.1410    0.9894
   -0.3777    0.6557    0.4280    1.5808   -0.1398    0.9906
   -0.3765    0.6567    0.4292    1.5819   -0.1386    0.9917
   -0.3752    0.6577    0.4304    1.5831   -0.1373    0.9928
   -0.3740    0.6587    0.4317    1.5842   -0.1361    0.9939
   -0.3727    0.6597    0.4329    1.5854   -0.1349    0.9950
   -0.3715    0.6607    0.4341    1.5865   -0.1337    0.9961
   -0.3702    0.6617    0.4353    1.5877   -0.1325    0.9972
   -0.3690    0.6627    0.4365    1.5889   -0.1312    0.9984
   -0.3677    0.6637    0.4377    1.5900   -0.1300    0.9995
   -0.3665    0.6648    0.4390    1.5912   -0.1288    1.0006
   -0.3652    0.6658    0.4402    1.5923   -0.1276    1.0017
   -0.3640    0.6668    0.4414    1.5935   -0.1263    1.0029
   -0.3627    0.6678    0.4426    1.5946   -0.1251    1.0040
   -0.3615    0.6688    0.4438    1.5958   -0.1239    1.0051
   -0.3602    0.6698    0.4451    1.5969   -0.1227    1.0062
   -0.3590    0.6708    0.4463    1.5981   -0.1214    1.0074
   -0.3577    0.6718    0.4475    1.5993   -0.1202    1.0085
   -0.3564    0.6728    0.4487    1.6004   -0.1190    1.0096
   -0.3552    0.6739    0.4500    1.6016   -0.1177    1.0107
   -0.3539    0.6749    0.4512    1.6027   -0.1165    1.0119
   -0.3527    0.6759    0.4524    1.6039   -0.1153    1.0130
   -0.3514    0.6769    0.4536    1.6050   -0.1141    1.0141
   -0.3502    0.6779    0.4548    1.6062   -0.1128    1.0152
   -0.3489    0.6789    0.4561    1.6074   -0.1116    1.0164
   -0.3477    0.6799    0.4573    1.6085   -0.1104    1.0175
   -0.3465    0.6809    0.4585    1.6097   -0.1092    1.0186
   -0.3452    0.6820    0.4597    1.6108   -0.1079    1.0198
   -0.3440    0.6830    0.4609    1.6120   -0.1067    1.0209
   -0.3427    0.6840    0.4622    1.6131   -0.1055    1.0220
   -0.3415    0.6850    0.4634    1.6143   -0.1043    1.0231
   -0.3403    0.6860    0.4646    1.6154   -0.1030    1.0243
   -0.3390    0.6870    0.4658    1.6166   -0.1018    1.0254
   -0.3378    0.6880    0.4670    1.6177   -0.1006    1.0265
   -0.3366    0.6890    0.4682    1.6189   -0.0994    1.0276
   -0.3353    0.6900    0.4694    1.6200   -0.0982    1.0288
   -0.3341    0.6910    0.4707    1.6212   -0.0969    1.0299
   -0.3329    0.6920    0.4719    1.6223   -0.0957    1.0310
   -0.3316    0.6931    0.4731    1.6235   -0.0945    1.0322
   -0.3304    0.6941    0.4743    1.6246   -0.0933    1.0333
   -0.3292    0.6951    0.4755    1.6257   -0.0921    1.0344
   -0.3280    0.6961    0.4767    1.6269   -0.0908    1.0355
   -0.3268    0.6971    0.4779    1.6280   -0.0896    1.0367
   -0.3255    0.6981    0.4791    1.6292   -0.0884    1.0378
   -0.3243    0.6991    0.4803    1.6303   -0.0872    1.0389
   -0.3231    0.7001    0.4815    1.6315   -0.0860    1.0400
   -0.3219    0.7011    0.4828    1.6326   -0.0847    1.0412
   -0.3207    0.7021    0.4840    1.6338   -0.0835    1.0423
   -0.3195    0.7031    0.4852    1.6349   -0.0823    1.0434
   -0.3183    0.7041    0.4864    1.6360   -0.0811    1.0446
   -0.3171    0.7051    0.4876    1.6372   -0.0799    1.0457
   -0.3158    0.7062    0.4888    1.6383   -0.0787    1.0468
   -0.3146    0.7072    0.4900    1.6395   -0.0774    1.0479
   -0.3134    0.7082    0.4912    1.6406   -0.0762    1.0491
   -0.3122    0.7092    0.4924    1.6418   -0.0750    1.0502
   -0.3110    0.7102    0.4936    1.6429   -0.0738    1.0513
   -0.3098    0.7112    0.4949    1.6441   -0.0725    1.0525
   -0.3086    0.7122    0.4961    1.6452   -0.0713    1.0536
   -0.3074    0.7133    0.4973    1.6464   -0.0701    1.0548
   -0.3061    0.7143    0.4985    1.6476   -0.0688    1.0559
   -0.3049    0.7153    0.4997    1.6487   -0.0676    1.0570
   -0.3037    0.7163    0.5010    1.6499   -0.0664    1.0582
   -0.3025    0.7174    0.5022    1.6511   -0.0651    1.0593
   -0.3012    0.7184    0.5034    1.6522   -0.0639    1.0605
   -0.3000    0.7194    0.5047    1.6534   -0.0626    1.0617
   -0.2988    0.7205    0.5059    1.6546   -0.0614    1.0628
   -0.2975    0.7215    0.5072    1.6558   -0.0601    1.0640
   -0.2963    0.7225    0.5084    1.6569   -0.0589    1.0651
   -0.2951    0.7236    0.5097    1.6581   -0.0576    1.0663
   -0.2938    0.7246    0.5109    1.6593   -0.0563    1.0675
   -0.2926    0.7257    0.5122    1.6605   -0.0551    1.0686
```

#### OK

Now we write an animation file for this.

```octave fold title:Animation.m
% --- Configuration ---
subjectIdx = 1;  % Which person?
trialIdx = 1;    % Which step/trial? (Column 1)
frameRate = 0.001; % Slow down animation

% Ensure data is loaded (Run this only once)
if ~exist('Sub', 'var')
    disp('Loading data... this might take a moment.');
    load('MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat'); 
end

% Shortcut to the specific trial data container (Using Left Side Time Basis)
% We use the 'LsideSegm' version so all body parts are synchronized to the same timeline
S = Sub(subjectIdx);
Base = S.LsideSegm_BsideData;
Left = S.LsideSegm_LsideData;
Right = S.LsideSegm_RsideData;

% --- Skeleton Definition (Parent -> Child connections) ---
% Format: {SourceStruct, 'SourceLabel', TargetStruct, 'TargetLabel'}
skeleton_map = {
    % Spine
    Base, 'PELO', Base, 'TRXO';
    Base, 'TRXO', Base, 'HEDO';
    
    % Left Leg (Pelvis -> Femur -> Tibia -> Foot -> Toe)
    Base, 'PELO', Left, 'FEO';
    Left, 'FEO',  Left, 'TIO';
    Left, 'TIO',  Left, 'FOO';
    Left, 'FOO',  Left, 'TOO';
    
    % Right Leg
    Base, 'PELO', Right, 'FEO';
    Right, 'FEO',  Right, 'TIO';
    Right, 'TIO',  Right, 'FOO';
    Right, 'FOO',  Right, 'TOO';
    
    % Left Arm (Thorax -> Clavicle -> Humerus -> Radius -> Hand)
    Base, 'TRXO', Left, 'CLO';
    Left, 'CLO',  Left, 'HUO';
    Left, 'HUO',  Left, 'RAO';
    Left, 'RAO',  Left, 'HNO';
    
    % Right Arm
    Base, 'TRXO', Right, 'CLO';
    Right, 'CLO',  Right, 'HUO';
    Right, 'HUO',  Right, 'RAO';
    Right, 'RAO',  Right, 'HNO';
};


% Initialize Figure
figure(1); clf;
axis equal; grid on; hold on;
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
view(3); % Set 3D view

% Get number of frames (rows) from a random marker (e.g., PELO)
numFrames = size(Base.PELO.x, 1);

% Prepare an array to store line handles so we can update them efficiently
lineHandles = gobjects(size(skeleton_map, 1), 1);

% --- Animation Loop ---

for f = 1:numFrames
    
    % Loop through every bone in our map
    for i = 1:size(skeleton_map, 1)
        
        % 1. Extract Source Point (P1)
        srcStruct = skeleton_map{i, 1};
        srcLabel  = skeleton_map{i, 2};
        
        P1 = [srcStruct.(srcLabel).x(f, trialIdx), ...
              srcStruct.(srcLabel).y(f, trialIdx), ...
              srcStruct.(srcLabel).z(f, trialIdx)];
          
        % 2. Extract Target Point (P2)
        tgtStruct = skeleton_map{i, 3};
        tgtLabel  = skeleton_map{i, 4};
        
        P2 = [tgtStruct.(tgtLabel).x(f, trialIdx), ...
              tgtStruct.(tgtLabel).y(f, trialIdx), ...
              tgtStruct.(tgtLabel).z(f, trialIdx)];
          
        % 3. Draw or Update Line
        if f == 1
            % First frame: Create the line object
            lineHandles(i) = plot3([P1(1), P2(1)], [P1(2), P2(2)], [P1(3), P2(3)], ...
                'LineWidth', 2, 'Color', 'b', 'Marker', 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
        else
            % Subsequent frames: Just update coordinates (Much faster)
            set(lineHandles(i), 'XData', [P1(1), P2(1)], ...
                                'YData', [P1(2), P2(2)], ...
                                'ZData', [P1(3), P2(3)]);
        end
    end
    
    % Set dynamic axis limits based on Pelvis position to follow the walker
    pelvisX = Base.PELO.x(f, trialIdx);
    pelvisY = Base.PELO.y(f, trialIdx);
    pelvisZ = Base.PELO.z(f, trialIdx);
    
    xlim([pelvisX - 1000, pelvisX + 1000]);
    ylim([pelvisY - 1000, pelvisY + 1000]);
    zlim([0, 2000]); % Height is roughly constant
    
    title(sprintf('Subject %d, Trial %d, Frame %d', subjectIdx, trialIdx, f));
    drawnow;
    pause(frameRate);
end
```

---

