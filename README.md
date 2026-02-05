# ISTA_498_Capstone
Replay data will be collected through ballchasing.gg API and processed to extract behavioral features across various rank tiers. 
Unsupervised learning techniques are first employed to construct rank-specific behavioral profiles that characterize gameplay patterns 
within each competitive tier. Players whose early gameplay exhibits significant deviation from these rank-specific behavioral 
distributions are assigned a smurf likelihood score based on behavioral abnormality. To complement this unsupervised profiling, a neural 
network model is trained on structured behavioral feature sequences to learn nonlinear and temporal relationships indicative of smurfing. 
The neural model produces an additional smurf likelihood estimate, which can be combined with the unsupervised deviation score to improve 
robustness and detection accuracy. 
