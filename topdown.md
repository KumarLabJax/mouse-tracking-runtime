###### Top-down approach 

*Using the masks create a framework to extract keypoints only form a masked image of the mouse.*

- Imperfect masks could lead to wrong keypoints

Tasks:
- [ ] Create DataLoader
- [ ] Train
- [ ] Evaluate

## how to create dataloader?
in my first attempt I plan to manipulate `MultiPoseDataset`. 
right now it returns:
```
'image': img,
'joint_heatmaps': joint_heatmaps,
'pose_instances': pose_instances_tensor,
'instance_count': len(pose_instances)
```
which I'm changing so return a masked image instead with instance_count of 1 and pose instance of the corresponding masked mouse and joint_heatmap only created for the one 

- changed max number of instances to 1
- OUTPUT_CHANNELS_PER_JOINT should be set to 1 as we do only one mouse and there is no need for embedding space
this means we have 12 output channels for 12 joints
- removed all code related to the embedding space
- the masked input is 1D, previously it was RGB, how to handle that?

