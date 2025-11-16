# submission. full project

# Tracking
> Track objects over time with a Kalman Filter

mean RMSE is 0.31
Image: ![RMSE](rsme_track_0.png)
Track: ![Tracking](tracking.png)

# Track Management
> Initialize, update and delete tracks

Please upload the RMSE plot as png or pdf file.

Mean RSME: 0.78
Image: ![RMSE](step2_rsme.png)


# Data Association
> Associate measurements to tracks with nearest neighbor association

Please upload the RMSE plot as png or pdf file.

Total tracks: 10
RSME: ![RSME](step3_rsme.png)
End frame: ![End frame](step3_endframe.png)

# Sensor Fusion
> SWBAT fuse measurements from lidar and camera

I am seeing some issues with ghost track being confirmed after introducing camera.
But the main tracks are still consistent (track 0, 1, 5)
Video: <video controls src="step4.mp4" title="step 4"></video>
RSME: ![rsme](step4_rsme.png)
End frame: ![end frame](step4_endframe.png)










