# SEARCH -> PAN_ALIGN -> APPROACHING. [ ] , . move arm (PREPOSITION); - = depth.
$env:PYTHONUNBUFFERED = "1"
if (-not $env:LEROBOT_LOG_LEVEL) { $env:LEROBOT_LOG_LEVEL = "INFO" }

# Change COM3 below to match Device Manager -> Ports (COM & LPT).
$PORT = "COM4"

# Path to the lerobot repo so relative paths (./SO101/..., ./yolov8s...) work.
$LEROBOT_DIR = "$env:USERPROFILE\Documents\lerobot"

Push-Location $LEROBOT_DIR
try {
    lerobot-gaze-engine `
      --robot.type=so101_follower `
      --robot.port=$PORT `
      '--robot.cameras={"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' `
      --urdf=./SO101/so101_new_calib.urdf `
      --query="red cube, toy block, red box" `
      --model-path=./yolov8s-worldv2.pt `
      '--gripper-camera-tf=-0.0503,0.0906,-0.1730,-0.2921,1.0770,-2.1688' `
      --gripper-camera-pitch-trim-deg=0 `
      --target-physical-size-m=0.03 `
      --bbox-depth-scale=1.0 `
      --bbox-depth-offset-m=0.02 `
      --final-standoff-m=0.06 `
      --grasp-enable=true `
      --grasp-trigger-standoff-m=0.10 `
      --grasp-final-approach-m=0.04 `
      --grasp-final-approach-along-optical=true `
      --grasp-contact-delta-current-counts=8 `
      --grasp-post-contact-squeeze-pct=8.0 `
      --grasp-lift-confirm=true `
      --search-startup-probe=true `
      --search-startup-lock-frames=2 `
      --search-min-detection-confidence=0.10 `
      --approach-el-deg=60 `
      --approach-steep-always-optical=true `
      --approach-coarse-look-at=false `
      --approach-coarse-ik-orientation-weight=0 `
      --approach-depth-priority=true `
      --approach-max-lin-vel-m-s=0.032 `
      --approach-gaze-scale-coarse=0.35 `
      --gaze-kp-tilt=0.48 `
      --approach-gaze-max-tilt-deg-coarse=3.0 `
      --approach-pause-vertical-err-px=0 `
      --preposition-enabled=false `
      --live-control-stdin=true `
      --live-control-keypress=true `
      --live-keys-auto-preposition=true `
      --live-keys-snap-targets=true `
      --live-key-max-steps-per-tick=8 `
      --live-el-slew-deg-s=72 `
      --live-preposition-boost-duration-s=1.2 `
      --live-preposition-boost-lin-vel-m-s=0.16 `
      --display-data=true `
      --display-sim3d=true
} finally {
    Pop-Location
}
