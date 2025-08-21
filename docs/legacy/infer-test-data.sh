TEST_SINGLE_VIDEO=/media/bgeuther/Storage/TempStorage/onnx/onnx-pipelines/tests/test-single-vid.avi
TEST_SINGLE_POSE=/media/bgeuther/Storage/TempStorage/onnx/onnx-pipelines/tests/test-single-vid_pose_est_v6.h5
TEST_SINGLE_RENDER=/media/bgeuther/Storage/TempStorage/onnx/onnx-pipelines/tests/test-single-vid_pose.avi
TEST_MULTI_VIDEO=/media/bgeuther/Storage/TempStorage/onnx/onnx-pipelines/tests/test-multi-vid.mp4
TEST_MULTI_POSE=/media/bgeuther/Storage/TempStorage/onnx/onnx-pipelines/tests/test-multi-vid_pose_est_v6.h5
TEST_MULTI_RENDER=/media/bgeuther/Storage/TempStorage/onnx/onnx-pipelines/tests/test-multi-vid_pose.avi

python3 mouse-tracking-runtime/infer_single_pose.py --video "${TEST_SINGLE_VIDEO}" --out-file "${TEST_SINGLE_POSE}" --batch-size 10
python3 mouse-tracking-runtime/infer_arena_corner.py --video "${TEST_SINGLE_VIDEO}" --out-file "${TEST_SINGLE_POSE}"
python3 mouse-tracking-runtime/infer_single_segmentation.py --video "${TEST_SINGLE_VIDEO}" --out-file "${TEST_SINGLE_POSE}"
python3 mouse-tracking-runtime/infer_fecal_boli.py --video "${TEST_SINGLE_VIDEO}" --out-file "${TEST_SINGLE_POSE}"
python3 mouse-tracking-runtime/render_pose.py --in-vid "${TEST_SINGLE_VIDEO}" --in-pose "${TEST_SINGLE_POSE}" --out-vid ${TEST_SINGLE_RENDER}

python3 mouse-tracking-runtime/infer_multi_segmentation.py --video "${TEST_MULTI_VIDEO}" --out-file "${TEST_MULTI_POSE}"
python3 mouse-tracking-runtime/infer_multi_pose.py --video "${TEST_MULTI_VIDEO}" --out-file "${TEST_MULTI_POSE}"
python3 mouse-tracking-runtime/infer_multi_identity.py --model 2023 --video "${TEST_MULTI_VIDEO}" --out-file "${TEST_MULTI_POSE}"
python3 mouse-tracking-runtime/stitch_tracklets.py --in-pose "${TEST_MULTI_POSE}"
python3 mouse-tracking-runtime/infer_arena_corner.py --video "${TEST_MULTI_VIDEO}" --out-file "${TEST_MULTI_POSE}"
python3 mouse-tracking-runtime/infer_food_hopper.py --video "${TEST_MULTI_VIDEO}" --out-file "${TEST_MULTI_POSE}"
python3 mouse-tracking-runtime/infer_lixit.py --video "${TEST_MULTI_VIDEO}" --out-file "${TEST_MULTI_POSE}"
python3 mouse-tracking-runtime/render_pose.py --in-vid "${TEST_MULTI_VIDEO}" --in-pose "${TEST_MULTI_POSE}" --out-vid ${TEST_MULTI_RENDER}
