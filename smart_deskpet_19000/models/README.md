# Model Files

Large model files for the `smart_deskpet_19000` stack are stored here.

Most files are tracked with Git LFS. The RKLLM file is split into parts because the original file is about 2.1 GB.

## Path Mapping

- `vision/yolo11n.rknn` -> `/home/linaro/yolo11n.rknn`
- `vision/bestxuboran.rknn` -> `/home/linaro/bestxuboran.rknn`
- `vision/bestrenlian22.rknn` -> `/userdata/bestrenlian22.rknn`
- `vision/best_v2.rknn` -> `/userdata/best_v2.rknn`
- `speaker/voxblink2_samresnet34_fp_rk3588.rknn` -> `/userdata/voxblink2_samresnet34_fp_rk3588.rknn`
- `voice_rknn/vits_no_split.rknn` -> `/userdata/rknn_voice_test/models/vits_no_split.rknn`
- `voice_rknn/vits_slice_reshape.rknn` -> `/userdata/rknn_voice_test/models/vits_slice_reshape.rknn`
- `gpt_sovits/ldnn-e15.ckpt` -> `/home/linaro/GPT_weights_v2Pro/ldnn-e15.ckpt`
- `gpt_sovits/xxx_e8_s640.pth` -> `/home/linaro/SoVITS_weights_v2Pro/xxx_e8_s640.pth`
- `gpt_sovits/witch.wav` -> `/home/linaro/witch.wav`
- `rkllm_parts/Qwen.rkllm.part*` -> combine to `/userdata/models/Qwen.rkllm`

## Restore RKLLM

On Linux:

```bash
cd smart_deskpet_19000/models/rkllm_parts
cat Qwen.rkllm.part* > Qwen.rkllm
sha256sum Qwen.rkllm
```

Expected SHA256:

```text
a481430214635cfd8e8095087b366594c6b7467dde764a27817ea1535703edd3
```

Then copy:

```bash
sudo mkdir -p /userdata/models
sudo cp Qwen.rkllm /userdata/models/Qwen.rkllm
```
