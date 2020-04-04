# OCT_Denoising

1. cGAN is the ISBI paper version of structure
2. Multiscale_cGAN is most recent
3. EP_cGAN+(train, test)+(human, mice)

## Self-fusion + conditional GAN
0. DataLoader is done in MotionCorrection-VoxelMorph
1. Train: SF_cGAN_train.py
2. Test: SF_cGAN_test.py

## Compare models
### Traditional method
1. Self-fusion
2. BM3D
3. K-SVD
### learning based method
1. Multiscale_cGAN
2. Patch GAN
3. GAN-UNet
4. SRResNet
