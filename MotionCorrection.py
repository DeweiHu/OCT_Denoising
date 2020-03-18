# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 07:51:29 2020

@author: hudew
"""

import itk

root = 'E:\\OCT_human\\'
fixedImageFile = root+'fix_img.nii.gz'
movingImageFile = root+'mov_img8.nii.gz'
outputImageFile = root+'opt8.nii.gz'

def MotionCorrect(fixedImageFile,movingImageFile,outputImageFile):

    PixelType = itk.ctype('float')

    fixedImage = itk.imread(fixedImageFile, PixelType)
    movingImage = itk.imread(movingImageFile, PixelType)

    Dimension = fixedImage.GetImageDimension()
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]
    
    TransformType = itk.TranslationTransform[itk.D, Dimension]
    initialTransform = TransformType.New()
    
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
            LearningRate=4,
            MinimumStepLength=0.001,
            RelaxationFactor=0.5,
            NumberOfIterations=200)
    
    metric = itk.MeanSquaresImageToImageMetricv4[
        FixedImageType, MovingImageType].New()
    
    registration = itk.ImageRegistrationMethodv4.New(FixedImage=fixedImage,
            MovingImage=movingImage,
            Metric=metric,
            Optimizer=optimizer,
            InitialTransform=initialTransform)
    
    movingInitialTransform = TransformType.New()
    initialParameters = movingInitialTransform.GetParameters()
    initialParameters[0] = 0
    initialParameters[1] = 0
    movingInitialTransform.SetParameters(initialParameters)
    registration.SetMovingInitialTransform(movingInitialTransform)
    
    identityTransform = TransformType.New()
    identityTransform.SetIdentity()
    registration.SetFixedInitialTransform(identityTransform)
    
    registration.SetNumberOfLevels(1)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])
    
    registration.Update()
    
    transform = registration.GetTransform()
    finalParameters = transform.GetParameters()
    translationAlongX = finalParameters.GetElement(0)
    translationAlongY = finalParameters.GetElement(1)
    
    numberOfIterations = optimizer.GetCurrentIteration()
    
    bestValue = optimizer.GetValue()
    
    print("Result = ")
    print(" Translation X = " + str(translationAlongX))
    print(" Translation Y = " + str(translationAlongY))
    print(" Iterations    = " + str(numberOfIterations))
    print(" Metric value  = " + str(bestValue))
    
    CompositeTransformType = itk.CompositeTransform[itk.D, Dimension]
    outputCompositeTransform = CompositeTransformType.New()
    outputCompositeTransform.AddTransform(movingInitialTransform)
    outputCompositeTransform.AddTransform(registration.GetModifiableTransform())
    
    resampler = itk.ResampleImageFilter.New(Input=movingImage,
            Transform=outputCompositeTransform,
            UseReferenceImage=True,
            ReferenceImage=fixedImage)
    resampler.SetDefaultPixelValue(100)
    
    OutputImageType = itk.Image[PixelType, Dimension]
    
    caster = itk.CastImageFilter[FixedImageType,
            OutputImageType].New(Input=resampler)
    
    writer = itk.ImageFileWriter.New(Input=caster, FileName=outputImageFile)
    writer.SetFileName(outputImageFile)
    writer.Update()