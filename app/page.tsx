"use client";

import Image from 'next/image'
import { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;


let model: any = null;

export default function Home() {
  
  // states
  const STOP_DATA_GATHER = -1;
  const CLASS_NAMES = ["1", "2"];

  const VIDEO = useRef<HTMLVideoElement>(null);
 

  const [gatherDataState, setGatherDataState] = useState(STOP_DATA_GATHER);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [trainingDataInput, setTrainingDataInputs] = useState<any>([]);
  const [trainingDataOutputs, setTrainingDataOutputs] = useState<any>([]);
  const [examplesCount, setExamplesCount] = useState<any>([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [status, setStatus] = useState('Awaiting TF.js load');
  const [mobilenet, setMobilenet] = useState<any>(undefined);
  const [animationFrame, setAnimationFrame] = useState<any>([undefined, undefined]);
  
  const [predictIntervalId, setPredictIntervalId] = useState<any>(null);
  const [xCoordRight, setXCoordRight] = useState(0);
  const [yCoordRight, setYCoordRight] = useState(0);

  const [xCoordLeft, setXCoordLeft] = useState(0);
  const [yCoordLeft, setYCoordLeft] = useState(0);


  const pointerRefRight = useRef<HTMLDivElement>(null);
  const pointerRefLeft = useRef<HTMLDivElement>(null);

  function enableCam() {
    if (hasGetUserMedia() && VIDEO.current) {
      // getUsermedia parameters.
      const constraints = {
        video: true,
        width: 640, 
        height: 480 
      };
  
      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        VIDEO.current!.srcObject = stream;
        VIDEO.current!.addEventListener('loadeddata', function() {
          setVideoPlaying(true);
        });
      });
    } else {
      console.warn('getUserMedia() is not supported by your browser');
    }
  }

  async function loadMobileNetFeatureModel() {
    const URL = 
      'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
    
    const mobi = await tf.loadGraphModel(URL, {fromTFHub: true});
    setStatus('MobileNet v3 loaded successfully!');
    
    // Warm up the model by passing zeros through it once.
    tf.tidy(function () {
      let answer = mobi.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
      console.log(answer);
    });
    setMobilenet(mobi);
  }

  function logProgress(epoch: any, logs: any) {
    console.log('Data for epoch ' + epoch, logs);
  }

  async function trainAndPredict() {
    tf.util.shuffleCombo(trainingDataInput, trainingDataOutputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    let inputsAsTensor = tf.stack(trainingDataInput);
    
    let results = await model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epochs: 10, 
        callbacks: {onEpochEnd: logProgress} });
    
    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    setIsPredicting(true)
    predictLoop();
  }

  function predictLoop() {
    if (isPredicting) {
      tf.tidy(function() {
        let videoFrameAsTensor: any = tf.browser.fromPixels(VIDEO.current!).div(255);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor,[MOBILE_NET_INPUT_HEIGHT, 
            MOBILE_NET_INPUT_WIDTH], true);
  
        let imageFeatures: any = mobilenet.predict(resizedTensorFrame.expandDims());
        let prediction = model.predict(imageFeatures).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();
  
        setStatus('Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence');
      });
      poseDectector();
      // window.requestAnimationFrame(predictLoop);
    }
  }

  useEffect(() => {
    if (isPredicting) {
      let timerId = setInterval(() => predictLoop(), 1000);
      setPredictIntervalId(timerId);
    } else {
      clearInterval(predictIntervalId);
    }
  }, [isPredicting]);

  function dataGatherLoop() {
    
    if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
      let imageFeatures = tf.tidy(function() {
        let videoFrameAsTensor = tf.browser.fromPixels(VIDEO.current!);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, 
            MOBILE_NET_INPUT_WIDTH], true);
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
      });
      setTrainingDataInputs([...trainingDataInput, imageFeatures]);
      setTrainingDataOutputs([...trainingDataOutputs, gatherDataState]);
      
      let exps = examplesCount;
      // Intialize array index element if currently undefined.
      if (examplesCount[gatherDataState] === undefined) {
        exps[gatherDataState] = 0;
      }

      exps[gatherDataState] += 1;
      setExamplesCount(exps);
  

      setStatus('Class ' + CLASS_NAMES[gatherDataState] + ' data count: ' + examplesCount[gatherDataState] + '. ');
      
      let _animationFrame = [...animationFrame];
      console.log("Gather data state is: ", gatherDataState)
      _animationFrame[gatherDataState] =  window.requestAnimationFrame(dataGatherLoop);
      setAnimationFrame(_animationFrame);
    }

    if (gatherDataState === STOP_DATA_GATHER) {
      stopGatheringData();
    }
    
  }

  
  function gatherDataForClass(classNumber: number) {
    setGatherDataState(classNumber);
  }

  useEffect(() => {
    dataGatherLoop()
  }, [gatherDataState])

  function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  function stopGatheringData() {
    setGatherDataState(STOP_DATA_GATHER);
    // clear both the animation frames
    console.log('calling stop gather data on: ', animationFrame);
    for (let a = 0; a < animationFrame.length; a++) {
      if (animationFrame[a]) window.cancelAnimationFrame(animationFrame[a]);
    }
    setAnimationFrame([undefined, undefined]);
    
  }

  function reset() {
    setIsPredicting(false);
    setExamplesCount([]);
    for (let i = 0; i < trainingDataInput.length; i++) {
      trainingDataInput[i].dispose();
    }
    stopGatheringData();
    setTrainingDataInputs([]);
    setTrainingDataOutputs([]);
    setStatus('No data collected');
    
    console.log('Tensors in memory: ' + tf.memory().numTensors);
  }

  async function poseDectector() {
    const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER });
    const poses = await detector.estimatePoses(VIDEO.current!);
    const leftShoulder = poses[0].keypoints[5];
    const rightShoulder = poses[0].keypoints[6];

    let coordX = 0;
    let coordY = 0;
    if (leftShoulder?.score && leftShoulder.score > 0.5) {
      coordX = leftShoulder.x;
      coordY = leftShoulder.y;
      setXCoordLeft(coordX);
      setYCoordLeft(coordY);
      // console.log('left shoulder deteced at position: (', leftShoulder.x, ', ', leftShoulder.y, ')');
    }
  
    if (rightShoulder?.score && rightShoulder.score > 0.5) {
      coordX = rightShoulder.x;
      coordY = rightShoulder.y;
      setXCoordRight(coordX);
      setYCoordRight(coordY);
      // console.log('right shoulder deteced at position: (', rightShoulder.x, ', ', rightShoulder.y, ')');
    }

    // console.log("scores - right:left", rightShoulder.score, ":", leftShoulder.score);

  }

  async function trainModel() {
    const res = await fetch('/api/trainModel', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/JSON'
      },
      body: JSON.stringify({
        trainingDataInputs: trainingDataInput,
        trainingDataOutputs,
        numberInputs: 2
      })
    });
    const data = await res.json();

    console.log('training data output is: ', trainingDataOutputs);
    console.log('training data input is: ', trainingDataInput);

    console.log(data);
  }
  


  useEffect(() => {
    setModel();
    loadMobileNetFeatureModel();
  }, [])

  const setModel = async () => {
    await tf.ready();
    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: 2, activation: 'softmax'}));
    
    model.summary();
    
    
    // Compile the model with the defined optimizer and specify a loss function to use.
    model.compile({
      // Adam changes the learning rate over time which is useful.
      optimizer: 'adam',
      // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
      // Else categoricalCrossentropy is used if more than 2 classes.
      loss:'binaryCrossentropy', 
      // As this is a classification problem you can record accuracy in the logs too!
      metrics: ['accuracy']  
    });
  }

  useEffect(() => {
    // console.log('arm position is: ', xCoordRight, ', ', yCoordRight)
    pointerRefRight.current!.style.left = xCoordRight.toString() + "px";
    pointerRefRight.current!.style.top = yCoordRight.toString() + "px";

    pointerRefLeft.current!.style.left = xCoordLeft.toString() + "px";
    pointerRefLeft.current!.style.top = yCoordLeft.toString() + "px";

  }, [xCoordRight, yCoordRight, xCoordLeft, yCoordLeft])


  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1>Make your own Teachable Machine using Transfer Learning with MobileNet v3 in TensorFlow.js using saved graph model from TFHub.</h1>
    
      <p id="status">{status}</p>

      <div className='relative w-[640px] h-[480px]'>
        <video ref={VIDEO} id="webcam" className='relative' autoPlay muted></video>
        <div ref={pointerRefRight} className={`bg-red-500 absolute z-10 text-xl w-4 h-4 rounded`}></div>
        <div ref={pointerRefLeft} className={`bg-green-500 absolute z-10 text-xl w-4 h-4 rounded`}></div>
      </div>
      
      <div className='flex flex-wrap gap-4'>
        {!videoPlaying && <button id="enableCam" className=' bg-green-600 text-white rounded px-4 py-2' onClick={enableCam}>Enable Webcam</button>}
        <button className="bg-blue-500 text-white px-4 py-2 rounded" data-1hot="0" onClick={() => gatherDataForClass(0)} data-name="Class 1">Gather Class 1 Data</button>
        <button className="bg-blue-500 text-white px-4 py-2 rounded" data-1hot="1" onClick={() => gatherDataForClass(1)} data-name="Class 2">Gather Class 2 Data</button>
        <button className="bg-blue-500 text-white px-4 py-2 rounded" data-1hot="2" onClick={stopGatheringData} data-name="Class 2">Stop gathering</button>
        <button id="train" onClick={trainAndPredict} className='bg-yellow-500 rounded px-4 py-2'>Train &amp; Predict!</button>
        <button id="train" onClick={() => setIsPredicting(false)} className='bg-yellow-500 rounded px-4 py-2'>Stop predicting</button>
        <button id="reset" onClick={reset} className='bg-red-500 text-white px-4 py-2 rounded'>Reset</button>
      </div>
      
    </main>
  )
}
