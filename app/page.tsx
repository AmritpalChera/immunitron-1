"use client";

import Image from 'next/image'
import { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;


export default function Home() {
  
  // states
  const STOP_DATA_GATHER = -1;
  const CLASS_NAMES = ["1", "2"];

  const VIDEO = useRef<HTMLVideoElement>(null);
 

  const [gatherDataState, setGatherDataState] = useState(STOP_DATA_GATHER);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [trainingDataInput, setTrainingDataInputs] = useState<any>([]);
  const [tainingDataOutputs, setTrainingDataOutputs] = useState<any>([]);
  const [examplesCount, setExamplesCount] = useState<any>([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [status, setStatus] = useState('Awaiting TF.js load');
  const [mobilenet, setMobilenet] = useState<any>(undefined);
  const [animationFrame, setAnimationFrame] = useState<any>([undefined, undefined]);

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
      setTrainingDataOutputs([...tainingDataOutputs, gatherDataState]);
      
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

  async function trainModel() {
    const res = await fetch('/api/trainModel', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/JSON'
      },
      body: JSON.stringify({
        tensorData: ['1', '1']
      })
    });
    const data = await res.json();

    console.log('training data output is: ', tainingDataOutputs);
    console.log('training data input is: ', trainingDataInput);

    console.log(data);
  }
  


  useEffect(() => {
    loadMobileNetFeatureModel();
  }, [])

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1>Make your own Teachable Machine using Transfer Learning with MobileNet v3 in TensorFlow.js using saved graph model from TFHub.</h1>
    
      <p id="status">{status}</p>
    
      <video ref={VIDEO} id="webcam" autoPlay muted></video>
      <div className='flex flex-wrap gap-4'>
        {!videoPlaying && <button id="enableCam" className=' bg-green-600 text-white rounded px-4 py-2' onClick={enableCam}>Enable Webcam</button>}
        <button className="bg-blue-500 text-white px-4 py-2 rounded" data-1hot="0" onClick={() => gatherDataForClass(0)} data-name="Class 1">Gather Class 1 Data</button>
        <button className="bg-blue-500 text-white px-4 py-2 rounded" data-1hot="1" onClick={() => gatherDataForClass(1)} data-name="Class 2">Gather Class 2 Data</button>
        <button className="bg-blue-500 text-white px-4 py-2 rounded" data-1hot="2" onClick={stopGatheringData} data-name="Class 2">Stop gathering</button>
        <button id="train" onClick={trainModel} className='bg-yellow-500 rounded px-4 py-2'>Train &amp; Predict!</button>
        <button id="reset" onClick={reset} className='bg-red-500 text-white px-4 py-2 rounded'>Reset</button>
      </div>
      
    </main>
  )
}
