import {kv} from '@vercel/kv';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';


const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

function logProgress(epoch: any, logs: any) {
  console.log('Data for epoch ' + epoch, logs);
}

async function loadMobileNetFeatureModel() {
  const URL = 
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  
  const mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  
  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer);
  });

  return mobilenet;
}


export async function POST(request: Request) {
  const body = await request.json();
  console.log('request is: ', body);
  const { trainingDataInputs, trainingDataOutputs, numberInputs} = body;
  await kv.set('trainingDataOutputs', JSON.stringify(trainingDataOutputs));
  await kv.set('trainingDataInputs', JSON.stringify(trainingDataInputs));
  await kv.set('numberInputs', JSON.stringify(numberInputs));
  console.log('kv set');

  const mobilenet = await loadMobileNetFeatureModel();
  
  console.log('mobilenet is: ', mobilenet)
  await kv.set('mobilenet', (mobilenet));

  console.log('mobilenet set in kv');

  let model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: numberInputs, activation: 'softmax'}));

  model.summary();


  // Compile the model with the defined optimizer and specify a loss function to use.
  model.compile({
    // Adam changes the learning rate over time which is useful.
    optimizer: 'adam',
    // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
    // Else categoricalCrossentropy is used if more than 2 classes.
    loss: (numberInputs === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy', 
    // As this is a classification problem you can record accuracy in the logs too!
    metrics: ['accuracy']  
  });

  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  let oneHotOutputs = tf.oneHot(outputsAsTensor, numberInputs);
  let inputsAsTensor = tf.stack(trainingDataInputs);
  
  let results = await model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epochs: 10, 
      callbacks: {onEpochEnd: logProgress} });
  
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  await kv.set('model', model);
  return Response.json({ data: 'success' });
}