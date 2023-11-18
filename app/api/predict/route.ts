import kv from '@vercel/kv';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

export async function POST(request: Request) {
  const body = await request.json();
  const { resizedTensorFrame } = body;
  const model: any = await kv.get('model');
  const mobilenet: tf.GraphModel | null = await kv.get('mobilenet');
  let highestIndex;
  let percent = 0;

  tf.tidy(function() {
    let imageFeatures: any = mobilenet!.predict(resizedTensorFrame.expandDims());
    let prediction = model!.predict(imageFeatures).squeeze();
    highestIndex = prediction.argMax().arraySync();
    let predictionArray = prediction.arraySync();
    percent = predictionArray[highestIndex];
  });
  return Response.json({ data: 'success', highestIndex, percent });
  // poseDectector();
}