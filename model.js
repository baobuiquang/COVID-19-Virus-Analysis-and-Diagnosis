// Classifier Variable
let classifier;
// Model URL
let imageModelURL = 'https://teachablemachine.withgoogle.com/models/V4ZNTcbKF/';
// Video
let video;
let flippedVideo;
// To store the classification
let label = "Waiting for capture device";
let label1 = "Waiting for capture device";
let label2 = "Waiting for capture device";
let label3 = "Waiting for capture device";
let confidence = 0;
let confidence1 = 0.0;
let confidence2 = 0.0;
let confidence3 = 0.0;
// Some variables
var w = window.innerWidth;
var h = window.innerHeight;
var videoWidth = w * 0.3;
var videoHeight = videoWidth / 4 * 3;
var lineHeight = 32;
// Load the model first
function preload() {
    classifier = ml5.imageClassifier(imageModelURL + 'model.json');
}

var canvas;
function setup() {
    canvas = createCanvas(w, h);
    canvas.position(0, 0);
    canvas.style('z-index', '-1');

    // Create the video
    video = createCapture(VIDEO);
    video.size(videoWidth, videoHeight);
    video.hide();

    // flippedVideo = ml5.flipImage(video);
    // Start classifying
    classifyVideo();
}

function draw() {
    background(0);
    // Draw the video


    rect(w / 2 - videoWidth - 55, h / 2 - videoHeight / 2 - 5, videoWidth + 9, videoHeight + 9);
    image(flippedVideo, w / 2 - videoWidth - 50, h / 2 - videoHeight / 2);


    stroke('transparent');
    // Vertically
    rect(w / 2 - videoWidth / 8 * 1 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 2 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 3 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 4 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 5 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 6 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 7 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    // Horizontally
    rect(w / 2 - videoWidth - 50, h / 2 + videoHeight / 6 * 1, videoWidth, 0);
    rect(w / 2 - videoWidth - 50, h / 2 + videoHeight / 6 * 2, videoWidth, 0);
    rect(w / 2 - videoWidth - 50, h / 2 + videoHeight / 6 * 0, videoWidth, 0);
    rect(w / 2 - videoWidth - 50, h / 2 - videoHeight / 6 * 1, videoWidth, 0);
    rect(w / 2 - videoWidth - 50, h / 2 - videoHeight / 6 * 2, videoWidth, 0);
    stroke('black');

    // Draw the label
    fill(255);
    textSize(22);
    textAlign(LEFT);

    text(label, w / 2, h / 2 - videoHeight / 2 + lineHeight);
    text(confidence.toFixed(5) + "%", w / 2, h / 2 - videoHeight / 2 + lineHeight * 2);
    rect(w / 2, h / 2 - videoHeight / 2 + lineHeight * 2.5, confidence * videoWidth / 100, lineHeight / 2);

    text(label1, w / 2, h / 2 - videoHeight / 2 + lineHeight * 4.5);
    text(confidence1.toFixed(5) + "%", w / 2, h / 2 - videoHeight / 2 + lineHeight * 5.5);
    rect(w / 2, h / 2 - videoHeight / 2 + lineHeight * 6, confidence1 * videoWidth / 100, lineHeight / 2);

    text(label2, w / 2, h / 2 - videoHeight / 2 + lineHeight * 8);
    text(confidence2.toFixed(5) + "%", w / 2, h / 2 - videoHeight / 2 + lineHeight * 9);
    rect(w / 2, h / 2 - videoHeight / 2 + lineHeight * 9.5, confidence2 * videoWidth / 100, lineHeight / 2);

    textSize(14);
    textAlign(LEFT);
    text("Important Notice: Keep the X-ray image filling the whole frame", w / 2 - videoWidth - 50, h / 2 - videoHeight / 2 - 20);

    textSize(18);
    textAlign(CENTER);
    text("Diagnosis from Radiography", width / 2, height - lineHeight * 3.5);
    textSize(36);
    text(label, width / 2, height - lineHeight * 2);

}

// Get a prediction for the current video frame
function classifyVideo() {
    flippedVideo = ml5.flipImage(video);
    classifier.classify(flippedVideo, gotResult);
    flippedVideo.remove();
}

// When we get a result
function gotResult(error, results) {
    // If there is an error
    if (error) {
        console.error(error);
        return;
    }
    // The results are in an array ordered by confidence.
    // console.log(results[0]);
    label = results[0].label;
    label1 = results[1].label;
    label2 = results[2].label;
    // label3 = results[3].label;
    confidence = results[0].confidence * 100;
    confidence1 = results[1].confidence * 100;
    confidence2 = results[2].confidence * 100;
    // confidence3 = results[3].confidence;
    // Classifiy again!
    classifyVideo();
}