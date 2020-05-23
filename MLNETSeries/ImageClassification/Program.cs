using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;
using ImageClassification.Models;

namespace ImageClassification
{
    public class Program
    {
        public static void Main(string[] args)
        {
            //Location of assets
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            var assetsRelativePath = Path.Combine(projectDirectory, "assets");

            //Initialize Model
            MLContext mlContext = new MLContext();

            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Label",
                outputColumnName: "LabelAsKey")
                    .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: assetsRelativePath,
                    inputColumnName: "ImagePath"));

            //Fit Data
            IDataView preProcessedData = preprocessingPipeline
                    .Fit(shuffledData)
                    .Transform(shuffledData);

            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            //Assign partitions

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;

            //Pipeline
            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true,
                WorkspacePath = workspaceRelativePath
            };
            //Define Estimator Chain
            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //Fit Model
            ITransformer trainedModel = trainingPipeline.Fit(trainSet);

            //Classify Single Image
            ClassifySingleImage(mlContext, testSet, trainedModel);

            //Classify Multiple Images
            //ClassifyImages(mlContext, testSet, trainedModel);

            Console.ReadLine();
        }

        //Use Model
        private static void OutputPrediction(ModelOutput prediction)
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        }

        //Predict Multiple Images
        public static void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            //Create DataView
            IDataView predictionData = trainedModel.Transform(data);
            IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);

            Console.WriteLine("Classifying multiple images");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }
        }

        //Predict Single Image
        public static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            //Create Prediction Engine
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
            //Access single Input Model
            ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
            //ModelInput input2 = new ModelInput { Label = "UD",ImagePath = @"C:\Users\slipk\Desktop\DeepLearning_ImageClassification_Binary\DeepLearning_ImageClassification_Binary\assets\UD\7001-1.jpg"};

            //Make prediction
            ModelOutput prediction = predictionEngine.Predict(image);
            Console.WriteLine("Classifying single image . . .");
            OutputPrediction(prediction);


        }


        //Load Data
        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }
                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }

    }
}
