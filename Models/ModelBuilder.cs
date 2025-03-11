using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Transaction_Anomaly_Detection.Models
{
    public class ModelBuilder
    {
        public static ITransformer BuildModel(MLContext mlContext, IDataView trainingData)
        {
            // Define the pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", "DescriptionEncoded", "NormalizedUNITCOST", "NormalizedQUANTITY", "NormalizedSTOCKONHAND", "NormalizedRUNNINGUNITCOST")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "IsFaulty"))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            // Train the model
            var model = pipeline.Fit(trainingData);

            return model;
        }
    }
}
