using Microsoft.ML;

namespace Transaction_Anomaly_Detection.Models
{
    public class ModelBuilder
    {
        public static ITransformer BuildModel(MLContext mlContext, IDataView trainingData)
        {
            // Define the pipeline
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("DESCRIPTION")
                           .Append(mlContext.Transforms.Categorical.OneHotEncoding("ITEMNAME"))
                           .Append(mlContext.Transforms.NormalizeMinMax("UNITCOST"))
                           .Append(mlContext.Transforms.NormalizeMinMax("QUANTITY"))
                           .Append(mlContext.Transforms.NormalizeMinMax("STOCKONHAND"))
                           .Append(mlContext.Transforms.NormalizeMinMax("RUNNINGUNITCOST"))
                           .Append(mlContext.Transforms.Concatenate("Features", "DESCRIPTION", "ITEMNAME",
                               "UNITCOST", "QUANTITY", "STOCKONHAND", "RUNNINGUNITCOST"))
                           .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            // Train the model
            var model = pipeline.Fit(trainingData);

            return model;
        }
    }
}
