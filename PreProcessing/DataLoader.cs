using Microsoft.ML;
using Microsoft.ML.Data;

namespace Transaction_Anomaly_Detection.PreProcessing
{
    public class DataLoader
    {
        public static IDataView LoadData(MLContext mlContext, string dataPath)
        {
            var dataView = mlContext.Data.LoadFromTextFile<TransactionData>(
                path: dataPath,
                separatorChar: ',',
                hasHeader: true,
                allowQuoting: true);

            return dataView;
        }
        public class TransactionData
        {
            [LoadColumn(0)] public string DATE { get; set; }
            [LoadColumn(1)] public string DESCRIPTION { get; set; }
            [LoadColumn(2)] public string ITEMNAME { get; set; }
            [LoadColumn(3)] public float QUANTITY { get; set; }
            [LoadColumn(4)] public float UNITCOST { get; set; }
            [LoadColumn(5)] public float STOCKONHAND { get; set; }
            [LoadColumn(6)] public float RUNNINGUNITCOST { get; set; }
        }
        public class TransactionPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool IsFaulty { get; set; }
        }
    }
}
